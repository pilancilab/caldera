import torch
import torch.nn as nn
from dataclasses import dataclass, field

from lib import codebook
from lib.utils import clean, dtype_from_str, get_hadK
from lib.linear import QuantizedLinear

import quiptools_cuda
from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda


class LatticeQuantizedParameter(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        idxs,
        scale,
        codebook_version,
        transposed=False
    ):
        super(LatticeQuantizedParameter, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.transposed = transposed
        self.scale = scale

        self.codebook_version = codebook_version
        self.codebook = codebook.get_codebook(codebook_version).to(idxs.device)

        idxs_dev = idxs.device
        self.idxs = idxs.cpu()
        codebook_class = codebook.get_quantized_class(
            codebook.get_id(codebook_version)
        )(self.idxs.device)

        split_idxs = codebook_class.maybe_unpack_idxs(
            self.idxs
        )
        self.idxs_list = []
        for i in range(len(split_idxs)):
            self.register_buffer(f'idxs_{i}', split_idxs[i].to(idxs_dev))
            exec(f'self.idxs_list.append(self.idxs_{i})')

    def forward(self, x, float_precision=False):
        dtype = x.dtype
        n = self.in_features
        m = self.out_features
        self.idxs_list[0] = self.idxs_list[0].to(x.device)
        if not float_precision:
            x = x.half()
        else:
            x = x.float()
        if self.codebook_version == 'E8P12':
            if x.size(0) == 1 and not self.transposed and not float_precision:
                x = quiptools_cuda.decode_matvec_e8p(
                    x[0].to(torch.float16),
                    self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs)
            else:
                W_decompressed = quiptools_cuda.decompress_packed_e8p(
                    self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs)
                if float_precision:
                    W_decompressed = W_decompressed.float()
                if self.transposed:
                    x = (x @ W_decompressed)
                else:
                    x = (x @ W_decompressed.T)

        elif self.codebook_version == 'E8P12RVQ3B':
            resid_scale = self.codebook.opt_resid_scale
            x16 = x.to(torch.float16)
            if x.shape[0] == 1 and not self.transposed and not float_precision:
                x_padded = torch.zeros(
                    8, x16.shape[1], dtype=torch.float16, device=x16.device)
                x_padded[0] = x16[0]
                z = torch.zeros(
                    8, m, dtype=torch.float16, device=x_padded.device)
                quiptools_cuda.lookupmatmul_e81b_k8(
                    x_padded / resid_scale, self.idxs_list[1],
                    self.codebook.e81b_grid, z
                )

                x = quiptools_cuda.decode_matvec_e8p(
                    x16[0], self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs) + z[0]

            else:
                W_decompressed = quiptools_cuda.decompress_packed_e8p(
                    self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs)

                W_resid_decompressed = torch.zeros(
                    self.idxs_list[1].shape[0],
                    64 * self.idxs_list[1].shape[-1],
                    device=self.idxs_list[1].device, dtype=torch.float16
                )

                quiptools_cuda.decompress_e81b_packed(
                    self.idxs_list[1], self.codebook.e81b_grid,
                    W_resid_decompressed
                )

                if float_precision:
                    W_decompressed = W_decompressed.float()                
                    W_resid_decompressed = W_resid_decompressed.float()

                if self.transposed:
                    x = (x @ (W_decompressed +
                                W_resid_decompressed / resid_scale))
                else:
                    x = (x @ (W_decompressed +
                                W_resid_decompressed / resid_scale).T)
        else:
            resid_scale = self.codebook.opt_resid_scale
            if x.size(0) == 1 and not self.transposed and not float_precision:
                x16 = x[0].to(torch.float16)
                x = (quiptools_cuda.decode_matvec_e8p(
                    x16, self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs) +
                     quiptools_cuda.decode_matvec_e8p(
                         x16 / resid_scale, self.idxs_list[1].view(
                             m // 16, n // 64, 8, 4),
                         self.codebook.grid_packed_abs))
            else:
                W_decompressed = quiptools_cuda.decompress_packed_e8p(
                    self.idxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs) + \
                        quiptools_cuda.decompress_packed_e8p(
                            self.idxs_list[1].view(m // 16, n // 64, 8, 4),
                            self.codebook.grid_packed_abs
                        ) / resid_scale
                if float_precision:
                    W_decompressed = W_decompressed.float()
                if self.transposed:
                    x = (x @ W_decompressed)
                else:
                    x = (x @ W_decompressed.T)
        x *= self.scale
        return x.to(dtype)


class LPLRQuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        Q_codebook_version,
        L_codebook_version,
        R_codebook_version,
        L, R,
        L_idxs, R_idxs, Q_idxs,
        L_scale, R_scale, Q_scale,
        global_scale,
        scaleWH,
        hadamard,
        SU, SV,
        rank=64,
        ft_rank=64,
        grad_ckpt=True
    ):
        super(LPLRQuantizedLinear, self).__init__()

        self.rank = rank
        self.ft_rank = ft_rank

        self.in_features = in_features
        self.out_features = out_features

        self.hadamard = hadamard
        self.global_scale = global_scale
        self.scaleWH = scaleWH
        if self.scaleWH is not None:
            self.scaleWH = nn.Parameter(self.scaleWH, requires_grad=False)

        self.Q = LatticeQuantizedParameter(
            in_features=in_features,
            out_features=out_features,
            idxs=Q_idxs,
            scale=Q_scale,
            codebook_version=Q_codebook_version
        )

        self.L_idxs = L_idxs
        self.R_idxs = R_idxs
        self.L_codebook_version = L_codebook_version
        self.R_codebook_version = R_codebook_version
        self.L_scale = L_scale
        self.R_scale = R_scale

        self.split_L_and_R_for_LoRA(ft_rank, L, R)

        self.SU = nn.Parameter(SU.bfloat16(), requires_grad=True)
        self.SV = nn.Parameter(SV.bfloat16(), requires_grad=True)

        had_left, K_left = get_hadK(in_features)
        had_right, K_right = get_hadK(out_features)

        self.had_left = nn.Parameter(had_left, requires_grad=False)
        self.had_right = nn.Parameter(had_right, requires_grad=False)

        self.K_left = K_left
        self.K_right = K_right

        self.grad_ckpt = grad_ckpt

    def split_L_and_R_for_LoRA(self, ft_rank, L, R):
        if ft_rank > 0:
            self.L_ft = nn.Parameter(
                L[:, :ft_rank].bfloat16(), requires_grad=True)
            self.R_ft = nn.Parameter(
                R[:ft_rank, :].bfloat16(), requires_grad=True)
            assert self.L_ft != [] and self.R_ft != []

        if self.rank > ft_rank:
            if self.L_codebook_version is not None:
                self.L = LatticeQuantizedParameter(
                    in_features=self.out_features,
                    out_features=self.rank - ft_rank,
                    idxs=self.L_idxs[ft_rank:, :],
                    scale=self.L_scale,
                    codebook_version=self.L_codebook_version,
                    transposed=True
                )
                self.quant_L = True
            else:
                self.L = nn.Parameter(L[:, ft_rank:].bfloat16(), requires_grad=False)
                self.quant_L = False

            if self.R_codebook_version is not None:
                self.R = LatticeQuantizedParameter(
                    in_features=self.in_features,
                    out_features=self.rank - ft_rank,
                    idxs=self.R_idxs[ft_rank:, :],
                    scale=self.R_scale,
                    codebook_version=self.R_codebook_version
                )
                self.quant_R = True
            else:
                self.R = nn.Parameter(R[ft_rank:, :].bfloat16(), requires_grad=False)
                self.quant_R = False
        else:
            self.L = None
            self.R = None

    def forward(self, input):
        # TODO: implement checkpoint
        return self.no_ckpt_forward(input)

    def no_ckpt_forward(self, x):
        old_dtype = x.dtype
        shape = x.shape
        n, m = len(self.SU), len(self.SV)
        x = x.bfloat16().view(-1, n)
        # x = x.float().view(-1, n)
        # Preprocessing
        x = x * self.SU 
        if self.scaleWH is not None:
            x /= self.scaleWH
        if self.hadamard:
            x = matmul_hadUt_cuda(x, self.had_left, self.K_left)

        # Apply Q
        output_no_ft = self.Q.forward(x)

        # Apply quantized L and R
        if self.L is not None:
            if self.quant_R:
                xR = self.R.forward(x, float_precision=True)
            else:
                xR = x @ self.R.T

            if self.quant_L:
                output_no_ft += self.L.forward(xR, float_precision=True)
            else:
                output_no_ft += xR @ self.L.T

        # Apply LoRA factors
        if self.ft_rank > 0:
            output = output_no_ft + x @ self.R_ft.T @ self.L_ft.T
        else:
            output = output_no_ft

        if self.hadamard:
            output = matmul_hadU_cuda(output, self.had_right, self.K_right) * self.global_scale
        
        output = output * self.SV
        if self.scaleWH is not None:
            output *= self.scaleWH

        return output.view(*shape[:-1], m).to(old_dtype)

    def compare_outputs(self, input, W_hat):
        output = self.no_ckpt_forward(input)
        comparison = input @ W_hat.T
        return (torch.linalg.matrix_norm(output - comparison, ord='fro') /
                torch.linalg.matrix_norm(comparison, ord='fro')).mean().item()
