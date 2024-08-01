from lib.utils.math_utils import block_LDL
import lib.algo.quip as quip
from lib import codebook

import torch
from tqdm import tqdm

from caldera.decomposition.dataclasses import *
from caldera.utils.quantization import QuantizerFactory

from collections import namedtuple
from copy import deepcopy


# Maps number of bits to name of the QuIP# lattice quantizer
BITS_TO_CODEBOOK = {
    2: 'E8P12',
    3: 'E8P12RVQ3B',
    4: 'E8P12RVQ4B'
}

def caldera(
    quant_params: CalderaParams,
    W: torch.Tensor,
    H: torch.Tensor = None,
    device: str = "cuda",
    use_tqdm: bool = True,
    scale_W: bool = True,
):
    """
    Runs the CALDERA algorithm, to decompose a weight matrix into Q + LR, where
    Q is full-rank, L and R are low-rank factors, and all matrices are in a low-
    precision format.
    """
    # scaling
    if scale_W:
        global_scale = W.square().mean().sqrt().item()
    else:
        global_scale = 1
    W = W / global_scale

    if H is None:
        H = torch.eye(W.shape[1]).to(device)
    
    # Compute the symmetric square root of H, because the data-aware
    # objective can be formulated as
    # min_{L, R} ||(W - LR - Q)H^{1/2}||_F^2.
    EigTuple = namedtuple("EigTuple", ["eigenvalues", "eigenvectors"])
    if not quant_params.activation_aware_LR and not quant_params.activation_aware_Q \
            and not quant_params.full_quip_sharp:
        H_sqrt = H
        eigH = EigTuple(torch.ones(W.shape[1]).to(device), H)
    else:
        eigH = torch.linalg.eigh(H)

        eigvals = eigH.eigenvalues
        # if eigvals.min() < quant_params.quip_args.sigma_reg:
        #     H = H + (quant_params.quip_args.sigma_reg - eigvals.min()) * \
        #             torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        #     eigvals += quant_params.quip_args.sigma_reg - eigvals.min()
        #     eigH = EigTuple(eigvals, eigH.eigenvectors)
            
        H_sqrt = (eigH.eigenvectors @
                    torch.diag(torch.sqrt(eigvals)) @
                    eigH.eigenvectors.T)
    
    # Initialization and Hadamard transform
    best_decomp = CalderaDecomposition(
        Q=torch.zeros_like(W).float(),
        L=torch.zeros(W.shape[0], quant_params.rank).to(device),
        R=torch.zeros(quant_params.rank, W.shape[1]).to(device))

    if quant_params.hadamard_transform:
        _, H, W, SU, SV, scaleWH = quip.incoherence_preprocess(
            H, W, quant_params.quip_args
        )
        best_decomp.SU = SU.to(W.device)
        best_decomp.SV = SV.to(W.device)
        best_decomp.scaleWH = scaleWH

        eigH = torch.linalg.eigh(H)
        H_sqrt = (eigH.eigenvectors @
                    torch.diag(torch.sqrt(eigH.eigenvalues)) @
                    eigH.eigenvectors.T)
    else:
        best_decomp.scaleWH = None
        best_decomp.SU = torch.ones(W.shape[1]).to(W.dtype).to(W.device)
        best_decomp.SV = torch.ones(W.shape[0]).to(W.dtype).to(W.device)

    best_decomp.W = W.cpu()
    errors = {}
    for mtx in quant_params.update_order:
        errors[mtx] = []

    min_error = float('inf')
    curr_decomp = deepcopy(best_decomp)

    updated = {mtx: False for mtx in quant_params.update_order}
    
    to_iter = range(quant_params.iters)
    if use_tqdm:
        to_iter = tqdm(to_iter)
    for _ in to_iter:
        for mtx in quant_params.update_order:
            if mtx == "LR":
                maybe_update_LR(curr_decomp, quant_params, W, H_sqrt, eigH, device)
            elif mtx == "Q":
                maybe_update_Q(curr_decomp, quant_params, W, H, device)
            updated[mtx] = True
            
            errors[mtx].append(
                activation_aware_error(W, H, curr_decomp, device)
            )
            if errors[mtx][-1] < min_error and all(updated.values()):
                min_error = errors[mtx][-1]
                best_decomp = deepcopy(curr_decomp)
    best_decomp.errors = errors

    # Update scales
    best_decomp.global_scale = global_scale
    return best_decomp


def activation_aware_error(
    W: torch.Tensor,
    H: torch.Tensor,
    caldera_info: CalderaDecomposition, 
    device: str
):
    """
    Computes the activation-aware loss for a sublayer as
    tr((W - W_hat) H (W - W_hat).T) / tr(W H^1/2),
    where H^1/2 is the symmetric square root.
    """

    W = W.to(device).float()
    W_hat = caldera_info.Q + caldera_info.L @ caldera_info.R
    W_hat *= caldera_info.global_scale

    error = (torch.trace((W_hat - W) @ H @ (W_hat - W).T) / 
                torch.trace(W @ H @ W.T)).sqrt().item()
    return error


def get_quant_info(
    use_lattice_quant: bool,
    quant_factory: QuantizerFactory,
    bits: int,
    device: str
):
    cb = None
    quantizer = None
    if use_lattice_quant:
        cb = codebook.get_codebook(BITS_TO_CODEBOOK[bits]).to(device)
    else:
        quantizer = quant_factory.get_quantizer(bits, device)
    return QuantInfo(
        lattice_quant=use_lattice_quant,
        lattice_cb=cb,
        quant=quantizer
    )


def quantize_matrix(
    A, quant_params,
    quant_info: QuantInfo = None
):
    QuantReturn = namedtuple(
        'QuantReturn', ['A_hat', 'A_idxs', 'scale']
    )
    if not quant_info.lattice_quant:
        quant_info.quant.block_size = A.shape[0] * A.shape[1]
        A_idxs, scales, shape = quant_info.quant.quantize_block(A)
        A_hat = quant_info.quant.dequantize_block(A_idxs, scales, shape)
        return QuantReturn(A_hat, A_idxs, scales)

    # Scale before quantization, as in QuIP#
    scale = A.square().mean().sqrt().item()

    m, n = A.shape

    A = A.reshape(-1, quant_info.lattice_cb.codesz).clone() / scale
    A_idxs = torch.zeros(
        m * n // quant_info.lattice_cb.codesz,
        dtype=quant_info.lattice_cb.idx_dtype,
        device=A.device
    )
    K = quant_params.lattice_quant_block_size
    for i in range(0, A.shape[0], K):
        A[i:i+K], A_idxs[i:i+K] = \
            quant_info.lattice_cb.quantize(A.float()[i:i+K])
    A = A.reshape(m, n)
    A_idxs = A_idxs.reshape(m, n // quant_info.lattice_cb.codesz)

    A = A * scale

    A_idxs = quant_info.lattice_cb.maybe_pack_idxs(A_idxs)
    return QuantReturn(A, A_idxs, scale)


def maybe_update_Q(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    W: torch.Tensor,
    H: torch.Tensor,
    device: str
):

    if quant_params.compute_quantized_component:
        residual = W - caldera_info.L @ caldera_info.R
        if not quant_params.compute_low_rank_factors:
            residual = W
        if quant_params.activation_aware_Q:
            update_Q_data_aware(caldera_info, quant_params, H, residual, device)
        else:
            update_Q_non_data_aware(caldera_info, quant_params, residual, device)


def update_Q_non_data_aware(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    residual: torch.Tensor,
    device: str
):
    quant_info = get_quant_info(
        use_lattice_quant=quant_params.lattice_quant_Q,
        quant_factory=quant_params.quant_factory_Q,
        bits=quant_params.Q_bits,
        device=device
    )

    quant_return = quantize_matrix(residual, quant_params, quant_info)
    caldera_info.Q = quant_return.A_hat
    caldera_info.Q_idxs = quant_return.A_idxs
    caldera_info.Q_scale = quant_return.scale


def update_Q_data_aware(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    H: torch.Tensor,
    residual: torch.Tensor,
    device: str
):
    """
    Performs an LDLQ update on the residual (W - LR)
    """

    # Scale the residual, as done in the quantize_linear function of QuIP#
    scale = residual.square().mean().sqrt().item()
    residual /= scale

    codebook_str = BITS_TO_CODEBOOK[quant_params.Q_bits]
    cb = codebook.get_codebook(codebook_str).to(residual.device)

    if quant_params.compute_low_rank_factors and \
            quant_params.Q_hessian_downdate:

        M = torch.linalg.cholesky(H)
        if quant_params.rand_svd:
            _, _, V = torch.svd_lowrank(
                caldera_info.L @ caldera_info.R @ M,
                quant_params.rank * 3, niter=10)
            V = V[:, :quant_params.rank]
        else:
            _, _, Vh = torch.linalg.svd(
                caldera_info.L @ caldera_info.R @ M, full_matrices=False)
            V = Vh.T[:, :quant_params.rank]

        H = H - (M @ V @ V.T @ M.T).to(H.dtype)
        min_eigval = torch.linalg.eigh(H).eigenvalues.min()
        H = H + (quant_params.quip_args.sigma_reg2 + max(-min_eigval, 0)) * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        alpha = torch.diag(H).mean().abs() * quant_params.quip_args.sigma_reg2
        print(alpha, min_eigval.abs())
        H = H + alpha * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    if quant_params.full_quip_sharp:
        assert not quant_params.hadamard_transform, \
            ("Full QuIP# incompatible with performing Hadamard transform "
                "on our end")
        assert quant_params.rank == 0 or \
            not quant_params.compute_low_rank_factors, \
            ("Full QuIP# incompatible with separately computing low-rank "
            "factors.")

        caldera_info.Q, attr = quip.quantize(
            H_orig=H,
            W_orig=residual,
            rank=0,
            codebook_orig=cb,
            args=quant_params.quip_args,
            device=device
        )
        caldera_info.Q_idxs = attr['Qidxs'].to(device)

        caldera_info.scaleWH = attr['scaleWH']
        caldera_info.SU = attr['SU']
        caldera_info.SV = attr['SV']
        if quant_params.quip_args.lora_rank != 0:
            caldera_info.L = attr['A'].to(device) / caldera_info.SV[0].abs().sqrt()
            caldera_info.R = attr['B'].to(device) / caldera_info.SV[0].abs().sqrt()
            caldera_info.L_scale = scale
            caldera_info.R_scale = scale
            caldera_info.Q -= caldera_info.L @ caldera_info.R

    else:
        # Just do LDLQ
        block_LDL_out = block_LDL(H, cb.codesz)
        assert block_LDL_out is not None

        L, D = block_LDL_out
        del block_LDL_out

        scale /= cb.opt_scale
        residual *= cb.opt_scale

        if quant_params.quip_args.no_use_buffered:
            Q, Qidxs = quip.LDLQ(
                residual, H, L, D, cb, quant_params.quip_args)
        elif quant_params.quip_args.lowmem_ldlq or \
                quant_params.quip_args.use_fp64:
            Q, Qidxs = quip.LDLQ_buffered_lowmem(
                residual, H, L, D, cb, quant_params.quip_args,
                buf_cols=128)
        else:
            Q, Qidxs = quip.LDLQ_buffered(
                residual, H, L, D, cb, quant_params.quip_args,
                buf_cols=128)
        caldera_info.Q_idxs = Qidxs
        caldera_info.Q = Q
        caldera_info.Q_idxs = cb.maybe_pack_idxs(caldera_info.Q_idxs)

    caldera_info.Q_scale = scale
    caldera_info.Q *= scale


def LR_init(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    H_sqrt: torch.Tensor,
    eigH: torch.Tensor,
    residual: torch.Tensor
):
    """
    Runs rank-constrained regression to minimize
    ||(residual - LR) eigH||_F^2
    over L, R in closed-form.
    """
    if quant_params.activation_aware_LR:
        Y = residual @ H_sqrt @ eigH.eigenvectors
        if quant_params.rand_svd:
            q = min(quant_params.rank*2, min(*caldera_info.W.shape))
            U, Sigma, V = torch.svd_lowrank(Y, q)
            Vh = V.T
        else:
            U, Sigma, Vh = torch.linalg.svd(Y, full_matrices=False)

        L = U[:, :quant_params.rank]
        R = torch.diag(Sigma[:quant_params.rank]) @ \
            Vh[:quant_params.rank, :] @ \
            torch.diag(1 / eigH.eigenvalues.sqrt()) @ eigH.eigenvectors.T
    else:
        if quant_params.rand_svd:
            q = min(quant_params.rank*2,
                    min(*caldera_info.W.shape))
            U, Sigma, V = torch.svd_lowrank(residual, q)
            Vh = V.T
        else:
            U, Sigma, Vh = torch.linalg.svd(residual, full_matrices=False)
        L = U[:, :quant_params.rank] @ \
            torch.diag(Sigma[:quant_params.rank].sqrt())
        R = torch.diag(Sigma[:quant_params.rank].sqrt()) @ \
            Vh[:quant_params.rank, :]
    return L, R
    
def maybe_update_LR(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    W: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device
):
    if quant_params.compute_low_rank_factors:
        residual = W - caldera_info.Q
        update_LR(caldera_info, quant_params, residual, H_sqrt, eigH, device)


def update_LR(
    caldera_info: CalderaDecomposition,
    quant_params: CalderaParams,
    residual: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device
):
    """
    Run LPLR on the residual (W - Q)
    """
    data_aware = quant_params.activation_aware_LR

    # Initialization of L, R
    L, R = LR_init(caldera_info, quant_params, H_sqrt, eigH, residual)

    if quant_params.L_bits < 16 or quant_params.R_bits < 16:
        quant_info_L = get_quant_info(
            use_lattice_quant=quant_params.lattice_quant_LR,
            quant_factory=quant_params.quant_factory_LR,
            bits=quant_params.L_bits,
            device=device
        )
        quant_info_R = get_quant_info(
            use_lattice_quant=quant_params.lattice_quant_LR,
            quant_factory=quant_params.quant_factory_LR,
            bits=quant_params.R_bits,
            device=device
        )
        
        best_L, best_R = L, R
        best_L_quant_out, best_R_quant_out = None, None
        best_error = float('inf')

        for _ in range(quant_params.lplr_iters):
            # L
            if data_aware:
                L = torch.linalg.lstsq((R @ H_sqrt).T, (residual @ H_sqrt).T)[0].T
                if torch.isnan(L).any():
                    L = (residual @ H_sqrt) @ torch.linalg.pinv(R @ H_sqrt)
            else:
                L = torch.linalg.lstsq(R.T, residual.T)[0].T
                if torch.isnan(R).any():
                    L = residual @ torch.linalg.pinv(R)

            quant_out_L = quantize_matrix(L.T, quant_params, quant_info_L)
            L = quant_out_L.A_hat.T
            
            # R
            R = torch.linalg.lstsq(L, residual)[0]
            if torch.isnan(R).any():
                R = torch.linalg.pinv(L) @ residual

            quant_out_R = quantize_matrix(R, quant_params, quant_info_R)
            R = quant_out_R.A_hat

            error = torch.linalg.matrix_norm((residual - L @ R) @ H_sqrt) #/ \
                    #  torch.linalg.matrix_norm((residual + caldera_info.Q) @ H_sqrt)
            if error < best_error:
                best_L, best_R = L, R
                best_L_quant_out = quant_out_L
                best_R_quant_out = quant_out_R
                best_error = error

        caldera_info.L_idxs = best_L_quant_out.A_idxs
        caldera_info.R_idxs = best_R_quant_out.A_idxs
        caldera_info.L_scale = best_L_quant_out.scale
        caldera_info.R_scale = best_R_quant_out.scale

        L, R = best_L, best_R

    caldera_info.L = L
    caldera_info.R = R