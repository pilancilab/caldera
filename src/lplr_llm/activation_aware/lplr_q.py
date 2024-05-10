from lib.utils.math_utils import block_LDL
import lib.algo.quip as quip
from lib import codebook

import torch
from tqdm import tqdm

from lplr_llm.activation_aware.dataclasses import *
from lplr_llm.utils.quantization import QuantizerFactory

from collections import namedtuple
from copy import deepcopy

from lib.utils.matmul_had import matmul_hadU, matmul_hadUt

# Maps number of bits to name of the QuIP# lattice quantizer
BITS_TO_CODEBOOK = {
    2: 'E8P12',
    3: 'E8P12RVQ3B',
    4: 'E8P12RVQ4B'
}

def LPLRq(
    quant_params: ActivationAwareQuantParams,
    W: torch.Tensor,
    H: torch.Tensor = None,
    device: str = "cuda",
    use_tqdm: bool = True,
    scale_W: bool = True,
):
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
    if not quant_params.activation_aware_LR and not quant_params.activation_aware_Q \
            and not quant_params.full_quip_sharp:
        H_sqrt = H
        EigTuple = namedtuple("EigTuple", ["eigenvalues", "eigenvectors"])
        eigH = EigTuple(torch.ones(W.shape[1]).to(device), H)
    else:
        eigH = torch.linalg.eigh(H)

        if eigH.eigenvalues.min() < quant_params.quip_args.sigma_reg:
            H = H + (quant_params.quip_args.sigma_reg - eigH.eigenvalues.min()) * \
                    torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            
        H_sqrt = (eigH.eigenvectors @
                    torch.diag(torch.sqrt(eigH.eigenvalues)) @
                    eigH.eigenvectors.T)
    
    # Initialization and Hadamard transform
    best_decomp = LPLRQInfo(
        Q=torch.zeros_like(W).float(),
        L=torch.zeros(W.shape[0], quant_params.rank).to(device),
        R=torch.zeros(quant_params.rank, W.shape[1]).to(device))
    
    if quant_params.quip_sharp_initialization:
            quant_params.hadamard_transform = True
            W, H, best_decomp = \
                quip_sharp_initialization(W, H, quant_params)

            eigH = torch.linalg.eigh(H)
            H_sqrt = (eigH.eigenvectors @
                      torch.diag(torch.sqrt(eigH.eigenvalues)) @
                      eigH.eigenvectors.T)

    elif quant_params.hadamard_transform:
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

    to_iter = range(quant_params.iters)
    if use_tqdm:
        to_iter = tqdm(to_iter)
    for _ in to_iter:
        for mtx in quant_params.update_order:
            if mtx == "LR":
                maybe_update_LR(curr_decomp, quant_params, W, H_sqrt, eigH, device)
            elif mtx == "Q":
                maybe_update_Q(curr_decomp, quant_params, W, H, device)
            
            errors[mtx].append(
                activation_aware_error(W, H, curr_decomp, device)
            )
            if errors[mtx][-1] < min_error:
                min_error = errors[mtx][-1]
                best_decomp = deepcopy(curr_decomp)
    best_decomp.errors = errors

    # Update scales
    best_decomp.global_scale = global_scale
    return best_decomp


def activation_aware_error(
    W: torch.Tensor,
    H: torch.Tensor,
    lplr_q_info: LPLRQInfo, 
    device: str
):
    """
    Computes the activation-aware loss for a sublayer as
    tr((W - W_hat) H (W - W_hat).T) / tr(W H^1/2),
    where H^1/2 is the symmetric square root.
    """

    W = W.to(device).float()
    W_hat = lplr_q_info.Q + lplr_q_info.L @ lplr_q_info.R
    W_hat *= lplr_q_info.global_scale

    error = (torch.trace((W_hat - W) @ H @ (W_hat - W).T) / 
                torch.trace(W @ H @ W.T)).item()
    return error

def quip_sharp_initialization(
    W, H, quant_params, device
):
    codebook_str = BITS_TO_CODEBOOK[quant_params.Q_bits]
    cb = codebook.get_codebook(codebook_str).to(W.device)

    old_lora_rank = quant_params.quip_args.lora_rank
    quant_params.quip_args.lora_rank = quant_params.rank
    Q, attr = quip.quantize(
        H_orig=H,
        W_orig=W,
        rank=quant_params.rank,
        codebook_orig=cb,
        args=quant_params.quip_args,
        device=device
    )
    Q_idxs = attr['Qidxs'].to(device)
    scaleWH = attr['scaleWH']
    SU = attr['SU'].to(device)
    SV = attr['SV'].to(device)

    Q = quip.RHT_W(
        Q, SU, 1 / SV).float()
    if scaleWH is not None:
        Q *= scaleWH[None, :]

    if quant_params.quip_args.lora_rank != 0:
        L = attr['A'].to(device).float()
        R = attr['B'].to(device).float()
        Q -= L @ R

    quant_params.quip_args.lora_rank = old_lora_rank

    # Re-do the incoherence pre-processing
    if scaleWH is not None:
        W *= scaleWH[None, :]
        H /= scaleWH[None, :]
        H /= scaleWH[:, None]

    H = quip.RHT_H(H, SU)
    W = quip.RHT_W(W, SU, 1 / SV)

    QuIPSharpInit = namedtuple("QuIPSharpInit", ["W", "H", "lplr_info"])
    return QuIPSharpInit(W, H, LPLRQInfo(
        W=W, Q=Q, L=L, R=R, Q_idxs=Q_idxs, SU=SU, SV=SV, scaleWH=scaleWH))


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
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
    W: torch.Tensor,
    H: torch.Tensor,
    device: str
):

    if quant_params.compute_quantized_component:
        residual = W - lplr_q_info.L @ lplr_q_info.R
        if not quant_params.compute_low_rank_factors:
            residual = W
        if quant_params.activation_aware_Q:
            update_Q_data_aware(lplr_q_info, quant_params, H, residual, device)
        else:
            update_Q_non_data_aware(lplr_q_info, quant_params, residual, device)


def update_Q_non_data_aware(
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
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
    lplr_q_info.Q = quant_return.A_hat
    lplr_q_info.Q_idxs = quant_return.A_idxs
    lplr_q_info.Q_scale = quant_return.scale


def update_Q_data_aware(
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
    H: torch.Tensor,
    residual: torch.Tensor,
    device: str
):
    """
    Perform a QuIP# update on (W - LR)
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
                lplr_q_info.L @ lplr_q_info.R @ M,
                quant_params.rank * 3, niter=10)
            V = V[:, :quant_params.rank]
        else:
            _, _, Vh = torch.linalg.svd(
                lplr_q_info.L @ lplr_q_info.R @ M, full_matrices=False)
            V = Vh.T[:, :quant_params.rank]

        H = H - (M @ V @ V.T @ M.T).to(H.dtype)
        min_eigval = torch.linalg.eigh(H).eigenvalues.min()
        H = H + min_eigval.abs() * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        alpha = torch.diag(H).mean().abs() * quant_params.quip_args.sigma_reg2
        H = H + alpha * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)

    if quant_params.full_quip_sharp:
        assert not quant_params.hadamard_transform, \
            ("Full QuIP# incompatible with performing Hadamard transform "
                "on our end")
        assert quant_params.rank == 0 or \
            not quant_params.compute_low_rank_factors, \
            ("Full QuIP# incompatible with separately computing low-rank "
            "factors.")

        lplr_q_info.Q, attr = quip.quantize(
            H_orig=H,
            W_orig=residual,
            rank=0,
            codebook_orig=cb,
            args=quant_params.quip_args,
            device=device
        )
        lplr_q_info.Q_idxs = attr['Qidxs'].to(device)

        lplr_q_info.scaleWH = attr['scaleWH']
        lplr_q_info.SU = attr['SU']
        lplr_q_info.SV = attr['SV']
        if quant_params.quip_args.lora_rank != 0:
            lplr_q_info.L = attr['A'].to(device) / lplr_q_info.SV[0].abs().sqrt()
            lplr_q_info.R = attr['B'].to(device) / lplr_q_info.SV[0].abs().sqrt()
            lplr_q_info.L_scale = scale
            lplr_q_info.R_scale = scale
            lplr_q_info.Q -= lplr_q_info.L @ lplr_q_info.R

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
        lplr_q_info.Q_idxs = Qidxs
        lplr_q_info.Q = Q
        lplr_q_info.Q_idxs = cb.maybe_pack_idxs(lplr_q_info.Q_idxs)

    lplr_q_info.Q_scale = scale
    lplr_q_info.Q *= scale


def LR_init(
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
    H_sqrt: torch.Tensor,
    eigH: torch.Tensor,
    residual: torch.Tensor
):
    if quant_params.activation_aware_LR:
        Y = residual @ H_sqrt @ eigH.eigenvectors
        if quant_params.rand_svd:
            q = min(quant_params.rank*2, min(*lplr_q_info.W.shape))
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
                    min(*lplr_q_info.W.shape))
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
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
    W: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device
):
    if quant_params.compute_low_rank_factors:
        residual = W - lplr_q_info.Q
        update_LR(lplr_q_info, quant_params, residual, H_sqrt, eigH, device)


def update_LR(
    lplr_q_info: LPLRQInfo,
    quant_params: ActivationAwareQuantParams,
    residual: torch.Tensor,
    H_sqrt: torch.Tensor,
    eigH,
    device
):
    """
    Run (potentially activation-aware) LPLR on (W - Q)
    """
    data_aware = quant_params.activation_aware_LR

    # Initialization of L, R
    L, R = LR_init(lplr_q_info, quant_params, H_sqrt, eigH, residual)

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

        debug_metadata = {"quant_error_L": [],
                            "relative_error_L": [],
                            "quant_error_R": [],
                            "relative_error_R": []}
        
        best_L, best_R = L, R
        best_L_quant_out, best_R_quant_out = None, None
        best_error = float('inf')

        for it in range(quant_params.lplr_iters):
            # L
            if data_aware:
                L = torch.linalg.lstsq((R @ H_sqrt).T, (residual @ H_sqrt).T)[0].T
                if torch.isnan(L).any():
                    L = (residual @ H_sqrt) @ torch.linalg.pinv(R @ H_sqrt)
            else:
                L = torch.linalg.lstsq(R.T, residual.T)[0].T
                if torch.isnan(R).any():
                    L = residual @ torch.linalg.pinv(R)

            if quant_params.hadamard_transform_L and it == 0:
                rand = torch.normal(torch.zeros(L.shape[1])).to(device)
                rademachers = torch.diag((rand > 0).float() - (rand <= 0).float())

                L_had = matmul_hadU(L @ rademachers)
                L = L_had
                quant_out_L = quantize_matrix(L.T, quant_params, quant_info_L)
                L_hat = quant_out_L.A_hat.T
            elif quant_params.Haar_transform_L and it == 0:
                k = L.shape[1]
                S = torch.normal(
                    mean=torch.zeros(k, k)).to(device)
                US, _, VSh = torch.linalg.svd(S)
                S = US @ VSh
                L = L @ S
                quant_out_L = quantize_matrix(L.T, quant_params, quant_info_L)
                L_hat = quant_out_L.A_hat.T
            else:
                quant_out_L = quantize_matrix(L.T, quant_params, quant_info_L)
                L_hat = quant_out_L.A_hat.T
            
            debug_metadata["quant_error_L"].append(
                torch.linalg.matrix_norm(L - L_hat).item())
            debug_metadata["relative_error_L"].append(
                torch.linalg.matrix_norm(L - L_hat).item() /
                torch.linalg.matrix_norm(L).item())
            L = L_hat

            # R
            R = torch.linalg.lstsq(L, residual)[0]
            if torch.isnan(R).any():
                R = torch.linalg.pinv(L) @ residual

            if quant_params.hadamard_transform_R and it == 0:
                rand = torch.normal(torch.zeros(R.shape[1])).to(device)
                rademachers = torch.diag((rand > 0).float() - (rand <= 0).float())
                R_had = matmul_hadUt(R @ rademachers)
                quant_out_R = quantize_matrix(R_had, quant_params,quant_info_R)
                R_hat = quant_out_R.A_hat
                R_hat = matmul_hadU(R_hat) @ rademachers
            else:
                quant_out_R = quantize_matrix(R, quant_params, quant_info_R)
                R_hat = quant_out_R.A_hat

                
            debug_metadata["quant_error_R"].append(
                torch.linalg.matrix_norm(R - R_hat).item())
            debug_metadata["relative_error_R"].append(
                torch.linalg.matrix_norm(R - R_hat).item() /
                torch.linalg.matrix_norm(R).item())
            R = R_hat

            error = torch.linalg.matrix_norm((residual - L @ R) @ H_sqrt)
            if error < best_error:
                best_L, best_R = L, R
                best_L_quant_out = quant_out_L
                best_R_quant_out = quant_out_R
                best_error = error

        lplr_q_info.L_idxs = best_L_quant_out.A_idxs
        lplr_q_info.R_idxs = best_R_quant_out.A_idxs
        lplr_q_info.L_scale = best_L_quant_out.scale
        lplr_q_info.R_scale = best_R_quant_out.scale

        L, R = best_L, best_R

        if quant_params.verbose:
            for key, arr in debug_metadata.items():
                print(f"{key}: {[round(val, 4) for val in arr]}")
            print("-"*60)

    lplr_q_info.L = L
    lplr_q_info.R = R