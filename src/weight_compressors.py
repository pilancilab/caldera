import torch
from loguru import logger
from tqdm import tqdm
from quantization import *
from lplr_utils import normalize_and_shift_wrt_inner_prod
from enum import Enum

class AlgorithmType(Enum):
    ALTERNATING_MIXED_LPLR = 0
    DIRECT_SVD_LPLR = 1
    LOFTQ = 2
    LOFTQ_LPLR = 3

def alternating_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quant_type: int = QuantType.UNIFORM,
    normalize_and_shift=False,
    iters=10,
    max_cond=5e6,
    use_tqdm=False,
    log_errors=False
):
    """
    Obtain a low-precision low-rank approximation of X using alternating optimization of
    left and right low rank factors

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors
        will be dropped.
    r1 (int, optional): No. of singular vectors to be kept in full precision
        for the first factor; Trailing (k - r) singular vectors will be quantized.
    r2 (int, optional): No. of singular vectors to be kept in full precision
        for the second factor
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional): Bit-budget for second low-rank factor
    quant_type (QuantType, optional): specifies whether to use uniform or 
        NormalFloat quantization. Member of the QuantType Enum.
    normalize_and_shift (bool, optional): Maintain additional scalars for
        better approximation
    iters (int, optional): Number of iterations of alternating optimization
    max_cond (float, optional): if the condition number of the rank-k
        approximation of X is above this number, reduce k such that Xk is
        well-conditioned. 
    [TODO] sketch (Sketch, optional): Sketch type
    log_errors (bool, optional): Return fro-norm errors for each iteration

    Outputs:
    - Tuple[torch.Tensor, torch.Tensor] : Low rank quantized factors (L, R)
    - torch.Tensor: Approximation of X (LR)
    - [only if log_errors] list[float]: fro-norm errors for each iteration.
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    quantizer_B1 = get_quantizer(
        B=B1, quant_type=quant_type, device=X.device
    )
    quantizer_B2 = get_quantizer(
        B=B2, quant_type=quant_type, device=X.device
    )

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

    # If the rank-k approximation of X is ill-conditioned, then the least squares
    # steps might be inaccurate (or, for the CUDA implementation, return NaNs). We
    # can avoid this by resetting k if necessary.
    if S[0] / S[k] >= max_cond:
        k = torch.sum(S > S[0] / max_cond)
        r1 = min(r1, k)
        r2 = min(r2, k)
        logger.warning(f"Could not find k non-negligible singular values of X, so setting k to {k}.")
    
    U = U[:, 0:k]
    S = S[0:k]
    VT = VT[0:k, :]
    S_sqrt = torch.diag(torch.sqrt(S))

    # Get the initial left low-rank factor
    L = quantize_small_sv_components(U @ S_sqrt, r1, quantizer=quantizer_B1)
    if torch.isnan(L).any().item():
        logger.error(f"NaNs encountered in quantizing first factor")

    # Get the initial right low-rank factor
    # Get the right low-rank factor
    W = torch.linalg.lstsq(L.float(), X.float())[0]

    if torch.isnan(W).any().item():
        logger.error(f"NaNs encountered in finding unquantized R.")

    R = quantize_small_sv_components(W.T, r2, quantizer=quantizer_B2).T
    if torch.isnan(R).any().item():
        logger.error(f"NaNs encountered in quantizing second factor")

    errors = [torch.norm(X - L @ R, p="fro").item()]

    best_error = errors[0]
    best_mtxs = (L, R)

    to_iter = range(1, iters)
    if use_tqdm:
        to_iter = tqdm(to_iter)
    for _ in to_iter:
        # Get the left low-rank factor
        Y = torch.linalg.lstsq(R.float().T, X.float().T)[0].T

        if torch.isnan(Y).any().item():
            logger.error(f"NaNs encountered in finding unquantized L. Giving up.")
            break

        L = quantize_small_sv_components(Y, r1, quantizer=quantizer_B1)
        if torch.isnan(L).any().item():
            logger.error(f"NaNs encountered in Q(L). Giving up.")
            break

        # Get the right low-rank factor
        W = torch.linalg.lstsq(L.float(), X.float())[0]

        if torch.isnan(W).any().item():
            logger.error(f"NaNs encountered in finding unquantized R. Giving up.")
            break

        R = quantize_small_sv_components(W.T, r2, quantizer=quantizer_B2).T
        if torch.isnan(R).any().item():
            logger.error(f"NaNs encountered in Q(R). Giving up.")
            break 

        errors.append(torch.norm(X - L @ R, p="fro").item())
        if errors[-1] < best_error:
            best_error = errors[-1]
            best_mtxs = (L, R)

    L, R = best_mtxs
    out = L @ R
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    if log_errors:
        return (L, R), out, errors

    return (L, R), out

def direct_svd_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quant_type: int = QuantType.UNIFORM,
    normalize_and_shift=False
):
    """
    Obtain a low-precision low-rank approximation of X without alternating
    optimization by directly quantizing the factorization produced by taking
    the SVD of X.

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors
        will be dropped.
    r1 (int, optional): No. of singular vectors to be kept in full precision
        for the first factor; Trailing (k - r) singular vectors will be quantized.
    r2 (int, optional): No. of singular vectors to be kept in full precision
        for the second factor.
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional):Bit-budget for second low-rank factor
    quant_type (QuantType, optional): specifies whether to use uniform or 
        NormalFloat quantization. Member of the QuantType Enum.
    normalize_and_shift (bool, optional): Maintain additional scalars for
        better approximation.
    [TODO] sketch (Sketch, optional): Sketch type

    Outputs:
    - Tuple[torch.Tensor, torch.Tensor] : Low rank quantized factors (L, R)
    - torch.Tensor: Approximation of X (LR)
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    quantizer_B1 = get_quantizer(
        B=B1, quant_type=quant_type, device=X.device
    )
    quantizer_B2 = get_quantizer(
        B=B2, quant_type=quant_type, device=X.device
    )

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

    U = U[:, 0:k]
    S = S[0:k]
    VT = VT[0:k, :]
    S_sqrt = torch.diag(torch.sqrt(S))

    # Get the initial left low-rank factor
    L = quantize_small_sv_components(U @ S_sqrt, r1, quantizer=quantizer_B1)
    if torch.isnan(L).any().item():
        logger.error(f"NaNs encountered in quantizing first factor")

    R = quantize_small_sv_components((S_sqrt @ VT).T, r2, quantizer=quantizer_B2).T
    if torch.isnan(R).any().item():
        logger.error(f"NaNs encountered in quantizing second factor")

    out = L @ R
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    return (L, R), out