import torch
import numpy as np
from scipy.sparse.linalg import cg
from loguru import logger
from tqdm import tqdm
import math
from quantization import *
from lplr_utils import normalize_and_shift_wrt_inner_prod


def alternating_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quantization_fn=quantize,
    normalize_and_shift=False,
    iters=10,
    log_errors=False
):
    """Obtain a low-precision low-rank approximation of X using alternating optimization of
    left and right low rank factors

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors will be dropped
    r1 (int, optional): No. of singular vectors to be kept in full precision for the first factor; Trailing (k - r) singular vectors will be quantized
    r2 (int, optional): No. of singular vectors to be kept in full precision for the second factor
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional):Bit-budget for second low-rank factor
    quantization_fn (function, optional): either `quantize` or `quantize_nf`; specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for better approximation
    iters (int, optional): Number of iterations of alternating optimization
    sketch (Sketch, optional): Sketch type
    log_errors (bool, optional): Return fro-norm errors for each iteration

    torch.Tensor: Low rank quantized factors
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

    U = U[:, 0:k]
    S = S[0:k]
    print(S)
    VT = VT[0:k, :]
    S_sqrt = torch.diag(torch.tensor([math.sqrt(x) for x in S])).to(X.device)

    # Get the initial left low-rank factor
    # logger.info(f"Getting initial condition for Alternating Mixed LPLR")

    L = quantize_small_sv_components(U @ S_sqrt, B1, r1, quantization_fn=quantization_fn)
    if torch.isnan(L).any().item():
        logger.error(f"NaNs encountered in quantizing first factor")

    # Get the initial right low-rank factor
    L_pinv = torch.linalg.pinv(L.float()).type(X.dtype)
    if torch.isnan(L_pinv).any().item():
        logger.error(f"NaNs encountered in pinv")

    W = L_pinv @ X
    if torch.isnan(W).any().item():
        logger.error(f"NaNs encountered in L_pinv @ X")

    R = quantize_small_sv_components(W, B2, r2, quantization_fn=quantization_fn)
    if torch.isnan(R).any().item():
        logger.error(f"NaNs encountered in quantizing second factor")

    errors = [torch.norm(X - L @ R, p="fro").item()]

    best_error = errors[0]
    best_mtxs = (L, R)

    for _ in tqdm(range(1, iters)):
        # Get the left low-rank factor
        Y = torch.linalg.lstsq(R.float().T, X.float().T)[0].T

        if torch.isnan(Y).any().item():
            logger.error(f"NaNs encountered in finding unquantized L. Giving up.")
            break

        L = quantize_small_sv_components(Y, B1, r1, quantization_fn=quantization_fn)
        if torch.isnan(L).any().item():
            logger.error(f"NaNs encountered in Q(L). Giving up.")
            break

        # Get the right low-rank factor
        W = torch.linalg.lstsq(L.float(), X.float())[0]

        if torch.isnan(W).any().item():
            logger.error(f"NaNs encountered in finding unquantized R. Giving up.")
            break

        R = quantize_small_sv_components(W, B2, r2, quantization_fn=quantization_fn)
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

def alternating_mixed_lplr_plus_q(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    BQ: int = 4,
    quantization_fn=quantize,
    normalize_and_shift=False,
    outer_iters=10,
    inner_iters=10,
    log_errors=False
):
    """Obtain a low-precision low-rank approximation of X using alternating optimization of
    left and right low rank factors

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors will be dropped
    r1 (int, optional): No. of singular vectors to be kept in full precision for the first factor; Trailing (k - r) singular vectors will be quantized
    r2 (int, optional): No. of singular vectors to be kept in full precision for the second factor
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional):Bit-budget for second low-rank factor
    quantization_fn (function, optional): either `quantize` or `quantize_nf`; specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for better approximation
    iters (int, optional): Number of iterations of alternating optimization
    sketch (Sketch, optional): Sketch type
    log_errors (bool, optional): Return fro-norm errors for each iteration

    torch.Tensor: Low rank quantized factors
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    L = torch.zeros(X.shape[0], k).to(X.device)
    R = torch.zeros(k, X.shape[1]).to(X.device)

    best_error = float('inf')
    best_mtxs = None
    errors = []

    for _ in range(1, outer_iters):
        Q = quantization_fn(X - L @ R, BQ)

        error = torch.norm(X - L @ R - Q, p="fro").item()
        mtxs, X_minus_Q_hat = alternating_mixed_lplr(
            X=X-Q,
            k=k, r1=r1, r2=r2,
            B1=B1, B2=B2,
            quantization_fn=quantization_fn,
            normalize_and_shift=normalize_and_shift,
            iters=inner_iters
        )
        error_after_setting_LR = torch.norm(X - X_minus_Q_hat - Q, p="fro").item()
        if error > error_after_setting_LR:
            error = error_after_setting_LR
            L, R = mtxs

        errors.append(error)
        if error < best_error:
            best_error = error
            best_mtxs = (L, R, Q)

    L, R, Q = best_mtxs
    out = L @ R + Q
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R + Q)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    if log_errors:
        return (L, R, Q), out, errors

    return (L, R, Q), out

def direct_svd_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quantization_fn=quantize,
    normalize_and_shift=False,
):
    """Obtain a low-precision low-rank approximation of X using alternating optimization of
    left and right low rank factors

    X (torch.Tensor, optional): Input matrix (Tall or square)
    k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors will be dropped
    r1 (int, optional): No. of singular vectors to be kept in full precision for the first factor; Trailing (k - r) singular vectors will be quantized
    r2 (int, optional): No. of singular vectors to be kept in full precision for the second factor
    B1 (int, optional): Bit-budget for first low-rank factor
    B2 (int, optional):Bit-budget for second low-rank factor
    quantization_fn (function, optional): either `quantize` or `quantize_nf`; specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for better approximation
    iters (int, optional): Number of iterations of alternating optimization
    sketch (Sketch, optional): Sketch type

    torch.Tensor: Low rank quantized factors
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
    assert (
        k >= r1 and k >= r2
    ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

    # Compute full SVD
    U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

    U = U[:, 0:k]
    S = S[0:k]
    VT = VT[0:k, :]
    S_sqrt = torch.diag(torch.tensor([math.sqrt(x) for x in S])).to(X.device)

    # Get the initial left low-rank factor
    # logger.info(f"Getting initial condition for Alternating Mixed LPLR")

    L = quantize_small_sv_components(U @ S_sqrt, B1, r1, quantization_fn=quantization_fn)
    if torch.isnan(L).any().item():
        logger.error(f"NaNs encountered in quantizing first factor")

    R = quantize_small_sv_components(S_sqrt @ VT, B2, r2, quantization_fn=quantization_fn)
    if torch.isnan(R).any().item():
        logger.error(f"NaNs encountered in quantizing second factor")

    out = L @ R
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in LPLRed matrix")

    return (L, R), out

def loftq(
    X: torch.Tensor = None,
    r: int = None,
    B: int = 8,
    quantization_fn=quantize_nf,
    normalize_and_shift=False,
    iters=10,
    log_errors=False
):
    """Obtain a low-precision low-rank approximation of X using alternating optimization of
    left and right low rank factors

    X (torch.Tensor, optional): Input matrix (Tall or square)
    r (int, optional): Dimension of the low-rank factor
    B (int, optional): Bit-budget for the quantized part
    quantization_fn (function, optional): either `quantize` or `quantize_nf`; specifies the function used for quantization.
    normalize_and_shift (bool, optional): Maintain additional scalars for better approximation
    iters (int, optional): Number of iterations of alternating optimization
    log_errors (bool, optional): Return fro-norm errors for each iteration

    torch.Tensor: Low rank quantized factors
    """

    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"

    n, d = X.shape
    L = torch.zeros(n, r).to(X.device)
    R = torch.zeros(r, d).to(X.device)

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in tqdm(range(iters)):
        Q = quantization_fn(X - L @ R, B)
        
        # Compute full SVD
        U, S, VT = torch.linalg.svd((X - Q).float(), full_matrices=False)

        U = U[:, 0:r]
        S = S[0:r]
        VT = VT[0:r, :]
        S_sqrt = torch.diag(torch.tensor([math.sqrt(x) for x in S])).to(X.device)
        
        L = U @ S_sqrt
        R = S_sqrt @ VT

        errors.append(torch.norm(X - L @ R - Q, p="fro").item())
        if errors[-1] < best_error:
            best_error = errors[-1]
            best_mtxs = (L, R, Q)        
    L, R, Q = best_mtxs
    out = L @ R + Q
    if normalize_and_shift:
        out = normalize_and_shift_wrt_inner_prod(X, L @ R + Q)

    if torch.isnan(out).any().item():
        logger.error(f"NaNs encountered in loftq-ed matrix")

    if log_errors:
        return (L, R, Q), out, errors

    return (L, R, Q), out