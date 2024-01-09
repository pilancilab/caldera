import torch
import numpy as np
from loguru import logger
import sys
from quantization import *
from weight_compressors import *

def lplr_sweep_alpha(
    X:torch.Tensor = None,
    budget: int = 0,
    alpha_start:float = 0,
    alpha_stop:float = 0.5,
    alpha_step:float = 0.1,
    B1:int = 8,
    B2:int = 8,
    quantization_fn = quantize,
    iters=50,
    run_alternating_optimization=True,
    debug=False
):
    """
    Perform a hyperparameter sweep on the LPLR parameter alpha, which is the
    ratio of the columns of L, R to keep in full precision. Beta, the fraction
    of singular values to keep, is set such that we meet the provided budget,
    in bits, of the mixed LPLR representation.
    """

    n, d = X.size()

    best_fro_err = float('inf')
    best_alpha = None
    best_beta = None
    best_L_R = None
    for alpha in np.arange(alpha_start, alpha_stop+alpha_step, alpha_step):
        beta = budget / (16*alpha*d*(n + d) + (1-alpha)*d*(B1*n + B2*d))
        k = int(d * beta)
        r = int(k * alpha)

        if debug:
            print("-"*50)
            sys.stdout.flush()
            logger.info(f"B1={B1}, B2={B2}, alpha={np.round(alpha, 8)}, beta={np.round(beta, 8)}")
        if k == 0:
            logger.warning(f"The bit budget of {budget} cannot be met for alpha={np.round(alpha, 8)}. Stopping early")
            break

        if run_alternating_optimization:
            L, R, X_hat = alternating_mixed_lplr(
                X=X, k=k, r1=r, r2=r, B1=B1, B2=B2,
                quantization_fn=quantization_fn,
                normalize_and_shift=True,
                log_errors=False,
                iters=iters
            )
        else:
            L, R, X_hat = direct_svd_mixed_lplr(
                X=X, k=k, r1=r, r2=r, B1=B1, B2=B2,
                quantization_fn=quantization_fn,
                normalize_and_shift=True,
                log_errors=False
            )

        fro_err = torch.norm(X - X_hat, p="fro") / torch.norm(X, p="fro").item()
        if debug:
            logger.info(f"Frobenius norm error: {fro_err}")
        if fro_err <= best_fro_err:
            best_fro_err = fro_err
            best_alpha = alpha
            best_beta = beta
            best_L_R = (L, R)

    if debug:
        print("-"*50)
        sys.stdout.flush()
        logger.info(f"[B1={B1}, B2={B2}] The best frobenius norm error was for alpha={np.round(best_alpha, 8)}: {best_fro_err}")
    return best_L_R, best_alpha, best_beta, best_fro_err

def lplr_sweep_alpha_and_B(
    X:torch.Tensor = None,
    budget: int = 0,
    alpha_start:float = 0,
    alpha_stop:float = 0.5,
    alpha_step:float = 0.1,
    B_options=[2, 4, 8],
    quantization_fn = quantize,
    iters=50,
    debug=False
):
    
    best_fro_err = float('inf')
    best_L_R = None
    best_alpha = None
    best_beta = None
    best_B = None
    for B in B_options:
        L_R, alpha, beta, fro_err = lplr_sweep_alpha(
            X=X, budget=budget, alpha_start=alpha_start,
            alpha_stop=alpha_stop, alpha_step=alpha_step,
            B1=B, B2=B, quantization_fn=quantization_fn,
            iters=iters, debug=debug
        )

        if fro_err < best_fro_err:
            best_fro_err = fro_err
            best_L_R = L_R
            best_alpha = alpha
            best_beta = beta
            best_B = B

    if debug:
        print("-"*50)
        sys.stdout.flush()
        logger.info(f"The best frobenius norm error was for B={best_B}, alpha={np.round(best_alpha, 8)}: {best_fro_err}")
    return best_L_R, best_alpha, best_beta, best_B, best_fro_err
