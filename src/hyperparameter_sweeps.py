import torch
import numpy as np
from loguru import logger
import sys
from quantization import *
from weight_compressors import *

def set_beta_lplr(
    alpha: float = 0,
    budget: int = 0,
    n: int = 0,
    d: int = 0,
    kwarg_dict:dict = {
        "B1": 4, "B2": 4
    }
):
    """
    Given an overall bit-budget and choice of alpha (the proportion of columns
    of the low-rank factors that are kept in full precision), determine beta
    (the proportion of singular values to keep). 

    Corresponds to alternating_mixed_lplr.

    - alpha (float): proportion of columns of the low-rank factors that are kept in
        full precision (16B).
    - budget (int): total number of bits allowed for L and R combined.
    - n, d (ints): number of rows, columns in the matrix being approximated.
    - kwarg_dict (dict): dictionary of keyword arguments for LPLR (see
        alternating_mixed_lplr). Must include keys "B1" and "B2".

    Output:
    - float: maximum beta allowed to remain within the budget.
    """
    B1 = kwarg_dict["B1"]
    B2 = kwarg_dict["B2"]
    return budget / (16*alpha*d*(n + d) + (1-alpha)*d*(B1*n + B2*d))

def set_beta_lplr_plus_q(
    alpha: float = 0,
    budget: int = 0,
    n: int = 0,
    d: int = 0,
    kwarg_dict: dict = {
        "B1": 4, "B2": 4, "BQ": 2
    }
):
    """
    Given an overall bit-budget and choice of alpha (the proportion of columns
    of the low-rank factors that are kept in full precision), determine beta
    (the proportion of singular values to keep).

    Corresponds to alternating_mixed_lplr_plus_q.

    - alpha (float): proportion of columns of the low-rank factors that are kept in
        full precision (16B).
    - budget (int): total number of bits allowed for L and R combined.
    - n, d (ints): number of rows, columns in the matrix being approximated.
    - kwarg_dict (dict): dictionary of keyword arguments for LPLR+LoftQ (see
        alternating_mixed_lplr_plus_q). Must include keys "B1", "B2", "BQ".

    Output:
    - float: maximum beta allowed to remain within the budget.
    """

    B1 = kwarg_dict["B1"]
    B2 = kwarg_dict["B2"]
    BQ = kwarg_dict["BQ"]
    budget -= BQ*n*d
    return budget / (16*alpha*d*(n + d) + (1-alpha)*d*(B1*n + B2*d))

def lplr_sweep_alpha(
    X: torch.Tensor = None,
    budget: int = 0,
    kwarg_dict: dict = {},
    alpha_start: float = 0,
    alpha_stop: float = 0.5,
    alpha_step: float = 0.1,
    lplr_type: int = LplrType.ALTERNATING_MIXED,
    prune: bool = False,
    debug: bool = False
):
    """
    Perform a hyperparameter sweep on the LPLR parameter alpha, which is the
    ratio of the columns of L, R to keep in full precision. Beta, the fraction
    of singular values to keep, is set such that we meet the provided budget,
    in bits, of the mixed LPLR representation.

    - X (torch.Tensor): matrix to approximate.
    - budget (int): total number of bits allowed for L and R combined.
    - kwarg_dict (dict): dictionary of keyword arguments for the specified LPLR
        vairant. Must include all bit-precision parameters.
    - alpha_start (float): first value of alpha to try.
    - alpha_stop (float): last value of alpha 
    """

    lplr_fn = alternating_mixed_lplr
    set_beta_fn = set_beta_lplr
    if lplr_type == LplrType.DIRECT_SVD:
        lplr_fn = direct_svd_mixed_lplr
    elif lplr_type == LplrType.WITH_Q:
        lplr_fn = alternating_mixed_lplr_plus_q
        set_beta_fn = set_beta_lplr_plus_q

    n, d = X.size()

    best_fro_err = float('inf')
    best_alpha = None
    best_beta = None
    best_mtxs = None

    kwargs = kwarg_dict.copy()
    kwargs["X"] = X
    for alpha in np.arange(alpha_start, alpha_stop+alpha_step, alpha_step):
        beta = set_beta_fn(alpha, budget, n, d, kwarg_dict)
        beta = min(beta, 1)
        k = int(d * beta)
        r = int(k * alpha)

        kwargs["k"] = k
        kwargs["r1"] = r
        kwargs["r2"] = r

        if debug:
            print("-"*50)
            sys.stdout.flush()
            logger.info(f"B1={kwargs['B1']}, B2={kwargs['B2']}, alpha={np.round(alpha, 8)}, beta={np.round(beta, 8)}")
        if k == 0:
            logger.warning(f"The bit budget of {budget} cannot be met for alpha={np.round(alpha, 8)}. Stopping early")
            break

        mtxs, X_hat = lplr_fn(**kwargs)

        fro_err = torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item()
        if debug:
            logger.info(f"Frobenius norm error: {fro_err}")
        if fro_err <= best_fro_err:
            best_fro_err = fro_err
            best_alpha = alpha
            best_beta = beta
            best_mtxs = mtxs
        elif prune:
            logger.warning("Error increased after increasing alpha and prune is set to True. Giving up.")
            break

    if debug:
        print("-"*50)
        sys.stdout.flush()
        logger.info(f"[B1={kwargs['B1']}, B2={kwargs['B2']}]The best frobenius norm error was for alpha={np.round(best_alpha, 8)}: {best_fro_err}")
    return best_mtxs, best_alpha, best_beta, best_fro_err

def lplr_sweep_alpha_and_B(
    X:torch.Tensor = None,
    budget: int = 0,
    kwarg_dict = {},
    alpha_start:float = 0,
    alpha_stop:float = 0.5,
    alpha_step:float = 0.1,
    B_options=[4, 8],
    lplr_type:int = LplrType.ALTERNATING_MIXED,
    prune=False,
    debug=False
):
    
    best_fro_err = float('inf')
    best_mtxs = None
    best_alpha = None
    best_beta = None
    best_B = None
    for B in B_options:
        kwargs = kwarg_dict.copy()
        kwargs["B1"] = B
        kwargs["B2"] = B
        mtxs, alpha, beta, fro_err = lplr_sweep_alpha(
            X=X, budget=budget, kwarg_dict=kwargs,
            alpha_start=alpha_start, alpha_stop=alpha_stop, alpha_step=alpha_step,
            lplr_type=lplr_type,
            prune=prune, debug=debug
        )

        if fro_err < best_fro_err:
            best_fro_err = fro_err
            best_mtxs = mtxs
            best_alpha = alpha
            best_beta = beta
            best_B = B

    if debug:
        print("-"*50)
        sys.stdout.flush()
        logger.info(f"The best frobenius norm error was for B={best_B}, alpha={np.round(best_alpha, 8)}: {best_fro_err}")
    return best_mtxs, best_alpha, best_beta, best_B, best_fro_err
