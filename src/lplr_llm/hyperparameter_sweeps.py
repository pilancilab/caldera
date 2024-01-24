import torch
import numpy as np
from loguru import logger
import sys
from lplr_llm.quantization import *
from lplr_llm.weight_compressors import *
from peft.utils.loftq_utils import loftq_init
from peft.utils.loftq_lplr_utils import loftq_lplr_init

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

def set_beta_loftq_lplr(
    alpha: float = 0,
    budget: int = 0,
    n: int = 0,
    d: int = 0,
    kwarg_dict: dict = {
        "num_bits_factors": 8, "num_bits": 4
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
    - kwarg_dict (dict): dictionary of keyword arguments for LoftQ-LPLR (see
        loftq_lplr_init in peft/src/peft/utils/loftq_lplr_utils). Must include
        keys "num_bits_factors" and "num_bits"

    Output:
    - float: maximum beta allowed to remain within the budget.
    """

    B = kwarg_dict["num_bits_factors"]
    BQ = kwarg_dict["num_bits"]
    budget -= BQ*n*d
    return budget / (16*alpha*d*(n + d) + (1-alpha)*d*B*(n+d))

def lplr_sweep_alpha(
    X: torch.Tensor = None,
    budget: int = 0,
    weight_comp_config: WeightCompressionConfig = WeightCompressionConfig(),
    alpha_start: float = 0,
    alpha_stop: float = 0.5,
    alpha_step: float = 0.1,
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
        vairant. Must include all bit-precision-related parameters.
    - alpha_start (float): first value of alpha to try.
    - alpha_stop (float): last value of alpha to try (inclusive).
    - alpha_step (float): interval between consecutive alpha values.
    - lplr_type (from Enum AlgorithmType): specifies the LPLR variant to use.
    - prune (bool): if set True (default is False), then we stop after the
        Frobenius norm error increases from one iteration to the next, even if
        we still have more values of alpha to try.
    - debug (bool): if set True (default is False), debug logs are printed.
    """
    n, d = X.size()

    kwarg_dict = weight_comp_config.algorithm_kwargs
    lplr_type = weight_comp_config.algorithm_type

    best_fro_err = float('inf')
    best_alpha = None
    best_beta = None
    best_mtxs = None

    kwargs = kwarg_dict.copy()
    for alpha in np.arange(alpha_start, alpha_stop+alpha_step, alpha_step):
        if lplr_type == AlgorithmType.ALTERNATING_MIXED_LPLR or \
                lplr_type == AlgorithmType.DIRECT_SVD_LPLR:
            beta = set_beta_lplr(alpha, budget, n, d, kwarg_dict)
        else:
            beta = set_beta_loftq_lplr(alpha, budget, n, d, kwarg_dict)
        
        beta = min(beta, 1)
        k = int(d * beta)
        r = int(k * alpha)

        if lplr_type == AlgorithmType.ALTERNATING_MIXED_LPLR or \
                lplr_type == AlgorithmType.DIRECT_SVD_LPLR:
            kwargs["k"] = k
            kwargs["r1"] = r
            kwargs["r2"] = r
            kwargs["X"] = X
        else: # LOFTQ-LPLR
            kwargs["num_full_precision_factors"] = r
            kwargs["reduced_rank"] = k
            kwargs["weight"] = X

        if lplr_type == AlgorithmType.LOFTQ_LPLR:
            B = kwargs["num_bits_factors"]
        else:
            B = kwargs['B1']

        if debug:
            print("-"*50)
            sys.stdout.flush()
            logger.info(f"B={B}, alpha={np.round(alpha, 8)}, beta={np.round(beta, 8)}")
        if k == 0:
            logger.warning(f"The bit budget of {budget} cannot be met for alpha={np.round(alpha, 8)}. Stopping early")
            break

        if lplr_type == AlgorithmType.ALTERNATING_MIXED_LPLR:
            mtxs, X_hat = alternating_mixed_lplr(**kwargs)
        elif lplr_type == AlgorithmType.DIRECT_SVD_LPLR:
            mtxs, X_hat = direct_svd_mixed_lplr(**kwargs)
        else: ## LOFTQ-LPLR
            Q, R, L = loftq_lplr_init(**kwargs)
            mtxs = (Q, R, L)
            X_hat = Q + L @ R

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
        logger.info(f"[B={B}]The best frobenius norm error was for alpha={np.round(best_alpha, 8)}: {best_fro_err}")
    return best_mtxs, best_alpha, best_beta, best_fro_err

def lplr_sweep_alpha_and_B(
    X:torch.Tensor = None,
    budget: int = 0,
    weight_comp_config: WeightCompressionConfig = WeightCompressionConfig(),
    alpha_start:float = 0,
    alpha_stop:float = 0.5,
    alpha_step:float = 0.1,
    B_options:list[int] = [4, 8],
    prune=False,
    debug=False
):
    """
    Perform a hyperparameter sweep on the LPLR parameters alpha, which is the
    ratio of the columns of L, R to keep in full precision, and B, which is the
    bit precision of the quantized factors. Beta, the fraction of singular values
    to keep, is set such that we meet the provided budget, in bits, of the mixed
    LPLR representation.

    - X (torch.Tensor): matrix to approximate.
    - budget (int): total number of bits allowed for L and R combined.
    - kwarg_dict (dict): dictionary of keyword arguments for the specified LPLR
        vairant. Must include all bit-precision-related parameters.
    - alpha_start (float): first value of alpha to try.
    - alpha_stop (float): last value of alpha to try (inclusive).
    - alpha_step (float): interval between consecutive alpha values.
    - B_options (list[int]): values of B to try. Must be a subset of [2, 4, 8].
    - lplr_type (from Enum AlgorithmType): specifies the LPLR variant to use.
    - prune (bool): if set True (default is False), then we stop after the
        Frobenius norm error increases from one iteration to the next, even if
        we still have more values of alpha to try.
    - debug (bool): if set True (default is False), debug logs are printed.
    """
    
    best_fro_err = float('inf')
    best_mtxs = None
    best_alpha = None
    best_beta = None
    best_B = None

    weight_comp_config = WeightCompressionConfig(
        algorithm_type=weight_comp_config.algorithm_type,
        algorithm_kwargs=weight_comp_config.algorithm_kwargs.copy()
    )
    for B in B_options:
        if weight_comp_config.algorithm_type != AlgorithmType.LOFTQ_LPLR:
        
            weight_comp_config.algorithm_kwargs["B1"] = B
            weight_comp_config.algorithm_kwargs["B2"] = B
        else:
            weight_comp_config.algorithm_kwargs["num_bits_factors"] = B

        mtxs, alpha, beta, fro_err = lplr_sweep_alpha(
            X=X, budget=budget, weight_comp_config=weight_comp_config,
            alpha_start=alpha_start, alpha_stop=alpha_stop,
            alpha_step=alpha_step,
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
