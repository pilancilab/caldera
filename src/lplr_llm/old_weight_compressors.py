import torch
from loguru import logger
import numpy as np
from scipy.linalg import hadamard
from tqdm import tqdm
from lplr_llm.quantization import *
from lplr_llm.lplr_utils import normalize_and_shift_wrt_inner_prod
from dataclasses import field, dataclass
from lplr_llm.enums import *
from typing import Union
from lplr_llm.error_metrics import *

@dataclass
class WeightCompressionConfig:
    algorithm_type: int = field(default=AlgorithmType.ALTERNATING_MIXED_LPLR)
    algorithm_kwargs: dict = field(default_factory=dict)
    hadamard: dict = field(default=False)

def rademacher_rv(*size):
    rand = torch.rand(*size)
    return 1 * (rand > 0.5) - 1 * (rand <= 0.5)

def undo_hadamard(X, n, d, HL=None, HR=None, col_perm=None, row_perm=None):
    if HL is not None:
        X = HL.T @ X
    if HR is not None:
        X = X @ HR.T
    
    if row_perm is not None:
        row_unperm = torch.argsort(row_perm)
        X = X[row_unperm, :]
    if col_perm is not None:
        col_unperm = torch.argsort(col_perm)
        X = X[:, col_unperm]
    
    return X[:n, :d]

def undo_hadamard_lplr(L, R, X_hat_or_Q, n, d, HL, HR, col_perm, row_perm):
    L = undo_hadamard(L, n=n, d=L.shape[1], HL=HL, row_perm=row_perm)
    R = undo_hadamard(R, n=R.shape[0], d=d, HR=HR, col_perm=col_perm)
    X_hat_or_Q = undo_hadamard(
        X_hat_or_Q, n, d, HL=HL, HR=HR, col_perm=col_perm, row_perm=row_perm
    )
    return L, R, X_hat_or_Q

def hadamard_weight_compression(
    X: torch.Tensor = None,
    config: WeightCompressionConfig = WeightCompressionConfig()
):
    n, d = X.shape
    H1_dim = int(2**np.ceil(np.log2(n)))
    H2_dim = int(2**np.ceil(np.log2(d)))

    H1 = torch.from_numpy(hadamard(H1_dim)).to(X.device).float()
    H1 *= rademacher_rv(H1_dim).to(X.device) # multiplication by diagonal matrix

    X2 = X
    row_perm = torch.arange(n)
    if n < H1_dim:
        # zero-pad X
        X2 = torch.cat((X2, torch.zeros(H1_dim-n, d).to(X.device)), dim=0)
        row_perm = torch.argsort(torch.rand(H1_dim))
        X2 = X2[row_perm, :]

    H1 *= 1/np.sqrt(H1_dim)

    H2 = torch.from_numpy(hadamard(H2_dim)).to(X.device).float()
    H2 *= rademacher_rv(H2_dim).to(X.device) # multiplication by diagonal matrix

    col_perm = torch.arange(d)
    if d < H2_dim:
        X2 = torch.cat((X2, torch.zeros(n, H2_dim-d).to(X.device)), dim=1)
        col_perm = torch.argsort(torch.rand(H2_dim))
        X2 = X2[:, col_perm]

    H2 *= 1/np.sqrt(H2_dim)

    X2 = H1 @ X2 @ H2

    if config.algorithm_type == AlgorithmType.ALTERNATING_MIXED_LPLR:
        result = alternating_mixed_lplr(X=X2, **config.algorithm_kwargs)
        L, R, X_hat = undo_hadamard_lplr(
            L=result[0][0], R=result[0][1], X_hat_or_Q=result[1],
            n=n, d=d,
            HL=H1, HR=H2, col_perm=col_perm, row_perm=row_perm
        )
        result[0] = (L, R)
        result[1] = X_hat

        assert torch.all_close(L @ R, X_hat)
        print(f"Final error: {torch.norm(X - X_hat, p='fro') / torch.norm(X, p='fro')}")

    elif config.algorithm_type == AlgorithmType.DIRECT_SVD_LPLR:
        result = direct_svd_mixed_lplr(X=X2, **config.algorithm_kwargs)
        L, R, X_hat = undo_hadamard_lplr(
            L=result[0][0], R=result[0][1], X_hat_or_Q=result[1],
            n=n, d=d,
            HL=H1, HR=H2, col_perm=col_perm, row_perm=row_perm
        )
        result[0] = (L, R)
        result[1] = X_hat
        assert torch.all_close(L @ R, X_hat)
        print(f"Final error: {torch.norm(X - X_hat, p='fro') / torch.norm(X, p='fro')}")
        
    elif config.algorithm_type == AlgorithmType.LOFTQ:
        from peft.utils.loftq_utils import loftq_init
        result = loftq_init(weight=X2, **config.algorithm_kwargs)
        result = list(result)

        L, R, Q = undo_hadamard_lplr(
            L=result[2], R=result[1], X_hat_or_Q=result[0], n=n, d=d,
            HL=H1, HR=H2, col_perm=col_perm, row_perm=row_perm
        )

        result[0] = Q
        result[1] = R
        result[2] = L
        print(f"Final error: {torch.norm(X - Q - L @ R, p='fro') / torch.norm(X, p='fro')}")
    elif config.algorithm_type == AlgorithmType.LOFTQ_LPLR:
        from peft.utils.loftq_lplr_utils import loftq_lplr_init
        result = loftq_lplr_init(weight=X2, **config.algorithm_kwargs)
        result = list(result)
        
        L, R, Q = undo_hadamard_lplr(
            L=result[2], R=result[1], X_hat_or_Q=result[0], n=n, d=d,
            HL=H1, HR=H2, col_perm=col_perm, row_perm=row_perm
        )

        result[0] = Q
        result[1] = R
        result[2] = L
        print(f"Final error: {torch.norm(X - Q - L @ R, p='fro') / torch.norm(X, p='fro')}")
    else:
        raise NotImplementedError("Other algorithm types not supported yet.")
    
    return tuple(result)

def apply_hadamard(X):
    n, d = X.shape
    H1_dim = int(2**np.ceil(np.log2(n)))
    H2_dim = int(2**np.ceil(np.log2(d)))

    H1 = torch.from_numpy(hadamard(H1_dim)).to(X.device).float()
    # H1 *= rademacher_rv(H1_dim).to(X.device) # multiplication by diagonal matrix

    row_perm = torch.arange(n)
    if n < H1_dim:
        # zero-pad X
        X = torch.cat((X, torch.zeros(H1_dim-n, d).to(X.device)), dim=0)
        row_perm = torch.argsort(torch.rand(H1_dim))
        X = X[row_perm, :]

    H1 *= 1/np.sqrt(H1_dim)

    H2 = torch.from_numpy(hadamard(H2_dim)).to(X.device).float()
    # H2 *= rademacher_rv(H2_dim).to(X.device) # multiplication by diagonal matrix

    col_perm = torch.arange(d)
    if d < H2_dim:
        X = torch.cat((X, torch.zeros(n, H2_dim-d).to(X.device)), dim=1)
        col_perm = torch.argsort(torch.rand(H2_dim))
        X = X[:, col_perm]

    H2 *= 1/np.sqrt(H2_dim)

    return H1 @ X @ H2

def make_sparse(A, sparsity):
    A_flat = torch.abs(A.flatten())
    nth_largest = torch.topk(A_flat, sparsity).values[-1]
    return A * (torch.abs(A) > nth_largest)

class IterativeWeightDecomposition:
    def __init__(
        self,
        X: torch.Tensor,
        compute_low_rank_factors: bool = True,
        compute_quantized_component: bool = True,
        compute_sparse_component: bool = True,
        bits_factors: Union[int, list[int]] = 8,
        bits_quant: int = 4,
        bits_sparse: int = 4,
        num_factors: Union[int, list[int]] = 16,
        sparse_ratio_nonzeros: int = 25,
        quantizer_factory: QuantizerFactory = QuantizerFactory(),
        rand_svd: bool = False,
        rand_svd_oversampling: int = 25,
        iters: int = 50,
        log_errors: bool = False,
        error_norm: ErrorMetric = FroError(),
        verbose: bool = True
    ):
        self.X = X
        self.compute_low_rank_factors = compute_low_rank_factors
        self.compute_quantized_component = compute_quantized_component
        self.compute_sparse_component = compute_sparse_component

        self.rand_svd = rand_svd
        self.rand_svd_oversampling = rand_svd_oversampling
        self.iters = iters
        self.log_errors = log_errors
        self.error_norm = error_norm
        self.verbose = verbose

        if compute_quantized_component:
            self.Q_quantizer = quantizer_factory.get_quantizer(bits_quant, device=X.device)

        self.L = torch.zeros_like(X)
        self.R = torch.zeros_like(X)
        if compute_low_rank_factors:
            self.num_factors = num_factors
            self.bits_factors = bits_factors

            if isinstance(bits_factors, list):
                self.mixed_precision = True
                self.total_rank = sum(num_factors)
                self.LR_quantizer = [
                    quantizer_factory.get_quantizer(B, device=X.device) \
                        for B in bits_factors
                ]
            else:
                self.mixed_precision = False
                self.total_rank = num_factors
                self.LR_quantizer = quantizer_factory.get_quantizer(bits_factors, device=X.device)

        if compute_sparse_component:
            self.bits_sparse = bits_sparse
            self.num_nonzeros = int(sparse_ratio_nonzeros * X.shape[0] * X.shape[1])
            self.S_quantizer = quantizer_factory.get_quantizer(bits_sparse, device=X.device)

        self.set_initial_condition()

        self.errors = []

    def set_initial_condition(self):
        self.Q = torch.zeros_like(self.X)
        if self.compute_quantized_component:
            self.Q = simulated_quant(self.Q_quantizer, self.X)

        self.L = torch.zeros_like(self.X)
        self.R = torch.zeros_like(self.X)
        if self.compute_low_rank_factors:
            U, Sigma, VT = torch.svd_lowrank(self.X - self.Q, 
                                self.total_rank + self.rand_svd_oversampling) if self.rand_svd \
                                else torch.linalg.svd(self.X - self.Q, full_matrices=False)
            sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:self.total_rank]))
            self.L = self.quantize_factor(U[:,:self.total_rank] @ sqrt_Sigma)
            self.R = self.quantize_factor(sqrt_Sigma @ VT[:,:self.total_rank],
                                          transposed=True)
        
        self.S = torch.zeros_like(X)

    def quantize_factor(self, factor, transposed=False):
        if self.bits_factors >= 16:
            return factor
        if self.mixed_precision:
            return mixed_precision_quantize(
                    factor, self.num_factors, self.LR_quantizer,
                    transposed=transposed
                )
        return simulated_quant(self.LR_quantizer, factor,
                               transposed=transposed)
    
    def get_updated_Q(self, residual):
        if not self.compute_quantized_component:
             return self.Q
        return simulated_quant(self.Q_quantizer, residual)

    def quantized_least_squares(self, A, B, default=None):
        Y = torch.linalg.lstsq(A, B)[0]
        if torch.isnan(Y).any().item():
            # try pinv instead; it might work even if R is ill-conditioned
            Y = torch.linalg.pinv(A) @ B
        if not torch.isnan(Y).any().item():
            return self.quantize_factor(Y)
        elif self.verbose:
            logger.error(f"NaNs encountered in finding unquantized factor.")
        return default
        
    def get_updated_LR(self, residual):
        if not self.compute_low_rank_factors:
            return self.L, self.R
        
        L = self.quantized_least_squares(self.R.T, residual.T,
                                         default=L.T).T
        R = self.quantized_least_squares(self.L, residual,
                                         default=R)
        
        return L, R
    
    def get_updated_S(self, residual):
        if not self.compute_sparse_component:
            return self.S
        
        S = make_sparse(residual, self.num_nonzeros)
        if self.bits_sparse < 16:
            S = simulated_quant(self.S_quantizer, S)
        return S
    
    def get_X_hat(self):
        return self.Q + self.S + self.L @ self.R
    
    def iter(self):
        self.Q = self.get_updated_Q(self.X - self.L @ self.R - self.S)
        self.L, self.R = self.get_updated_LR(self.X - self.Q - self.S)
        self.S = self.get_updated_S(self.X - self.Q - self.L @ self.R)

        if self.log_errors:
            X_hat = self.get_X_hat()
            self.errors.append(
                self.error_norm.error(X_hat=X_hat, X_exact=self.X)
            )

    def run(self, use_tqdm=True):
        to_iter = range(self.iters)
        if use_tqdm:
            to_iter = tqdm(to_iter)
        for _ in to_iter:
            self.iter()

# def alternating_mixed_lplr(   
#     X: torch.Tensor = None,
#     k: int = None,
#     r1: int = None,
#     r2: int = None,
#     B1: int = 8,
#     B2: int = 8,
#     quantizer_factory: QuantizerFactory = QuantizerFactory(),
#     normalize_and_shift=False,
#     iters=10,
#     max_cond=5e6,
#     use_tqdm=False,
#     log_errors=False
# ):
#     """
#     Obtain a low-precision low-rank approximation of X using alternating optimization of
#     left and right low rank factors

#     X (torch.Tensor, optional): Input matrix (Tall or square)
#     k: (int, optional): Target rank; Trailing (X.shape[1] - k) singular vectors
#         will be dropped.
#     r1 (int, optional): No. of singular vectors to be kept in full precision
#         for the first factor; Trailing (k - r) singular vectors will be quantized.
#     r2 (int, optional): No. of singular vectors to be kept in full precision
#         for the second factor
#     B1 (int, optional): Bit-budget for first low-rank factor
#     B2 (int, optional): Bit-budget for second low-rank factor
#     quant_type (QuantType, optional): specifies whether to use uniform or 
#         NormalFloat quantization. Member of the QuantType Enum.
#     normalize_and_shift (bool, optional): Maintain additional scalars for
#         better approximation
#     iters (int, optional): Number of iterations of alternating optimization
#     max_cond (float, optional): if the condition number of the rank-k
#         approximation of X is above this number, reduce k such that Xk is
#         well-conditioned. 
#     [TODO] sketch (Sketch, optional): Sketch type
#     log_errors (bool, optional): Return fro-norm errors for each iteration

#     Outputs:
#     - Tuple[torch.Tensor, torch.Tensor] : Low rank quantized factors (L, R)
#     - torch.Tensor: Approximation of X (LR)
#     - [only if log_errors] list[float]: fro-norm errors for each iteration.
#     """
    
#     # logger.info(f"LPLR: k={k}, r={r1}, B={B1}, quantizer_factory={quantizer_factory}, iters={iters}")

#     assert (
#         X.shape[0] >= X.shape[1]
#     ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"
#     assert (
#         k >= r1 and k >= r2
#     ), "No. of singular vectors to be in full precision (r) should be less than or equal to target rank (k)"

#     quantizer_B1 = quantizer_factory.get_quantizer(B1, device=X.device)
#     quantizer_B2 = quantizer_factory.get_quantizer(B2, device=X.device)

#     # Compute full SVD
#     U, S, VT = torch.linalg.svd(X.float(), full_matrices=False)

#     # If the rank-k approximation of X is ill-conditioned, then the least squares
#     # steps might be inaccurate (or, for the CUDA implementation, return NaNs). We
#     # can avoid this by resetting k if necessary.
#     if S[0] / S[k] >= max_cond:
#         k = torch.sum(S > S[0] / max_cond)
#         r1 = min(r1, k)
#         r2 = min(r2, k)
#         logger.warning(f"Could not find k non-negligible singular values of X, so setting k to {k}.")
    
#     U = U[:, 0:k]
#     S = S[0:k]
#     VT = VT[0:k, :]
#     S_sqrt = torch.diag(torch.sqrt(S))

#     # Get the initial left low-rank factor
#     L = quantize_small_sv_components(U @ S_sqrt, r1, quantizer=quantizer_B1)
#     if torch.isnan(L).any().item():
#         logger.error(f"NaNs encountered in quantizing first factor")

#     # Get the initial right low-rank factor
#     # Get the right low-rank factor
#     W = torch.linalg.lstsq(L.float(), X.float())[0]

#     if torch.isnan(W).any().item():
#         logger.error(f"NaNs encountered in finding unquantized R.")

#     R = quantize_small_sv_components(W.T, r2, quantizer=quantizer_B2).T
#     if torch.isnan(R).any().item():
#         logger.error(f"NaNs encountered in quantizing second factor")

#     errors = [torch.norm(X - L @ R, p="fro").item()]

#     best_error = errors[0]
#     best_mtxs = (L, R)

#     to_iter = range(1, iters)
#     if use_tqdm:
#         to_iter = tqdm(to_iter)
#     for _ in to_iter:
#         # Get the left low-rank factor
#         Y = torch.linalg.lstsq(R.float().T, X.float().T)[0].T

#         if torch.isnan(Y).any().item():
#             logger.error(f"NaNs encountered in finding unquantized L. Giving up.")
#             break

#         L = quantize_small_sv_components(Y, r1, quantizer=quantizer_B1)
#         if torch.isnan(L).any().item():
#             logger.error(f"NaNs encountered in Q(L). Giving up.")
#             break

#         # Get the right low-rank factor
#         W = torch.linalg.lstsq(L.float(), X.float())[0]

#         if torch.isnan(W).any().item():
#             logger.error(f"NaNs encountered in finding unquantized R. Giving up.")
#             break

#         R = quantize_small_sv_components(W.T, r2, quantizer=quantizer_B2).T
#         if torch.isnan(R).any().item():
#             logger.error(f"NaNs encountered in Q(R). Giving up.")
#             break 

#         errors.append(torch.norm(X - L @ R, p="fro").item())
#         if errors[-1] < best_error:
#             best_error = errors[-1]
#             best_mtxs = (L, R)

#     L, R = best_mtxs
#     out = L @ R
#     if normalize_and_shift:
#         out = normalize_and_shift_wrt_inner_prod(X, L @ R)

#     if torch.isnan(out).any().item():
#         logger.error(f"NaNs encountered in LPLRed matrix")

#     if log_errors:
#         return (L, R), out, errors

#     return (L, R), out

def direct_svd_mixed_lplr(
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
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

    quantizer_B1 = quantizer_factory.get_quantizer(B1, device=X.device)
    quantizer_B2 = quantizer_factory.get_quantizer(B2, device=X.device)

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

def weight_decomposition_admm(
    X: torch.Tensor,
    rank: int = None,
    sparsity: int = None,
    admm_rho: float = 1,
    BQ: int = 4,
    BLR: int = 16,
    BS: int = 4,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
    iters: int = 50,
    admm_type: int = ADMMType.ADMM_Q,
    admm_r_update_p: int = 0.75,
    admm_s_beta: int = 1,
    log_errors: bool = False,
    error_norm: ErrorMetric = FroError(),
    verbose: bool = False
):
    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"

    ###########################################################################
    ## Optimization variables:
    ## - Q: quantized matrix
    ## - Q': real-valued matrix, constrained to be = Q
    ## - L, R: rank-m matrices such that LR is the size of X
    ## - S: k-sparse matrix
    ###########################################################################
    ## Optimization problem:
    ## min 1/2 ||X - LR - Q - S||^2_F s.t. Q=Q'
    ###########################################################################
    ## Initialization: similar to LoftQ
    ## - Q = Q' = Quant(X)
    ## - L, R via SVD
    ## - S = 0
    ###########################################################################
    # Note: by https://arxiv.org/pdf/2009.03482.pdf, for convergence
    # guarantees, we want rho >= the Lipschitz constant of the gradient w.r.t.
    # Q of 1/2 ||X - LR - Q - S||^2_F, which is just 1.
    
    quant_Q = quantizer_factory.get_quantizer(BQ, device=X.device)
    quant_LR = quantizer_factory.get_quantizer(BLR, device=X.device)
    quant_S = quantizer_factory.get_quantizer(BS, device=X.device)

    Q = simulated_quant(quant_Q, X)
    Qp = Q

    U, Sigma, VT = torch.linalg.svd(X - Qp)
    sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:rank]))
    L = U[:, :rank] @ sqrt_Sigma
    R = sqrt_Sigma @ VT[:rank, :]

    S = torch.zeros_like(X).to(X.device)
    lmbda = torch.zeros_like(X).to(X.device)

    errors = []
    errors_Qp = []
    lagrangians = []
    constraint_vals = []

    for _ in tqdm(range(iters)):
        ## Update Q'
        Qp = 1/(1 + admm_rho) * (X - L @ R - S - lmbda + admm_rho * Q)

        # Update Q
        # There are three different update rules, based on the admm_type
        # See https://arxiv.org/pdf/2009.03482.pdf for more details
        if admm_type == ADMMType.ADMM_Q:
            # regular projected ADMM step
            Q = simulated_quant(quant_Q, lmbda / admm_rho + Qp)
        elif admm_type == ADMMType.ADMM_R:
            # Randomized ADMM step; only update some parameters
            m = torch.rand_like(Q) < admm_r_update_p
            Q_new = simulated_quant(quant_Q, lmbda / admm_rho + Qp)
            Q = Q_new * m + Q * (~m)
        elif admm_type == ADMMType.ADMM_S:
            # Soft projection
            Q_update = lmbda / admm_rho + Qp
            Q_update_proj = simulated_quant(quant_Q, Q_update)
            Q_d = Q_update_proj - Q_update 
            use_quant_bitmask = torch.norm(Q_d, p="fro") < admm_s_beta / admm_rho
            Q = use_quant_bitmask * Q_update_proj + \
                (~use_quant_bitmask) * (Q_update + admm_s_beta * Q_d / \
                                        (admm_rho * torch.norm(Q_d, p="fro")))

        # Update S
        if sparsity > 0:
            S = make_sparse(X - L @ R - Qp, sparsity)
            S = simulated_quant(quant_S, S)

        # Update L, R
        Y = torch.linalg.lstsq(R.T, (X - Qp - S).T)[0].T
        if not torch.isnan(Y).any().item():
            L = simulated_quant(quant_LR, Y)
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized L.")

        W = torch.linalg.lstsq(L, X - Qp - S)[0]
        # W = torch.linalg.pinv(L) @ (X - Qp - S)
        if not torch.isnan(W).any().item():
            R = simulated_quant(quant_LR, W)
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized R.")

        # Update lambda
        lmbda = lmbda + admm_rho * (Qp - Q)

        lagrangian = 1/2 * torch.norm(X - L @ R - Qp - S, p="fro").item() ** 2 + \
            torch.trace(lmbda.T @ (Qp - Q)).item() + admm_rho/2 * \
                torch.norm(Qp - Q, p="fro").item() ** 2
        error = error_norm.error(X_hat = L@R + simulated_quant(quant_Q, Q) + S,
                                 X_exact=X)
        error_Qp = error_norm.error(X_hat=L @ R + Qp + S,
                                    X_exact=X)
        
        constraint_vals.append(torch.norm(Q - Qp, p="fro").item())                
        lagrangians.append(lagrangian)
        errors.append(error)
        errors_Qp.append(error_Qp)
    out = L @ R + Q + S

    if log_errors:
        return (L, R, Q, S), out, (errors, errors_Qp, lagrangians, constraint_vals)
    return (L, R, Q, S), out
   
def weight_decomposition(
    X: torch.Tensor,
    ranks: list[int] = None,
    sparsity: int = None,
    BQ: int = 4,
    BLR_list: list[int] = [16],
    BS: int = 16,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
    iters: int = 50,
    log_errors: bool = False,
    error_norm: ErrorMetric = FroError(),
    verbose: bool = True
):
    assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"


    if BQ > 0:
        quant_Q = quantizer_factory.get_quantizer(BQ, device=X.device)
    quant_LRs = [
        quantizer_factory.get_quantizer(B, device=X.device) \
            for B in BLR_list
    ]
    quant_S = quantizer_factory.get_quantizer(BS, device=X.device)

    if BQ > 0:
        Q = simulated_quant(quant_Q, X)
    else:
        Q = torch.zeros_like(X)

    k = sum(ranks)
    U, Sigma, VT = torch.linalg.svd(X - Q)
    sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:k]))
    L = mixed_precision_quantize(U[:,:k] @ sqrt_Sigma, ranks, quant_LRs)
    R = mixed_precision_quantize(VT.T[:, :k] @ sqrt_Sigma, ranks, quant_LRs).T

    S = torch.zeros_like(X).to(X.device)

    errors = []

    for _ in tqdm(range(iters)):
        # Update Q
        if BQ > 0:
            Q = simulated_quant(quant_Q, X - L @ R - S)

        # Update L, R
        Y = torch.linalg.lstsq(R.T, (X - Q - S).T)[0].T
        # Y = (X - Q - S) @ torch.linalg.pinv(R)
        if not torch.isnan(Y).any().item():
            L = mixed_precision_quantize(Y, ranks, quant_LRs)
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized L.")

        W = torch.linalg.lstsq(L, X - Q - S)[0]
        # W = torch.linalg.pinv(L) @ (X - Q - S)
        if not torch.isnan(W).any().item():
            R = mixed_precision_quantize(W, ranks, quant_LRs, transposed=True)  
        elif verbose:
            logger.error(f"NaNs encountered in finding unquantized R.")

        # Update S
        if sparsity > 0:
            S = make_sparse(X - L @ R - Q, sparsity)
            if BS < 16:
                S = simulated_quant(quant_S, S)

        error = error_norm.error(X_hat=L@R + Q + S, X_exact=X)
        
        errors.append(error)
    out = L @ R + Q + S

    if log_errors:
        return (L, R, Q, S), out, (errors)
    return (L, R, Q, S), out
