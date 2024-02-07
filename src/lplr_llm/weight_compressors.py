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
import matplotlib.pyplot as plt

def rademacher_rv(*size):
    """
    Returns a pytorch Tensor of specified size where each element is i.i.d.
    Uniform({-1, +1}).
    """
    rand = torch.rand(*size)
    return 1.0 * (rand > 0.5) - 1.0 * (rand <= 0.5)

def undo_hadamard(
        X: torch.Tensor,
        n: int,
        d: int,
        HL: torch.Tensor = None,
        HR: torch.Tensor = None,
        col_perm: torch.Tensor = None,
        row_perm: torch.Tensor = None
    ):
    """
    Undoes an orthogonal Hadamard transform.
    
    X (torch.Tensor): Hadamard-transformed tensor.
    n (int): number of rows in the original matrix.
    d (int): number of columns in the original matrix.
    HL (torch.Tensor, optional): orthonormal (potentially randomized) Hadamard
        matrix applied to the left. None if only a right transform was
        performed.
    HR (torch.Tensor, optional): orthonormal (potentially randomized) Hadamard
        matrix applied to the right. None if only a left transform was
        performed.
    col_perm (torch.Tensor, optional): permutation (as a vector of indices)
        applied to the columns, if any.
    row_perm (torch.Tensor, optional): permutation, if any, applied to the rows.
    """
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

def apply_hadamard(X: torch.Tensor):
    """
    Applies a non-randomized Hadamard transform to the left and right of X.
    If the dimensions of X are not powers of two, rows and/or columns of zeros
    are shuffled into X until the dimensions are the next highest powers of two.
    """
    n, d = X.shape
    H1_dim = int(2**np.ceil(np.log2(n)))
    H2_dim = int(2**np.ceil(np.log2(d)))

    H1 = torch.from_numpy(hadamard(H1_dim)).to(X.device).float()

    row_perm = torch.arange(n)
    if n < H1_dim:
        # zero-pad X
        X = torch.cat((X, torch.zeros(H1_dim-n, d).to(X.device)), dim=0)
        row_perm = torch.argsort(torch.rand(H1_dim))
        X = X[row_perm, :]

    H1 *= 1/np.sqrt(H1_dim)

    H2 = torch.from_numpy(hadamard(H2_dim)).to(X.device).float()

    col_perm = torch.arange(d)
    if d < H2_dim:
        X = torch.cat((X, torch.zeros(n, H2_dim-d).to(X.device)), dim=1)
        col_perm = torch.argsort(torch.rand(H2_dim))
        X = X[:, col_perm]

    H2 *= 1/np.sqrt(H2_dim)

    return H1 @ X @ H2

def make_sparse(A, sparsity):
    """
    Performs absolute top-k sparsification, where the sparsity argument is k.
    Returns a version of A where all but the largest (in absolute magnitude) k
    elements are zeroed out.
    """
    A_flat = torch.abs(A.flatten())
    nth_largest = torch.topk(A_flat, sparsity).values[-1]
    return A * (torch.abs(A) > nth_largest)

@dataclass
class LPLRParameters:
    split_quantization: bool = field(
        default=False, metadata={ "help": (
            "Whether to quantize the left and right components with different "
            "parameters. If False, \"bits\" and \"rank\" are used for both "
            "factors. If True, \"bits_left\", \"bits_right\, etc., are used."
        )}
    )
    bits: Union[int, list[int]] = field(
        default=8, metadata={ "help": (
            "Quantization bits for both factors. For mixed-precision"
            "quantization, pass in a list"
        )}
    )
    rank: Union[int, list[int]] = field(
        default=16, metadata={ "help": (
            "Rank for both factors. For mixed-precision quantization, pass in "
            "a list, which has a one-to-one correspondence with the list for "
            "the \"bits\" parameter. For instance, if rank=[16, 64] and "
            "bits=[8, 4], the first 16 singular vectors will be 8-bit "
            "quantized and next 64 will be 4-bit quantized."
        )}
    )
    bits_left: Union[int, list[int]] = field(
        default=8, metadata={ "help": "Quantization bits for left factor."}
    )
    bits_right: Union[int, list[int]] = field(
        default=8, metadata={ "help": "Quantization bits for right factor."}
    )
    rank_left: Union[int, list[int]] = field(
        default=16, metadata={ "help": (
            "Rank for left factor. If an integer, must equal rank_right. If a"
            "list, sum(rank_left) must match sum(rank_right)."
        )}
    )
    rank_right: Union[int, list[int]] = field(
        default=16, metadata={ "help": (
            "Rank for right factor. If an integer, must equal rank_left. If a"
            "list, sum(rank_left) must match sum(rank_right)."
        )}
    )
    iters: int = field(
        default=1, metadata={ "help": (
            "Number of alternating least squares steps per L, R update."
        )}
    )

@dataclass
class IterativeWeightDecompositionParams:
    compute_quantized_component: bool = field(
        default=True, metadata={ "help": (
            "Whether the decomposition should include a quantized full-size"
            "component (denoted Q)."
        )}
    )
    compute_low_rank_factors: bool = field(
        default=True, metadata={ "help": (
            "Whether the decomposition should include low-rank factors (L, R)."
        )}
    )
    compute_sparse_component: bool = field(
        default=False, metadata={ "help": (
            "Whether the decomposition should include a sparse, potentially "
            "quantized component (denoted S)."
        )}
    )
    lplr_params: LPLRParameters = field(
        default_factory=LPLRParameters, metadata={ "help": "Parameters for the low-rank factors."}
    )
    bits_quant: int = field(
        default=4, metadata={ "help": "Bits of precision for Q."}
    )
    bits_sparse: int = field(default=4, metadata={ "help": (
            "Bits of precision for S."
        )}
    )
    sparse_ratio_nonzeros: int = field(
        default=25, metadata={ "help": (
            "Fraction of the elements of S that should be nonzero."
        )}
    )
    quantizer_factory: QuantizerFactory = field(
        default_factory=QuantizerFactory, metadata={ "help": (
            "QuantizerFactory (from lplr_llm.quantizers) object used to "
            "instantiate quantizers."
        )}
    )
    rand_svd: bool = field(
        default=False, metadata={ "help": (
            "Whether to use a randomized SVD. This is fast but will be less "
            "accurate than a full SVD."
        )}
    )
    rand_svd_oversampling: int = field(
        default=25, metadata={ "help": (
            "If using a randomized SVD, by how much to oversample."
        )}
    )
    iters: int = field(default=50, metadata={ "help": (
            "Number of iterations for the algorithm."
        )}
    )
    log_errors: bool = field(
        default=True, metadata={ "help": (
            "Whether to compute approximation error for each iteration."
        )}
    )
    error_norm: ErrorMetric = field(
        default_factory=FroError, metadata={ "help": (
            "Object from lplr_llm.error_metrics, used to compute different "
            "metrics of approximation error (e.g., Frobenius vs. Spectral norm)."
        )}
    )

class IterativeWeightDecomposition:
    """
    Decomposes a weight matrix into any combination of the following:
        - Q: a quantized full-rank matrix
        - L, R: low-rank factors, potentially quantized
        - S: a sparse matrix, potentially quantized
    See IterativeWeightDecompositionParams and LPLRParameters for information
    on algorithm parameters.

    Usage example:
    ```
    weight_decomp = IterativeWeightDecomposition(
        X=your_weight_matrix,
        params = IterativeWeightDecompositionParams(
            compute_quantized_component=True,
            compute_low_rank_factors=True,
            compute_sparse_component=False,
            bits_quant=2,
            lplr_params=LPLRParameters(
                bits=[8, 4],
                rank=[16, 128]
            ),
            iters=30,
            error_norm=RandSpectralError()
        )
    )

    # Run the algorithm for the specified number of iterations
    weight_decomp.run()

    # Get the components as a dictionary with keys "Q", "L", "R", "S"
    components = weight_decomp.get_components()

    # Get the per-iteration relative spectral error
    errors = weight_decomp.errors

    # Plot the errors
    weight_decomp.plot_errors()

    # Get the approximation of the weight matrix
    X_hat = weight_decomp.get_X_hat()
    ```
    """
    def __init__(
        self,
        X: torch.Tensor,
        label: str = None,
        params: IterativeWeightDecompositionParams = \
            IterativeWeightDecompositionParams()
    ):
        """
        Initialize the iterative weight decomposition algorithm (instantiate
        quantizers and set the initial condition).

        X (torch.Tensor): input weight matrix to decompose.
        label (string): used as part of the title when plotting.
            Examples: None, "LoftQ", "LoftQ-LPLR", "LoftQ-LPLR + Sparse"
        params (IterativeWeightDecompositionParams): algorithm parameters
        """
        device = X.device

        self.X = X
        self.label = label
        self.params = params

        # Set up quantizers
        if params.compute_quantized_component:
            self.Q_quantizer = params.quantizer_factory.get_quantizer(
                params.bits_quant, device=device
            )

        if params.compute_low_rank_factors:
            if params.lplr_params.split_quantization:
                self.rank_factors = {
                    "left": params.lplr_params.rank_left,
                    "right": params.lplr_params.rank_right
                }
                self.bits_factors = {
                    "left": params.lplr_params.bits_left,
                    "right": params.lplr_params.bits_right,
                }
            else:
                self.rank_factors = {
                    "left": params.lplr_params.rank,
                    "right": params.lplr_params.rank
                }
                self.bits_factors = {
                    "left": params.lplr_params.bits,
                    "right": params.lplr_params.bits
                }
            
            self.mixed_precision = {}
            total_rank = {}
            self.LR_quantizer = {}
            for l_or_r in ["left", "right"]:
                if type(self.bits_factors[l_or_r]) != int:
                    self.mixed_precision[l_or_r] = True
                    total_rank[l_or_r] = sum(self.rank_factors[l_or_r])
                    self.LR_quantizer[l_or_r] = [
                        params.quantizer_factory.get_quantizer(B, device=device) \
                            for B in self.bits_factors[l_or_r]
                    ]
                else:
                    self.mixed_precision[l_or_r] = False
                    total_rank[l_or_r] = self.rank_factors[l_or_r]
                    self.LR_quantizer[l_or_r] = \
                        params.quantizer_factory.get_quantizer(
                            self.bits_factors[l_or_r], device=device
                        )
            assert total_rank["left"] == total_rank["right"]
            self.total_rank = total_rank["left"]

        if params.compute_sparse_component:
            self.bits_sparse = params.bits_sparse
            self.num_nonzeros = int(params.sparse_ratio_nonzeros * X.shape[0] * X.shape[1])
            self.S_quantizer = params.quantizer_factory.get_quantizer(
                params.bits_sparse, device=device
            )

        # Set the initial condition for the alternating algorithm
        self._set_initial_condition()
        self.best_error = self._get_current_error()
        self.best_components = self.get_components()

        self._errors = []
    
    def get_components(self):
        return {
            "L": self.L, "R": self.R, "Q": self.Q, "S": self.S
        }
    
    def get_best_components(self):
        return self.get_best_components

    @property
    def errors(self):
        """
        Access the per-iteration approximation errors
        """
        return self._errors
    
    def plot_errors(self):
        """
        Plot the per-iteration approximation errors
        """
        plt.figure(figsize=(12, 4))
        title = f"Relative {self.params.error_norm.name} per iteration"
        if self.label is not None:
            title = f"{title} ({self.label})"
        plt.title(title)
        plt.plot(self._errors, marker='o', linestyle='-', color='b')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def _set_initial_condition_LR(self):
        # Low-rank factors
        self.L = torch.zeros_like(self.X)
        self.R = torch.zeros_like(self.X)
        if self.params.compute_low_rank_factors:
            # Either randomized or normal SVD
            if self.params.rand_svd:
                U, Sigma, V = torch.svd_lowrank(
                    self.X - self.Q, 
                    self.total_rank + self.params.rand_svd_oversampling
                )
                VT = V.T
            else:
                U, Sigma, VT = torch.linalg.svd(self.X - self.Q, full_matrices=False)
            sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:self.total_rank]))

            self.L = self._quantize_factor(U[:,:self.total_rank] @ sqrt_Sigma,
                                           left_or_right="left")
            self.R = self._quantized_least_squares(
                self.L, self.X - self.Q,
                default=sqrt_Sigma @ VT[:self.total_rank, :],
                left_or_right="right"
            )

    def _set_initial_condition(self):
        """
        Set the initial condition for Q, L, R, and/or S. Matrices that are not
        used are set to all zeros.
        """

        # Quantized matrix
        self.Q = self._get_updated_Q(self.X)

        self._set_initial_condition_LR()
        
        # Sparse matrix
        self.S = torch.zeros_like(self.X)

    def _quantize_factor(self, factor, left_or_right="left"):
        """
        Quantization logic for L and R. Returns the quantized factor.
        """
        assert left_or_right in ["left", "right"]
        transposed = left_or_right == "right"

        if not self.mixed_precision[left_or_right] and \
            self.bits_factors[left_or_right] >= 16:
            return factor
        if self.mixed_precision[left_or_right]:
            return mixed_precision_quantize(
                    factor, self.rank_factors[left_or_right],
                    self.LR_quantizer[left_or_right],
                    transposed=transposed
                )
        return simulated_quant(
            self.LR_quantizer[left_or_right], factor
        )
    
    def _get_updated_Q(self, residual):
        """
        Returns the quantized residual.
        """
        if not self.params.compute_quantized_component:
             return torch.zeros_like(residual)
        return simulated_quant(self.Q_quantizer, residual)

    def _quantized_least_squares(self, A, B, left_or_right="left", default=None):
        """
        Performs least squares to get Y such that AY approx B for the right
        factor or YA approx B for the left factor, and then
        returns the quantized version of Y. If least squares fails, returns
        the object passed into the "default" parameter.
        """
        assert left_or_right in ["left", "right"]
        if left_or_right == "left":
            A = A.T
            B = B.T
        Y = torch.linalg.lstsq(A, B)[0]
        if torch.isnan(Y).any().item():
            # try pinv instead; it might work even if R is ill-conditioned
            Y = torch.linalg.pinv(A) @ B

        if left_or_right == "left":
            Y = Y.T

        if not torch.isnan(Y).any().item():
            return self._quantize_factor(Y, left_or_right=left_or_right)
        logger.error(f"NaNs encountered in finding unquantized factor.")
        return default
        
    def _get_updated_LR(self, residual):
        """
        Performs one update step for the low-rank factors and returns the
        result as the tuple (L, R).
        """
        if not self.params.compute_low_rank_factors:
            return torch.zeros_like(residual), torch.zeros_like(residual)
        
        best_LR = (self.L, self.R)
        best_err = float('inf')

        for _ in range(self.params.lplr_params.iters):
            L = self._quantized_least_squares(self.R, residual,
                                            default=self.L, 
                                            left_or_right="left")
            R = self._quantized_least_squares(self.L, residual,
                                            default=self.R,
                                            left_or_right="right")
            err = self.params.error_norm.error(
                X_hat=L@R, X_exact=residual
            )
            if err < best_err:
                best_err = err
                best_LR = (L, R)
        
        return best_LR
    
    def _get_updated_S(self, residual):
        """
        Performs one update step for the sparse component and returns the
        result.
        """
        if not self.params.compute_sparse_component:
            return torch.zeros_like(residual)
        
        S = make_sparse(residual, self.num_nonzeros)
        if self.bits_sparse < 16:
            S = simulated_quant(self.S_quantizer, S)
        return S
    
    def get_X_hat(self):
        """
        Returns the estimated weight matrix.
        """
        return self.Q + self.S + self.L @ self.R
    
    def get_best_X_hat(self):
        """
        Returns the estimated weight matrix.
        """
        mtxs = self.get_best_components()
        return mtxs["Q"] + mtxs["S"] + mtxs["L"] @ mtxs["R"]
    

    def _get_current_error(self):
        X_hat = self.get_X_hat()
        return self.params.error_norm.error(X_hat=X_hat, X_exact=self.X)

    def _update_best_and_log_errors(self):
        err = self._get_current_error()
        if err < self.best_error:
            self.best_error = err
            self.best_components = self.get_components()

        if self.params.log_errors:
            self._errors.append(err)
    
    def iter(self):
        """
        Performs one update step for all matrices (Q, L, R, S).
        """
        self.Q = self._get_updated_Q(self.X - self.L @ self.R - self.S)
        self.L, self.R = self._get_updated_LR(self.X - self.Q - self.S)
        self.S = self._get_updated_S(self.X - self.Q - self.L @ self.R)

        self._update_best_and_log_errors()

    def run(self, use_tqdm=True):
        """
        Runs the weight decomposition algorithm for the specified number of
        iterations.
        """
        to_iter = range(self.params.iters)
        if use_tqdm:
            to_iter = tqdm(to_iter)
        for _ in to_iter:
            self.iter()

class HadamardWeightDecomposition(IterativeWeightDecomposition):
    """
    Applied a randomized Hadamard transform to the left and right of the weight
    matrix before runing the iterative weight decomposition algorithm.

    See IterativeWeightDecomposition for more details.

    Note: get_untransformed_components and get_X_hat correspond to the
    Hadamard-embedded weight matrix. Use get_untransformed_components and
    get_X_hat_untransformed instead.
    """
    def __init__(
        self,
        X: torch.Tensor,
        label: str = "Hadamard",
        params: IterativeWeightDecompositionParams = \
            IterativeWeightDecompositionParams()
    ):
        """
        Applies a randomized Hadamard transform to the left and right of X
        before computing the initial condition of the weight decomposition
        algorithm.

        If the dimensions of X are not powers of two, rows and/or columns of zeros
        are shuffled into X until the dimensions are the next highest powers of two.
        """
        self.untransformed_X = X
        n, d = X.shape

        # Next highest power of 2
        H1_dim = int(2**np.ceil(np.log2(n)))
        H2_dim = int(2**np.ceil(np.log2(d)))

        self.H1_dim = H1_dim
        self.H2_dim = H2_dim

        # Compute the left transform
        H1 = torch.from_numpy(hadamard(H1_dim)).to(X.device).float()
        D1 = rademacher_rv(H1_dim).to(X.device)
        self.left_diag = D1
        H1 *= D1

        row_perm = torch.arange(n)
        if n < H1_dim:
            # zero-pad X
            X = torch.cat((X, torch.zeros(H1_dim-n, d).to(X.device)), dim=0)
            row_perm = torch.argsort(torch.rand(H1_dim))
            X = X[row_perm, :]
        self.row_perm = row_perm

        H1 *= 1/np.sqrt(H1_dim)

        # Compute the right transform
        H2 = torch.from_numpy(hadamard(H2_dim)).to(X.device).float()
        D2 = rademacher_rv(H2_dim).to(X.device)
        self.right_diag = D2
        H2 *= D2

        col_perm = torch.arange(d)
        if d < H2_dim:
            X = torch.cat((X, torch.zeros(n, H2_dim-d).to(X.device)), dim=1)
            col_perm = torch.argsort(torch.rand(H2_dim))
            X = X[:, col_perm]
        self.col_perm = col_perm
        H2 *= 1/np.sqrt(H2_dim)

        X = H1 @ X @ H2

        # Initialize the algorithm with the Hadamard-transformed matrix.
        super().__init__(X, label=label, params=params)

    def get_untransformed_components(self, components=None):
        """
        Get the components (Q, L, R, S) corresponding to the original weight
        matrix.
        """
        if components is None:
            components = self.get_components()

        Q = components["Q"]
        L = components["L"]
        R = components["R"]
        S = components["S"]

        H1 = torch.from_numpy(hadamard(self.H1_dim)).to(self.X.device).float()
        H1 *= self.left_diag / np.sqrt(self.H1_dim)

        H2 = torch.from_numpy(hadamard(self.H2_dim)).to(self.X.device).float()
        H2 *= self.right_diag / np.sqrt(self.H2_dim)

        n, d = self.untransformed_X.shape

        return {
            "Q": undo_hadamard(Q, n, d, HL=H1, HR=H2,
                               col_perm=self.col_perm,
                               row_perm=self.row_perm),
            "L": undo_hadamard(L, n=n, d=L.shape[1],
                               HL=H1, row_perm=self.row_perm),
            "R": undo_hadamard(R, R.shape[0], d=d,
                               HR=H2, col_perm=self.col_perm),
            "S": undo_hadamard(S, n, d, HL=H1, HR=H2,
                               col_perm=self.col_perm,
                               row_perm=self.row_perm)
        }
    
    def get_best_untransformed_components(self):
        return self.get_best_untransformed_components(
            components=self.get_best_components()
        )

    def get_X_hat_untransformed(self, best=False):
        """
        Compute the estimate of the weight matrix corresponding to the original
        weight matrix.
        """
        H1 = torch.from_numpy(hadamard(self.H1_dim)).to(self.X.device).float()
        H1 *= self.left_diag / np.sqrt(self.H1_dim)

        H2 = torch.from_numpy(hadamard(self.H2_dim)).to(self.X.device).float()
        H2 *= self.right_diag / np.sqrt(self.H2_dim)

        n, d = self.untransformed_X.shape

        X_hat = self.get_X_hat() if not best else self.get_best_X_hat()
        return undo_hadamard(X_hat, n, d, HL=H1, HR=H2,
                             col_perm=self.col_perm,
                             row_perm=self.row_perm)
    
    def get_best_X_hat_untransformed(self):
        return self.get_X_hat_untransformed(best=True)

@dataclass
class ADMMParameters:
    rho: float = field(
        default=1, metadata={ "help": (
            "Coefficient of the squared constraint term of the augmented "
            "Lagrangian."
        )}
    )
    admm_type: int = field(
        default=ADMMType.ADMM_Q, metadata={ "help": (
            "Member of the ADMMType enum. ADMM_Q is the default formulation "
            "for ADMM with a quantized variable, ADMM_R has randomized "
            " updates, and ADMM_S uses \"soft\" quantization. See the paper "
            "https://arxiv.org/pdf/2009.03482.pdf for more details."
        )}
    )
    admm_r_update_p: float = field(
        default=0.75, metadata={ "help": (
            "Probability that each element of Q is updated for ADMM_R."
        )}
    )
    admm_s_beta: float = field(
        default=1, metadata={ "help": (
            "Coefficient of the soft-quantization term in the ADMM objective. "
            "See https://arxiv.org/pdf/2009.03482.pdf."
        )}
    )
    
class ADMMWeightDecomposition(IterativeWeightDecomposition):
    """
    Weight decomposition is formulated as the following problem:
    ---------------------------------------------------------------------------
    Optimization variables:
    - Q: quantized matrix
     - Q': real-valued matrix, constrained to be = Q
     - L, R: rank-m matrices such that LR is the size of X
     - S: k-sparse matrix
    ---------------------------------------------------------------------------
    Optimization problem:
        min 1/2 ||X - LR - Q - S||^2_F + Indic(Q' is quantized)
            s.t. Q=Q',
    
        where Idic(condition) is 0 if the condition is met and infinity
        otherwise. 
    
    For ADMM-S, Indic(Q' is quantized) is replaced by the distance of Q' to the
    nearest quantized point, scaled by coefficient beta.
    ---------------------------------------------------------------------------
    Initialization: similar to LoftQ
     - Q = Q' = Quant(X)
     - L, R via SVD
     - S = 0
    ---------------------------------------------------------------------------
    Note: by https://arxiv.org/pdf/2009.03482.pdf, for convergence
    guarantees, we want rho >= the Lipschitz constant of the gradient w.r.t.
    Q of 1/2 ||X - LR - Q - S||^2_F, which is just 1.
    ---------------------------------------------------------------------------
    """
    def __init__(
        self,
        X: torch.Tensor,
        label: str = "ADMM",
        params: IterativeWeightDecompositionParams = \
            IterativeWeightDecompositionParams(),
        admm_params: ADMMParameters = ADMMParameters()
    ):
        self.admm_params = admm_params
        super().__init__(X, label=label, params=params)

        self._lagrangians = []
        self._constraint_vals = []

    def _set_initial_condition(self):
        # Quantized matrix
        self.Q = simulated_quant(self.Q_quantizer, self.X)

        self._set_initial_condition_LR()
        
        # Sparse matrix
        self.S = torch.zeros_like(self.X)

        self.Qp = self.Q.clone()
        self.lmbda = torch.zeros_like(self.X)

    def _get_updated_Q(self, residual):
        return (residual - self.lmbda + self.admm_params.rho * self.Qp) / \
                    (1 + self.admm_params.rho)
    
    def _get_updated_Qp(self, Q):
        """
        Update Qp, which is forced to be quantized.

        There are three different update rules, based on the admm_type.
        See https://arxiv.org/pdf/2009.03482.pdf for more details.
        """
        if self.admm_params.admm_type == ADMMType.ADMM_Q:
            # regular projected ADMM step
            return simulated_quant(
                self.Q_quantizer,
                self.lmbda / self.admm_params.rho + Q
            )
        elif self.admm_params.admm_type == ADMMType.ADMM_R:
            # Randomized ADMM step; only update some parameters
            m = torch.rand_like(Q) < self.admm_params.admm_r_update_p
            Qp_new = simulated_quant(
                self.Q_quantizer,
                self.lmbda / self.admm_params.rho + Q
            )
            return Qp_new * m + self.Qp * (~m)
        elif self.admm_params.admm_type == ADMMType.ADMM_S:
            # Soft projection
            Qp_update = self.lmbda / self.admm_params.rho + Q
            Qp_update_proj = simulated_quant(
                self.Q_quantizer, Qp_update
            )
            Q_d = Qp_update_proj - Qp_update 
            use_quant_bitmask = torch.norm(Q_d, p="fro") < \
                self.admm_params.admm_s_beta / self.admm_params.rho
            return use_quant_bitmask * Qp_update_proj + (~use_quant_bitmask) * \
                (Qp_update + self.admm_params.admm_s_beta * Q_d / \
                    (self.admm_params.rho * torch.norm(Q_d, p="fro")))
        
    def get_X_hat(self):
        """
        Returns the estimated weight matrix.
        """
        return self.Qp + self.S + self.L @ self.R
    
    @property
    def per_iter_data(self):
        """
        Access the per-iteration approximation errors
        """
        return {
            "errors": self._errors,
            "lagrangians": self._lagrangians,
            "constraint_vals": self._constraint_vals
        }
    
    def plot_errors(
            self,
            plot_errors = True,
            plot_lagrangians = False,
            plot_constraint_vals = False
        ):
        """
        Plot the per-iteration approximation errors
        """
        plt.figure(figsize=(12, 4))

        title = ""
        separator_char = ""
        if plot_errors:
            plt.plot(
                self._errors, marker='o', linestyle='-', color='b',
                label="Error"
            )
            title = f"{title}{separator_char} Relative {self.params.error_norm.name}"
            separator_char = ","

        if plot_lagrangians:
            plt.plot(
                self._lagrangians, marker='o', linestyle='-', color='r',
                label="Lagrangian"
            )
            title = f"{title}{separator_char} Lagrangian"
            separator_char = ","

        if plot_constraint_vals:
            plt.plot(
                self._constraint_vals, marker='o', linestyle='-', color='m',
                label="Constraint"
            )
            title = f"{title}{separator_char} Constraint norm"

        title = f"{title}"
        if self.label is not None:
            title = f"{title} ({self.label})"

        plt.title(title)
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()

    def _update_best_and_log_errors(self):
        super()._update_best_and_log_errors()

        if self.params.log_errors:            
            self._constraint_vals.append(
                torch.norm(self.Q - self.Qp, p="fro").item()
            )
            lagrangian = 1/2 * torch.norm(
                    self.X - self.L @ self.R - self.Q - self.S, p="fro"
                ).item() ** 2 + torch.trace(
                    self.lmbda.T @ (self.Q - self.Qp)
                ).item() + self.admm_params.rho / 2 * torch.norm(
                    self.Qp - self.Q, p="fro"
                ).item() ** 2
            self._lagrangians.append(lagrangian)
        
    
    def iter(self):
        """
        Performs one update step for all matrices (Q, Qp, L, R, S, lambda).
        """
        self.Q = self._get_updated_Q(self.X - self.L @ self.R - self.S)
        self.Qp = self._get_updated_Qp(self.Q)
        self.L, self.R = self._get_updated_LR(self.X - self.Q - self.S)
        self.S = self._get_updated_S(self.X - self.Q - self.L @ self.R)
        self.lmbda = self.lmbda + self.admm_params.rho * (self.Q - self.Qp)

        self._update_best_and_log_errors()


class SharedQWeightDecomposition(IterativeWeightDecomposition):
    """
    Decomposes a set of weight matrices as:
        W1 approx Q + L1 R1 + S1
        W2 approx Q + L2 R2 + S2
        etc.
    via the same alternating steps as the vanilla iterative weight
    decomposition algorithm.
    """
    plot_colors = ["b", "r", "g", "c", "k", "o"]

    def __init__(
        self,
        Xs: list[torch.Tensor] = None,
        label: str = "Shared Q",
        params = IterativeWeightDecompositionParams()
    ):
        self.num_layers = len(Xs)
        self._per_layer_errors = [[] for _ in range(self.num_layers)]

        # Note: self.X will be just the first layer, and self.Xs will be a list
        # holding all of the weight matrices. self.L, self.R, and self.S will
        # also be lists holding the respective components for each layer
        self.Xs = Xs
        params.compute_quantized_component = True

        super().__init__(Xs[0], label=label, params=params)

    def _set_initial_condition_LR(self):
        # Low-rank factors
        self.L = [torch.zeros_like(Xi) for Xi in self.Xs]
        self.R = [torch.zeros_like(Xi) for Xi in self.Xs]
        if self.params.compute_low_rank_factors:
            for i in range(self.num_layers):
                # Either randomized or normal SVD
                if self.params.rand_svd:
                    U, Sigma, V = torch.svd_lowrank(
                            self.Xs[i] - self.Q, 
                            self.total_rank + self.params.rand_svd_oversampling
                        )
                    VT = V.T
                else:
                    U, Sigma, VT = torch.linalg.svd(self.Xs[i] - self.Q, full_matrices=False)
                
                sqrt_Sigma = torch.diag(torch.sqrt(Sigma[:self.total_rank]))

                self.L[i] = self._quantize_factor(U[:,:self.total_rank] @ sqrt_Sigma,
                                            left_or_right="left")
                self.R[i] = self._quantized_least_squares(
                    self.L[i], self.Xs[i] - self.Q,
                    default=sqrt_Sigma @ VT[:self.total_rank, :],
                    left_or_right="right"
                )
                
    def _set_initial_condition(self):
        """
        Set the initial condition for all matrices. Matrices that are not
        used are set to all zeros.
        """

        # Quantized matrix
        self.Q = simulated_quant(
            self.Q_quantizer, sum(self.Xs) / self.num_layers
        )

        self._set_initial_condition_LR()
            
        self.S = [torch.zeros_like(Xi) for Xi in self.Xs]

    def get_X_hat(self):
        """
        Returns the estimated weight matrix.
        """
        return [
            self.Q + self.S[i] + self.L[i] @ self.R[i] \
                for i in range(self.num_layers)
        ]

    @property
    def avg_errors(self):
        """
        Access the per-iteration average approximation errors
        """
        return self._errors
    
    @property
    def per_layer_errors(self):
        """
        Access the per-iteration approximation errors per layer
        """
        return self._per_layer_errors
    
    def plot_errors(self):
        """
        Plot the per-iteration, per-layer approximation errors
        """
        plt.figure(figsize=(12, 4))
        title = f"Relative {self.params.error_norm.name} per iteration"
        if self.label is not None:
            title = f"{title} ({self.label})"
        plt.title(title)
        for i in range(self.num_layers):
            plt.plot(
                self._per_layer_errors[i],
                marker='o', linestyle='-',
                color=self.plot_colors[i % len(self.plot_colors)],
                label=f"Layer {i}"
            )
        plt.plot(
            self._errors,
            marker='o', linestyle='-',
            color="m", label=f"Average"
        )
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        plt.show()

    def _get_updated_LR(self, residual, index):
        """
        Performs one update step for the low-rank factors and returns the
        result as the tuple (L, R).
        """
        if not self.params.compute_low_rank_factors:
            return torch.zeros_like(residual), torch.zeros_like(residual)
        
        best_LR = (self.L[index], self.R[index])
        best_err = float('inf')

        for _ in range(self.params.lplr_params.iters):
            L = self._quantized_least_squares(self.R[index], residual,
                                            default=self.L[index], 
                                            left_or_right="left")
            R = self._quantized_least_squares(self.L[index], residual,
                                            default=self.R[index],
                                            left_or_right="right")
            
            err = self.params.error_norm.error(
                X_hat=L@R, X_exact=residual
            )
            if err < best_err:
                best_err = err
                best_LR = (L, R)
        
        return best_LR
    
    def _get_current_error(self):
        X_hat = self.get_X_hat()
        per_layer_errors = [
            self.params.error_norm.error(X_hat=X_hat[i], X_exact=self.Xs[i]) \
                for i in range(self.num_layers)
        ]
        return sum(per_layer_errors) / self.num_layers
    
    def _update_best_and_log_errors(self):
        X_hat = self.get_X_hat()
        per_layer_errors = [
            self.params.error_norm.error(X_hat=X_hat[i], X_exact=self.Xs[i]) \
                for i in range(self.num_layers)
        ]
        err = sum(per_layer_errors) / self.num_layers

        if err < self.best_error:
            self.best_error = err
            self.best_components = self.get_components()

        if self.params.log_errors:
            self._errors.append(err)

            for i in range(self.num_layers):
                self._per_layer_errors[i].append(per_layer_errors[i])

    def iter(self):
        """
        Performs one update step for all matrices (Q, L, R, S).
        """
        self.Q = self._get_updated_Q(
            sum([self.Xs[i] - self.L[i] @ self.R[i] - self.S[i] \
                for i in range(self.num_layers)]) / self.num_layers
        )
        for i in range(self.num_layers):
            self.L[i], self.R[i] = self._get_updated_LR(
                self.Xs[i] - self.Q - self.S[i], i
            )
            self.S[i] = self._get_updated_S(self.Xs[i] - self.Q - self.L[i] @ self.R[i])

        self._update_best_and_log_errors()


class SharedQWeightDecompositionWithCorrelations(SharedQWeightDecomposition):
    """
    Decomposes two weight matrices as:
        W1 approx Q + L1 R1 + S1
        W2 approx Q + L2 R2 + S2,
    where Q is only nonzero for the columns (or rows) that are most correlated
    between W1 and W2. S1 and S2 contain the columns or rows that are not
    included in Q.
    """
    def __init__(
        self,
        Xs: list[torch.Tensor],
        shared_ratio: int = 0.5,
        label: str = "Shared Q cols/rows",
        params: IterativeWeightDecompositionParams = \
            IterativeWeightDecompositionParams()
    ):
        assert len(Xs) == 2, "Only implemented for 2 weight matrices"
        params.compute_sparse_component = True

        row_corrs = torch.diag((Xs[0] @ Xs[1].T) / \
                               (torch.norm(Xs[0], dim=1)*torch.norm(Xs[1], dim=1)))
        col_corrs = torch.diag((Xs[0].T @ Xs[1]) \
                               / (torch.norm(Xs[0], dim=0)*torch.norm(Xs[1], dim=0)))
        
        # We want to find the rows or columns with the highest positive
        # correlation (not absolute).
        row_corrs += (-row_corrs.min() + 1)
        col_corrs += (-col_corrs.min() + 1)

        top_row_corrs = make_sparse(row_corrs,
                                    int(np.ceil(shared_ratio*len(row_corrs)))
        )
        top_col_corrs = make_sparse(col_corrs,
                                    int(np.ceil(shared_ratio*len(col_corrs)))
        )

        self.shared_bitmask = torch.zeros_like(Xs[0])
        if sum(col_corrs) > sum(row_corrs):
            self.shared_bitmask[top_row_corrs != 0, :] = 1
        else:
            self.shared_bitmask[:, top_col_corrs != 0] = 1 

        super().__init__(Xs, label=label, params=params)

    def _set_initial_condition(self):
        """
        Set the initial condition for all matrices. Matrices that are not
        used are set to all zeros.
        """

        # Quantized matrix
        self.Q = simulated_quant(
            self.Q_quantizer,
            sum(self.Xs) / self.num_layers * self.shared_bitmask
        )

        self._set_initial_condition_LR()
            
        self.S = [torch.zeros_like(Xi) for Xi in self.Xs]

    def _get_updated_S(self, residual):
        """
        Performs one update step for the sparse component and returns the
        result.
        """
        S = residual * (1 - self.shared_bitmask)
        if self.bits_sparse < 16:
            S = simulated_quant(self.S_quantizer, S)
        return S
    
    def _get_updated_Q(self, residual):
        """
        Returns the quantized residual.
        """
        return simulated_quant(
            self.Q_quantizer,
            residual * self.shared_bitmask
        )
    
class ColumnPrunedDecomposition(IterativeWeightDecomposition):
    """
    Performs iterative weight decomposition, but replaces the quantized Q
    matrix with:
        Q_left [I Q_right] P,
    where Q_left is a quantized subset of columns from Q, I is an identity
    matrix, Q_left Q_right approximates the columns that were dropped, and
    P is a column permutation matrix.

    If m columns are pruned and X is n x d, then Q_left is n x (d-m), I is
    (d-m) x (d-m), Q_right is (d-m) x m, and P is d x d.

    Columns of Q_left are chosen as in CUR (Mahoney 2008,
    https://www.pnas.org/doi/10.1073/pnas.0803205106), and Q_right is
    computed via least squares.

    Then, Q_left and Q_right are quantized to the level params.bits_quant
    and refined via alternating least squares.

    P can be computed using only knowledge of which columns were selected
    to be part of Q_left. See ColumnPrunedDecomposition._get_Q for more
    details.
    """
    def __init__(
        self,
        X: torch.Tensor,
        pruned_cols: int = 64,
        prune_iters: int = 10,
        label: str = "LR + Pruned",
        params: IterativeWeightDecompositionParams = \
            IterativeWeightDecompositionParams()
    ):
        # Pruned cols needs to be a multiple of the quantizer block size, or
        # else quantization errors. This can be fixed on the part of the quantizer,
        # but it hasn't been yet.
        pruned_cols = (pruned_cols + params.quantizer_factory.block_size - 1) // \
            params.quantizer_factory.block_size  * params.quantizer_factory.block_size
        self.pruned_cols = pruned_cols
        self.prune_iters = prune_iters

        params.compute_quantized_component = True
        super().__init__(X, label=label, params=params)

    def _set_initial_condition(self):
        # Quantized matrix
        self.Q_left, self.Q_right = self._get_updated_Q(self.X)
        self.Q = self._get_Q(self.Q_left, self.Q_right, self.Q_cols_to_keep)

        self._set_initial_condition_LR()
        
        # Sparse matrix
        self.S = torch.zeros_like(self.X)
        
    def get_components(self):
        cols_to_keep = self.Q_cols_to_keep
        d = len(cols_to_keep)
        n_kept = d - self.pruned_cols

        perm = torch.zeros(d, d).to(self.X.device)
        perm[torch.arange(n_kept), torch.nonzero(cols_to_keep).T[0]] = 1
        perm[torch.arange(n_kept, d), torch.nonzero(~cols_to_keep).T[0]] = 1
        return {
            "L": self.L, "R": self.R, "Q_left": self.Q_left, 
            "Q_right": self.Q_right, "S": self.S,
            "P": perm
        }
    
    def _get_Q(self, Q_left, Q_right, cols_to_keep):
        """
        Returns Q = Q_left [I Q_right] P, where P is a column permutation
        matrix computed using cols_to_keep.
        """
        d = len(cols_to_keep)
        n_kept = d - self.pruned_cols

        perm = torch.zeros(d, d).to(Q_left.device)
        perm[torch.arange(n_kept), torch.nonzero(cols_to_keep).T[0]] = 1
        perm[torch.arange(n_kept, d), torch.nonzero(~cols_to_keep).T[0]] = 1

        return Q_left @ torch.hstack((
            torch.eye(n_kept).to(Q_left.device),
            Q_right
        )) @ perm
    
    def _get_updated_Q(self, residual):
        # Compute:
        #   residual =approx= Q_left [I Q_right] P,
        # where Q_left is a quantized subset of columns of the residual,
        # Q_right is computed via least squares and also quantized, and P is a
        # permutation matrix.

        d = residual.shape[1]

        # Compute the columns to keep with as in the Mahoney CUR paper
        # (https://www.pnas.org/doi/10.1073/pnas.0803205106)
        # cols_to_keep is a length-d bitmask that is 1 for columns selected
        # to be in Q_left and 0 otherwise. The columns not in Q_left are called
        # "pruned colums".
        cols_to_keep = torch.arange(d) < 0
        if self.params.rand_svd:
            svd_length = min(d - self.pruned_cols + self.params.rand_svd_oversampling, d)
            _, _, V = torch.svd_lowrank(residual, svd_length)
            VT = V.T
        else:
            _, _, VT = torch.linalg.svd(residual, full_matrices=False)
        col_idxs = torch.topk(
            torch.norm(VT.T[:, :d-self.pruned_cols], dim=1),
            d-self.pruned_cols
        ).indices
        cols_to_keep[col_idxs] = True
        self.Q_cols_to_keep = cols_to_keep

        # X_minus_cols is a matrix of the columns included in Q_left ("minus"
        # refers to removing the pruned columns).
        X_minus_cols = residual[:, cols_to_keep]
        # cols is a matrix of the pruned columns
        cols = residual[:, ~cols_to_keep]
        Q_left = simulated_quant(
            self.Q_quantizer, X_minus_cols
        )

        # Q_left W =approx= pruned columns
        W = torch.linalg.lstsq(Q_left, cols)[0]
        if torch.isnan(W).any():
            W = torch.linalg.pinv(Q_left) @ cols

        Q_right = simulated_quant(self.Q_quantizer, W)

        best_err = self.params.error_norm.error(
            X_hat=self._get_Q(Q_left, Q_right, cols_to_keep),
            X_exact=residual
        )

        # Alternating least squares to counteract the quantization error.
        best_err = self.params.error_norm.error(
            X_hat=self._get_Q(Q_left, Q_right, cols_to_keep),
            X_exact=residual
        )
        best_factors = (Q_left, Q_right)
        for _ in range(self.prune_iters):
            A = torch.hstack((torch.eye(Q_right.shape[0]).to(Q_right.device), Q_right))
            B = torch.hstack((X_minus_cols, cols))
            Q_left = simulated_quant(
                self.Q_quantizer, torch.linalg.lstsq(A.T, B.T)[0].T
            )
            Q_right = simulated_quant(
                self.Q_quantizer, torch.linalg.lstsq(Q_left, cols)[0]
            )

            err = self.params.error_norm.error(
                X_hat=self._get_Q(Q_left, Q_right, cols_to_keep),
                X_exact=residual
            )
            if err < best_err:
                best_err = err
                best_factors = (Q_left, Q_right)
        return best_factors
    
    def iter(self):
        """
        Performs one update step for all matrices (Q, L, R, S).
        """
        self.Q_left, self.Q_right = self._get_updated_Q(self.X - self.L @ self.R - self.S)
        self.Q = self._get_Q(self.Q_left, self.Q_right, self.Q_cols_to_keep)
        
        self.L, self.R = self._get_updated_LR(self.X - self.Q - self.S)
        self.S = self._get_updated_S(self.X - self.Q - self.L @ self.R)

        self._update_best_and_log_errors()
    

def alternating_mixed_lplr(   
    X: torch.Tensor = None,
    k: int = None,
    r1: int = None,
    r2: int = None,
    B1: int = 8,
    B2: int = 8,
    quantizer_factory: QuantizerFactory = QuantizerFactory(),
    normalize_and_shift=False,
    iters=10,
    use_tqdm=False,
    log_errors=False
):
    """
    Uses IterativeWeightDecomposition to perform alternating mixed LPLR. This
    function is plugged into LoftQ initialization to get LoftQ-LPLR.
    """
    weight_decomp = IterativeWeightDecomposition(
        X=X,
        params = IterativeWeightDecompositionParams(
            compute_quantized_component=False,
            compute_low_rank_factors=True,
            compute_sparse_component=False,
            lplr_params=LPLRParameters(
                split_quantization=True,
                bits_left=[16, B1],
                bits_right=[16, B2],
                rank_left=[r1, k-r1],
                rank_right=[r2, k-r1],
                iters=iters
            ),
            iters=1,
            log_errors=log_errors,
            quantizer_factory=quantizer_factory
        )
    )
    weight_decomp.run(use_tqdm=use_tqdm)
    components = weight_decomp.get_components()
    X_hat = weight_decomp.get_X_hat()
    if normalize_and_shift:
        X_hat = normalize_and_shift_wrt_inner_prod(X_hat)
    return (components["L"], components["R"]), X_hat
