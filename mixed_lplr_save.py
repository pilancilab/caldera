from phantominator import shepp_logan
from matplotlib import pyplot as plt

import torch
import numpy as np
import math

from scipy import stats

from loguru import logger

from tqdm import tqdm

from lplr_utils import normalize_and_shift_wrt_inner_prod



def loftq(
    X: torch.Tensor = None,
    r: int = None,
    B: int = 4,
    quantization_fn=quantize_nf,
    normalize_and_shift=False,
    iters=10,
    log_errors=False
):
     assert (
        X.shape[0] >= X.shape[1]
    ), "Input matrix X should satisfy X.shape[0] >= X.shape[1]"

    


def test_alternating_mixed_lplr(
    lplr_params=[{
        "alpha": 0.5, # Fraction of retained columns to be in full precision
        "beta": 0.4, # Fraction of columns to be retained
        "B1": 8,
        "B2": 8,
        "quantization_fn": quantize
    }],
    iters=100
):

    plot_colors = ["b", "r", "g", "c", "m", "k"]
    plot_markers = ["o", "X", "*"]
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Generate a phantominator matrix
    X = torch.Tensor(shepp_logan(2048))
    print(f"Phantom shape: {X.shape[1]}")

    plt.figure(figsize=(11, 3))

    for i, param_set in enumerate(lplr_params):
        kwargs = param_set.copy()
        kwargs["k"] = int(kwargs["beta"] * X.shape[1])
        kwargs["r1"] = int(kwargs["alpha"] * kwargs["k"])
        kwargs["r2"] = kwargs["r1"]
        kwargs["normalize_and_shift"] = True
        kwargs["log_errors"] = True
        kwargs["iters"] = iters
        del kwargs["alpha"]
        del kwargs["beta"]

        # Call alternating_mixed_lplr and retrieve errors
        kwargs["X"] = torch.Tensor(X)
        _, _, _, errors = alternating_mixed_lplr(**kwargs)

        fro_norm_X = torch.norm(X, p="fro").item()
        relative_errors = [error / fro_norm_X for error in errors]

        # Plot errors over iterations
        
        plt.plot(
            range(1, iters + 1),
            relative_errors,
            marker=plot_markers[(i // len(plot_colors)) % len(plot_markers)],
            linestyle="-",
            markersize=4,
            color=plot_colors[i % len(plot_colors)],
            label=f"Param Set {i+1}*")

    print("-"*80, "\n* Legend Key")
    for i, kwargs in enumerate(lplr_params):
        print(F"Param Set {i+1}: ", kwargs)

    plt.title("Frobenius Norm Errors over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


test_alternating_mixed_lplr([
    {
        "alpha": 0.5, # Fraction of retained columns to be in full precision
        "beta": 0.4, # Fraction of columns to be retained
        "B1": 8,
        "B2": 8,
        "quantization_fn": quantize
    },
    {
        "alpha": 0.5, # Fraction of retained columns to be in full precision
        "beta": 0.4, # Fraction of columns to be retained
        "B1": 8,
        "B2": 8,
        "quantization_fn": quantize_nf
    }
])


def lplr_sweep_alpha(
    X:torch.Tensor = None,
    budget: int = 0,
    alpha_start:float = 0,
    alpha_stop:float = 0.5,
    alpha_step:float = 0.1,
    B1:int = 8,
    B2:int = 8,
    quantization_fn = quantize,
    iters=50
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
        k = int(X.shape[1] * beta)
        r = int(k*alpha)

        logger.info(f"alpha={alpha}, beta={beta}")
        if k == 0:
            logger.warning(f"The bit budget of {budget} cannot be met for alpha={alpha}. Stopping early")
            break

        L, R, X_hat = alternating_mixed_lplr(
            X=X, k=k, r1=r, r2=r, B1=B1, B2=B2,
            quantization_fn=quantization_fn,
            normalize_and_shift=True,
            iters=iters
        )

        fro_err = torch.norm(X - X_hat, p="fro") / torch.norm(X, p="fro")
        logger.info(f"Frobenius norm error: {fro_err}")
        if fro_err < best_fro_err:
            best_fro_err = fro_err
            best_alpha = alpha
            best_beta = beta
            best_L_R = (L, R)

        logger.info(f"The best frobenius norm error was for alpha={best_alpha}: {best_fro_err}")
    return best_L_R, best_alpha, best_beta, best_fro_err
        

class WeightCompressionBenchmarker:
    def __init__(
        self,
        params:dict,
        label = "Weight Compressor"
    ):
        self.params = params
        self._errors = []
        self._label = label

    def run(self, X:torch.Tensor, budget:int):
        raise NotImplementedError()

    @property
    def errors(self):
        return self._errors

    @property
    def label(self):
        return self._label
    

class LplrBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        params:dict,
        label = "Mixed Alternating LPLR"
    ):
        super().__init__(params, label)
        
    def run(self, X:torch.Tensor, budget:int):
        kwargs = self.params.copy()
        kwargs["X"] = X
        kwargs["budget"] = budget

        _, _, _, error = lplr_sweep_alpha(**kwargs)
        self.errors.append(error)

class FullQuantBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        params:dict,
        label = "Full Quantization"
    ):
        super().__init__(params, label)

    def run(self, X:torch.Tensor, budget:int):
        n, d = X.size()
        b = int(np.floor(budget / (n*d)))
        assert b > 1, "For full quantization, we need at least two bits of precision."

        X_hat = self.params["quantization_function"]
        self.errors.append(torch.norm(X - X_hat, p="fro") / torch.norm(X, p="fro"))


def compare_against_benchmarks(
    X_list:list[torch.Tensor] = None,
    average_bit_level:int = 3,
    benchmarkers=[
        LplrBenchmarker(
            {
                "B1": 8,
                "B2": 8,
                "quantization_fn": quantize,
                "iters": 50,
                "alpha_start": 0,
                "alpha_stop": 0.5,
                "alpha_step": 0.1
            },
            label="Mixed Alternating LPLR(B=8)"
        ),
        FullQuantBenchmarker(
            {
                "quantization_fn": quantize,
            },
            label="Uniform Quantization"
        ),
        FullQuantBenchmarker(
            {
                "quantization_fn": quantize_nf,
            },
            label="Normal Float Quantization"
        )
    ]
):
    for layer in X_list:
        n, d = X.size()
        budget = n*d*average_bit_level

        for benchmarker in benchmarkers:
            benchmarker.run(X, budget)
            
    for benchmarker in benchmarkers:
        print(f"{benchmarker.label}: {benchmarker.errors}")
        

compare_against_benchmarks([X])