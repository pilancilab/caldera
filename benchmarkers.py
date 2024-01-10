import torch
import numpy as np
from loguru import logger
from quantization import *
from weight_compressors import *
from hyperparameter_sweeps import *

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
    
    def reset_errors(self):
        self._errors = []

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

        _, _, _, _, error = lplr_sweep_alpha_and_B(**kwargs)
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

        quant_fn = self.params["quantization_fn"] if "quantization_fn" in self.params else quantize
        X_hat = normalize_and_shift_wrt_inner_prod(X, quant_fn(X, b))
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())

class LoftqBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        params:dict,
        label = "LoftQ"
    ):
        super().__init__(params, label)

    def run(self, X:torch.Tensor, budget:int):
        kwargs = self.params.copy()
        n, d = X.size()
        b = kwargs["B"] if "B" in kwargs.keys() else 4
        if budget < b*n*d:
            b = int(np.floor(budget / (n*d)))
            logger.warning("Budget cannot be met with the given precision, reducing b to {b}.")

        kwargs["B"] = b
        r = int(np.floor((budget - n*d*b) / (16*(n + d))))
        kwargs["r"] = r
        kwargs["X"] = X
        kwargs["normalize_and_shift"] = True
        kwargs["log_errors"] = False

        _, _, _, X_hat = loftq(**kwargs)
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())

class DirectSvdBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        params:dict,
        label = "Direct SVD"
    ):
        super().__init__(params, label)

    def run(self, X:torch.Tensor, budget:int):
        kwargs = self.params.copy()
        kwargs["X"] = X
        kwargs["budget"] = budget
        kwargs["run_alternating_optimization"] = False

        _, _, _, _, error = lplr_sweep_alpha_and_B(**kwargs)
        self.errors.append(error)