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
        lplr_type,
        params:dict = {},
        label = None
    ):
        if label is None:
            label = "Mixed Alternating LPLR"
            if lplr_type == LplrType.DIRECT_SVD:
                label = "Direct SVD LPLR"
            if lplr_type == LplrType.WITH_Q:
                label = "LoftQ with LPLR"

        self.lplr_type = lplr_type
        super().__init__(params, label)
        
    def run(self, X:torch.Tensor, budget:int):
        lplr_kwargs = self.params.copy()
        sweep_kwargs = {}
        param_sweep_arguments = [
            "alpha_start", "alpha_stop", "alpha_step",
            "prune", "B_options"]
        
        for arg in param_sweep_arguments:
            if arg in lplr_kwargs.keys():
                sweep_kwargs[arg] = lplr_kwargs[arg]
                del lplr_kwargs[arg]

        # extra logic required w.r.t. the precision of the full quantized part
        if self.lplr_type == LplrType.WITH_Q:
            if "BQ" not in lplr_kwargs.keys():
                lplr_kwargs["BQ"] = 2
            if budget < lplr_kwargs["BQ"]*X.shape[0]*X.shape[1]:
                lplr_kwargs["BQ"] = int(np.floor(budget / (X.shape[0]*X.shape[1])))
                logger.warning(f"Budget cannot be met with the given precision, reducing b to {lplr_kwargs['BQ']}.")

        _, _, _, _, error = lplr_sweep_alpha_and_B(
            X=X, budget=budget, kwarg_dict=lplr_kwargs,
            lplr_type=self.lplr_type, **sweep_kwargs
        )
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
            logger.warning(f"Budget cannot be met with the given precision, reducing b to {b}.")

        kwargs["B"] = b
        r = int(np.floor((budget - n*d*b) / (16*(n + d))))
        kwargs["r"] = r
        kwargs["X"] = X
        kwargs["normalize_and_shift"] = True
        kwargs["log_errors"] = False

        _, _, X_hat = loftq(**kwargs)
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())