import torch
import numpy as np
from loguru import logger
from quantization import *
from weight_compressors import *
from hyperparameter_sweeps import *

import sys
sys.path.insert(0,'../peft/src')
from peft.utils.loftq_lplr_utils import loftq_lplr_init
from peft.utils.loftq_utils import loftq_init

class WeightCompressionBenchmarker:
    def __init__(
        self,
        algorithm_params:dict,
        label = "Weight Compressor"
    ):
        self.algorithm_params = algorithm_params
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
        lplr_type: int = AlgorithmType.ALTERNATING_MIXED_LPLR,
        algorithm_params:dict = {},
        run_hyper_parameter_sweep:bool = True,
        hyperparameter_sweep_params:dict = {},
        label = None
    ):
        if label is None:
            label = "Mixed Alternating LPLR"
            if lplr_type == AlgorithmType.DIRECT_SVD_LPLR:
                label = "Direct SVD LPLR"
            if lplr_type == AlgorithmType.LOFTQ_LPLR:
                label = "LoftQ-LPLR"

        self.lplr_type = lplr_type
        self.run_hyper_parameter_sweep = run_hyper_parameter_sweep
        self.hyperparameter_sweep_params = hyperparameter_sweep_params

        super().__init__(algorithm_params, label)
        
    def run(self, X:torch.Tensor, budget:int):
        lplr_kwargs = self.algorithm_params.copy()

        # extra logic required w.r.t. the precision of the full quantized part
        if self.lplr_type == AlgorithmType.LOFTQ_LPLR:
            if "num_bits" not in lplr_kwargs.keys():
                lplr_kwargs["num_bits"] = 4
            if budget < lplr_kwargs["num_bits"]*X.shape[0]*X.shape[1]:
                lplr_kwargs["num_bits"] = int(np.floor(budget / (X.shape[0]*X.shape[1])))
                logger.warning(f"Budget cannot be met with the given precision, reducing to {lplr_kwargs['num_bits']}.")

        if self.run_hyper_parameter_sweep:
            _, _, _, _, error = lplr_sweep_alpha_and_B(
                X=X, budget=budget, kwarg_dict=lplr_kwargs,
                lplr_type=self.lplr_type,
                **self.hyperparameter_sweep_params
            )
        else:
            if self.lplr_type == AlgorithmType.ALTERNATING_MIXED_LPLR:
                lplr_kwargs["X"] = X
                _, X_hat = alternating_mixed_lplr(**lplr_kwargs)
            elif self.lplr_type == AlgorithmType.DIRECT_SVD_LPLR:
                lplr_kwargs["X"] = X
                _, X_hat = direct_svd_mixed_lplr(**lplr_kwargs)
            else: ## Loftq-LPLR
                lplr_kwargs["weight"] = X
                _, X_hat = loftq_lplr_init(**lplr_kwargs)
            error = torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item()
        self.errors.append(error)

class FullQuantBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        quant_type = QuantType.UNIFORM,
        label = "Full Quantization"
    ):
        self.quant_type = quant_type
        super().__init__({}, label)

    def run(self, X:torch.Tensor, budget:int):
        n, d = X.size()

        b = 0
        for b_option in [8, 4, 2]:
            if b_option*n*d <= budget:
                b = b_option
                break
        
        assert b > 1, "For full quantization, we need at least two bits of precision."

        quantizer = get_quantizer(
            B=b, quant_type=self.quant_type, device=X.device
        )
        X_hat = quantizer.dequantize_block(*quantizer.quantize_block(X))
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())

class LoftqBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        algorithm_params:dict,
        label = "LoftQ"
    ):
        super().__init__(algorithm_params, label)

    def run(self, X:torch.Tensor, budget:int):
        kwargs = self.params.copy()
        n, d = X.size()
        b = kwargs["num_bits"] if "num_bits" in kwargs.keys() else 8
        if budget < b*n*d:
            b = int(np.floor(budget / (n*d)))
            logger.warning(f"Budget cannot be met with the given precision, reducing to {b}.")

        kwargs["num_bits"] = b
        r = int(np.floor((budget - n*d*b) / (16*(n + d))))
        kwargs["reduced_rank"] = r
        kwargs["weight"] = X

        Q, R, L = loftq_init(**kwargs)
        X_hat = Q + L @ R
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())