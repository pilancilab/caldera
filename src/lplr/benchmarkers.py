import torch
import numpy as np
from loguru import logger
from lplr.quantization import *
from lplr.weight_compressors import *
from lplr.hyperparameter_sweeps import *
from peft.utils.loftq_utils import loftq_init
from peft.utils.loftq_lplr_utils import loftq_lplr_init

class WeightCompressionBenchmarker:
    def __init__(
        self,
        weight_comp_config:WeightCompressionConfig,
        label = "Weight Compressor"
    ):
        self.weight_comp_config = weight_comp_config
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
        weight_comp_config:WeightCompressionConfig = WeightCompressionConfig(),
        run_hyper_parameter_sweep:bool = True,
        hyperparameter_sweep_params:dict = {},
        label = None
    ):
        if label is None:
            label = "Mixed Alternating LPLR"
            if weight_comp_config.algorithm_type == AlgorithmType.DIRECT_SVD_LPLR:
                label = "Direct SVD LPLR"
            if weight_comp_config.algorithm_type == AlgorithmType.LOFTQ_LPLR:
                label = "LoftQ-LPLR"

        self.run_hyper_parameter_sweep = run_hyper_parameter_sweep
        self.hyperparameter_sweep_params = hyperparameter_sweep_params

        super().__init__(weight_comp_config, label)
        
    def run(self, X:torch.Tensor, budget:int):
        lplr_kwargs = self.weight_comp_config.algorithm_kwargs.copy()

        # extra logic required w.r.t. the precision of the full quantized part
        if self.weight_comp_config.algorithm_type == AlgorithmType.LOFTQ_LPLR:
            if "num_bits" not in lplr_kwargs.keys():
                lplr_kwargs["num_bits"] = 4
            if budget < lplr_kwargs["num_bits"]*X.shape[0]*X.shape[1]:
                lplr_kwargs["num_bits"] = int(np.floor(budget / (X.shape[0]*X.shape[1])))
                logger.warning(f"Budget cannot be met with the given precision, reducing to {lplr_kwargs['num_bits']}.")


        if self.run_hyper_parameter_sweep:
            _, _, _, _, error = lplr_sweep_alpha_and_B(
                X=X, budget=budget,
                weight_comp_config=WeightCompressionConfig(
                    algorithm_type=self.weight_comp_config.algorithm_type,
                    algorithm_kwargs=lplr_kwargs
                ),
                **self.hyperparameter_sweep_params
            )
        else:
            if self.weight_comp_config.algorithm_type == AlgorithmType.ALTERNATING_MIXED_LPLR:
                lplr_kwargs["X"] = X
                _, X_hat = alternating_mixed_lplr(**lplr_kwargs)
            elif self.weight_comp_config.algorithm_type == AlgorithmType.DIRECT_SVD_LPLR:
                lplr_kwargs["X"] = X
                _, X_hat = direct_svd_mixed_lplr(**lplr_kwargs)
            else: ## Loftq-LPLR
                lplr_kwargs["weight"] = X
                Q, R, L = loftq_lplr_init(**lplr_kwargs)
                X_hat = Q + L @ R
                del Q, L ,R
            
            error = torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item()
            del X_hat
        torch.cuda.empty_cache()
        self.errors.append(error)

class FullQuantBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        quantizer_factory:QuantizerFactory = QuantizerFactory(),
        label = "Full Quantization"
    ):
        self.quantizer_factory = quantizer_factory
        super().__init__(WeightCompressionConfig(), label)

    def run(self, X:torch.Tensor, budget:int):
        n, d = X.size()

        b = 0
        for b_option in [8, 4, 2]:
            if b_option*n*d <= budget:
                b = b_option
                break
        
        assert b > 1, "For full quantization, we need at least two bits of precision."

        quantizer = self.quantizer_factory.get_quantizer(b, device=X.device)
        X_hat = quantizer.dequantize_block(*quantizer.quantize_block(X))
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())

class LoftqBenchmarker(WeightCompressionBenchmarker):
    def __init__(
        self,
        weight_comp_config:WeightCompressionConfig = \
            WeightCompressionConfig(algorithm_type=AlgorithmType.LOFTQ),
        fixed_rank:bool = False,
        label = "LoftQ"
    ):
        self.fixed_rank = fixed_rank
        super().__init__(weight_comp_config, label)

    def run(self, X:torch.Tensor, budget:int):
        kwargs = self.weight_comp_config.algorithm_kwargs.copy()
        n, d = X.size()
        b = kwargs["num_bits"] if "num_bits" in kwargs.keys() else 8
        if budget < b*n*d:
            b = 0
            for b_option in [8, 4, 2]:
                if b_option*n*d <= budget:
                    b = b_option
                    break
            logger.warning(f"Budget cannot be met with the given precision, reducing to {b}.")

        kwargs["num_bits"] = b

        if not self.fixed_rank:
            r = int(np.floor((budget - n*d*b) / (16*(n + d))))
            kwargs["reduced_rank"] = r
        kwargs["weight"] = X

        Q, R, L = loftq_init(**kwargs)
        X_hat = Q + L @ R
        self.errors.append(torch.norm(X - X_hat, p="fro").item() / torch.norm(X, p="fro").item())

        del X_hat, Q, L, R
        torch.cuda.empty_cache()