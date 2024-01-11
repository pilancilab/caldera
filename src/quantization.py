import torch
import numpy as np
from scipy import stats
from enum import Enum

import sys
sys.path.insert(0,'../peft/src')
from peft.utils.loftq_utils import NFQuantizer

class QuantType(Enum):
    UNIFORM = 0
    NF = 1

def get_quantizer(
    B: int = 8,
    quant_type: int = QuantType.UNIFORM,
    device:str = "cpu"
):
    method = "uniform"
    if quant_type == QuantType.NF:
        method = "normal"
    return NFQuantizer(
        num_bits = B,
        device=device,
        method=method
    )
    
def quantize_small_sv_components(
    X: torch.Tensor = None,
    r: int = 0,
    quantizer: NFQuantizer = None
) -> torch.Tensor:
    """
    Keep the first r columns in original dtype and quantize the last
    (X.shape[1] - r) columns.
    
    The parameter `quantization_fn` allows you to specify uniform quantization
    (via the `quantize` function) or normal float quantization (via the
    `quantize_nf` function).
    """

    assert r <= X.shape[1], "r should be less than X.shape[1]"

    if r == X.shape[1]:
        return X
    
    # Perform simulated quantization by quantizing and the dequantizing
    quantized_component = quantizer.dequantize_block(*quantizer.quantize_block(X[:, r:]))
    return torch.cat((X[:, :r], quantized_component), dim=1)

def absmax_quantize_int8(X: torch.Tensor) -> tuple[torch.Tensor, torch.float16]:
    """Quantize each float16/32 data type to int8 and return the maximum value in float16"""
    scale = X.abs().max().item() / 127.0
    int8_tensor = (X / scale).round().to(torch.int8)
    return scale, int8_tensor

def absmax_dequantize_int8(Xq: torch.Tensor, scale: torch.float16) -> torch.Tensor:
    """Dequantize int8 data type to float16/32"""
    return Xq.to(torch.float16) * scale