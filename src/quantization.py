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
    OUR_UNIFORM = 2
    OUR_UNIFORM_CLIPPED = 3
    OUR_NF = 4

class OurQuantizer:
    def __init__(self, num_bits=2, method="normal", block_size=64):
        self.num_bits = num_bits
        self.method = method
        self.block_size = block_size

        if self.method != "normal" and self.method != "uniform" and self.method != "uniform_clipped":
            raise NotImplementedError("Other quantization methods not supported yet.")

    def _quantize_uniform(self, weight_divabs):
        # weight_divabs is between -1 and 1, inclusive
        weight_scaled = weight_divabs * (2**(self.num_bits - 1) - 1)
        return weight_scaled.round().to(torch.int8)

    def _quantize_nf(self, weight_divabs):
        Normal = torch.distributions.Normal(0, 1)
        # We quantize the range [-1, 0) and the range [0, 1] separately, with
        # each having 2^{b-1} levels.
        #
        # The quantization levels are found as follows: take 2^{b-1} evenly-spaced
        # points from [delta, 1/2] and 2{b-1} + 1 from [1/2, 1-delta], where delta
        # is as defined below. The quantization levels are the corresponding
        # quantiles of a standard normal distribution, scaled such that they lie
        # in the range [-1, 1].
        M = 2**(self.num_bits-1)
        delta = 1/2 * (1/30 + 1/32) # as described above
        res_neg = (1/2 - delta) / (M - 1) # resolution for [delta, 1/2]
        res_pos = (1/2 - delta) / M # resolution for [1/2, 1-delta]
                                                # levels to be in [-1, 1]
        
        # We index into q_neg and q_pos with these indices to get the quantized
        # values for the negative and positive parts of A, respectively.
        q_neg = Normal.icdf(res_neg * torch.arange(M).to(weight_divabs.device) + delta) / stats.norm.ppf(1-delta)
        q_pos = Normal.icdf(res_pos * torch.arange(M + 1).to(weight_divabs.device) + 1/2) / stats.norm.ppf(1-delta)

        neg_quantiles = (weight_divabs < 0) * \
            ((Normal.cdf(weight_divabs * stats.norm.ppf(1-delta)) - delta) / res_neg)
        neg_quantiles_round_down = neg_quantiles.floor().long()
        neg_quantiles_round_up = neg_quantiles.ceil().long()
        mask = (torch.abs(weight_divabs - q_neg[neg_quantiles_round_down]) <= torch.abs(weight_divabs - q_neg[neg_quantiles_round_up]))
        neg_quant_idxs = neg_quantiles_round_down * mask + neg_quantiles_round_up * (~mask)

        pos_quantiles = (weight_divabs >= 0) * \
            ((Normal.cdf(weight_divabs * stats.norm.ppf(1-delta)) - 1/2) / res_pos)
        pos_quantiles_round_down = pos_quantiles.floor().long()
        pos_quantiles_round_up = torch.minimum(pos_quantiles.ceil().long(), torch.tensor(M))
        mask = (torch.abs(weight_divabs - q_pos[pos_quantiles_round_down]) <= torch.abs(weight_divabs - q_pos[pos_quantiles_round_up]))
        pos_quant_idxs = pos_quantiles_round_down * mask + pos_quantiles_round_up * (~mask)

        idxs = neg_quant_idxs + (weight_divabs >= 0) * (pos_quant_idxs + M - 1)
        return idxs.to(torch.uint8)

    def _dequantize_uniform(self, weight_quant):
        return weight_quant.float() / (2**(self.num_bits - 1) - 1)
    
    def _dequantize_nf(self, weight_quant):
        Normal = torch.distributions.Normal(0, 1)
        M = 2**(self.num_bits-1)
        delta = 1/2 * (1/30 + 1/32) # as described above
        res_neg = (1/2 - delta) / (M - 1) # resolution for [delta, 1/2]
        res_pos = (1/2 - delta) / M # resolution for [1/2, 1-delta]
                                                # levels to be in [-1, 1]
        # quantization levels for the negative and positive halves, respectively
        q_neg = Normal.icdf(res_neg * torch.arange(M - 1).to(weight_quant.device) + delta) / stats.norm.ppf(1-delta)
        q_pos = Normal.icdf(res_pos * torch.arange(M + 1).to(weight_quant.device) + 1/2) / stats.norm.ppf(1-delta)
        q_levels = torch.cat((q_neg, q_pos))
        print(q_levels)
        return q_levels[weight_quant.long()]
        
    def quantize_block(self, weight):
        if len(weight.shape) != 2:
            raise ValueError(f"Only support 2D matrix, but your input has {len(weight.shape)} dimensions.")
        if weight.shape[0] * weight.shape[1] % self.block_size != 0:
            raise ValueError(
                f"Weight with shape ({weight.shape[0]} x {weight.shape[1]}) "
                f"is not dividable by block size {self.block_size}."
            )
        
        weight_reshape = weight.flatten().reshape(-1, self.block_size) # (L, M*N/B)
        weight_max = weight_reshape.abs().max(dim=-1)[0].unsqueeze(-1)
        if self.method == "uniform_clipped":
            weight_max = weight_reshape.mean(dim=-1) + 2.5 * weight_reshape.std(dim=-1)
            weight_reshape = torch.minimum(weight_reshape, weight_max)
            weight_reshape = torch.maximum(weight_reshape, -weight_max)

        weight_divabs = weight_reshape / weight_max
        if self.method == "normal":
            weight_quant = self._quantize_nf(weight_divabs)
        else:
            weight_quant = self._quantize_uniform(weight_divabs)
        return weight_quant, weight_max, weight.shape
    
    def dequantize_block(self, weight_quant, weight_max, weight_shape):
        if self.method == "normal":
            weight = self._dequantize_nf(weight_quant)
        else:
            weight = self._dequantize_uniform(weight_quant)
        
        return (weight * weight_max).reshape(weight_shape)


def get_quantizer(
    B: int = 8,
    quant_type: int = QuantType.UNIFORM,
    device:str = "cpu"
):
    if quant_type == QuantType.UNIFORM:
        return NFQuantizer(
            num_bits = B,
            device=device,
            method="uniform"
        )
    if quant_type == QuantType.NF:
        return NFQuantizer(
            num_bits = B,
            device=device,
            method="normal"
        )
    if quant_type == QuantType.OUR_UNIFORM:
        return OurQuantizer(
            num_bits = B,
            method = "uniform"
        )
    
    if quant_type == QuantType.OUR_UNIFORM_CLIPPED:
        return OurQuantizer(
            num_bits=B,
            method="uniform_clipped"
        )
    return OurQuantizer(
        num_bits = B,
        method = "normal"
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