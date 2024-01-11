import sys
sys.path.insert(0,'../peft/src')
from weight_compressors import alternating_mixed_lplr
from quantization import QuantType, get_quantizer
from peft.utils.loftq_utils import _low_rank_decomposition, NFQuantizer
from peft.import_utils import is_bnb_4bit_available

import torch
from tqdm import tqdm

import logging
from typing import Union

@torch.no_grad()
def loftq_lplr_init(
    weight: Union[torch.Tensor, torch.nn.Parameter],
    num_bits: int,
    num_bits_factors: int,
    reduced_rank: int,
    num_full_rank_factors: int = 0,
    quant_type: int = QuantType.NF,
    num_iter=1,
    num_iter_lplr=10,
    log_errors=False 
):
    if num_bits not in [2, 4, 8]:
        raise ValueError("Only support 2, 4, 8 bits quantization")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")
    
    n, d = weight.size()
    transposed_weight = False
    if n < d:
        weight = weight.T
        transposed_weight = True
        n, d = d, n

    device = weight.device
    dtype = weight.dtype

    logging.info(
        f"Weight: ({n}, {d}) | Rank: {reduced_rank} "
        f"| Num Iter: {num_iter} | Num Bits: {num_bits}"
    )
    
    quantizer = get_quantizer(
        B=num_bits, quant_type=quant_type, device=device
    )

    weight = weight.to(device=device, dtype=torch.float32)
    res = weight

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in tqdm(range(num_iter)):
        torch.cuda.empty_cache()
        # Quantization
        dequantized_weight = quantizer.dequantize_block(*quantizer.quantize_block(res))
        res = weight - dequantized_weight

        # Decompose the residual by SVD
        mtxs, _ = alternating_mixed_lplr(
            X=res, k=reduced_rank,
            r1=num_full_rank_factors, r2=num_full_rank_factors,
            B1=num_bits_factors, B2=num_bits_factors,
            quant_type=quant_type, iters=num_iter_lplr
        )
        L, R = mtxs
        res = weight - torch.mm(L, R)

        error = torch.norm(weight - dequantized_weight - L@R, p="fro").item()
        errors.append(error)
        if error < best_error:
            best_mtxs = (dequantized_weight, R, L)
            best_error = error

    dequantized_weight, R, L = best_mtxs
    if transposed_weight:
        dequantized_weight = dequantized_weight.T
        L, R = R.T, L.T

    lora_A, lora_B = R, L

    if log_errors:
        return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B, errors

    return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B

@torch.no_grad()
def loftq_init(
    weight: Union[torch.Tensor, torch.nn.Parameter],
    num_bits: int,
    reduced_rank: int,
    num_iter=1,
    quant_type=QuantType.NF,
    log_errors=False 
):
    if num_bits not in [2, 4, 8]:
        raise ValueError("Only support 2, 4, 8 bits quantization")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")

    out_feature, in_feature = weight.size()
    device = weight.device
    dtype = weight.dtype

    logging.info(
        f"Weight: ({out_feature}, {in_feature}) | Rank: {reduced_rank} "
        f"| Num Iter: {num_iter} | Num Bits: {num_bits}"
    )

    quantizer = get_quantizer(
        B=num_bits, quant_type=quant_type, device=device
    )
    compute_device = device

    weight = weight.to(device=compute_device, dtype=torch.float32)
    res = weight

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in tqdm(range(num_iter)):
        torch.cuda.empty_cache()
        # Quantization
        quantized_weight, max_abs, shape = quantizer.quantize_block(res)
        dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)

        res = weight - dequantized_weight

        # Decompose the residual by SVD
        output = _low_rank_decomposition(res, reduced_rank=reduced_rank)
        L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
        res = weight - torch.mm(L, R)

        error = torch.norm(weight - dequantized_weight - L@R, p="fro").item()
        errors.append(error)
        if error < best_error:
            best_mtxs = (dequantized_weight, R, L)
            best_error = error

    dequantized_weight, R, L = best_mtxs
    lora_A, lora_B = R, L

    if log_errors:
        return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B, errors

    return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B
