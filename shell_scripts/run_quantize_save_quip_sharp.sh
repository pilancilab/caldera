#!/bin/bash
RANK=128

QUANT_PARAMS="--Q_bits 2 \
    --compute_low_rank_factors false \
    --compute_quantized_component true \
    --iters 1"

QUIP_PARAMS="--lora_rank $RANK"

python scripts/quantize_save_llama.py \
    --hessian_save_path data/hessians/llama-2-7b \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/quip-sharp-128 \
    --devices cuda:4 cuda:5 cuda:6 cuda:7 \
    --base_model meta-llama/Llama-2-7b-hf \
    --full_quip_sharp true \
    $QUANT_PARAMS \
    $QUIP_PARAMS