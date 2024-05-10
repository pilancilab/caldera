#!/bin/bash

RANK=256

QUANT_PARAMS="--Q_bits 2 \
    --compute_low_rank_factors true \
    --compute_quantized_component true \
    --L_bits 4 \
    --R_bits 4 \
    --lattice_quant_LR true \
    --rank $RANK \
    --activation_aware_LR true \
    --activation_aware_Q true \
    --hadamard_transform true \
    --iters 15 \
    --lplr_iters 10 \
    --rand_svd false \
    --Q_hessian_downdate true \
    --update_order LR Q"

python scripts/quantize_save_llama.py \
    --hessian_save_path /media/hdd1/lplr-q-hessians/llama-3-8b \
    --model_save_path /media/hdd1/lplr-q-models/llama-3-8b/lplr-ldlq-rank-256-4B-factors-downdate \
    --devices cuda:4 cuda:5 cuda:6 cuda:7 \
    --base_model meta-llama/Meta-Llama-3-8B \
    $QUANT_PARAMS \