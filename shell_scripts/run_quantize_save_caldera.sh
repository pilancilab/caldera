#!/bin/bash
RANK=64
BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
HESSIAN_SAVE_DIR="YOUR DIRECTORY HERE"
TOKEN="YOUR HUGGINGFACE TOKEN HERE"

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
    --rand_svd true \
    --update_order LR Q \
    --Q_hessian_downdate true \
    --ft_rank 64"

python scripts/quantize_save_llama.py \
    --hessian_save_path $HESSIAN_SAVE_DIR \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --base_model $BASE_MODEL \
    --token $TOKEN \
    $QUANT_PARAMS \