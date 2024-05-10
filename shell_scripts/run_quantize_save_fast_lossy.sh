#!/bin/bash
RANK=128

QUANT_PARAMS="--Q_bits 4 \
    --compute_low_rank_factors true \
    --compute_quantized_component true \
    --L_bits 4 \
    --R_bits 4 \
    --lattice_quant_LR true \
    --rank $RANK \
    --activation_aware_LR true \
    --activation_aware_Q false \
    --hadamard_transform true \
    --iters 2 \
    --lplr_iters 5 \
    --rand_svd true \
    --update_order LR Q
    --ft_rank 32"

python scripts/quantize_save_llama.py \
    --hessian_save_path /media/hdd1/lplr-q-hessians/llama-2-7b \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/temp-test \
    --devices cuda:4 cuda:5 cuda:6 cuda:7 \
    --base_model meta-llama/Llama-2-7b-hf \
    $QUANT_PARAMS \