#!/bin/bash
RANK=64

TOKEN=$(cat hf_access_token.txt | xargs)
echo "Using HF token $TOKEN"

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
    --update_order LR Q
    --Q_hessian_downdate true
    --ft_rank 64"

python scripts/quantize_save_llama.py \
    --hessian_save_path /media/hdd1/lplr-q-hessians/llama-2-70b \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-70b/lplr-ldlq-4B-factors-downdate \
    --devices cuda:4 \
    --base_model meta-llama/Llama-2-70b-hf \
    --token $TOKEN \
    $QUANT_PARAMS \