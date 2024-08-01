#!/bin/bash

set -euxo pipefail

RANK=256
FT_RANK=0
LORA_BITS=4
CALIBRATION_SIZE=6144

MODEL_NAME="llama-2-13b"
HF_MODEL="meta-llama/${MODEL_NAME^}-hf"

BASE_PATH="/media/hdd1/caldera"
CALDERA_MODEL_SAVE_DIR="$BASE_PATH/quantized/$MODEL_NAME/${RANK}R-${LORA_BITS}B-${CALIBRATION_SIZE}H"

# HESSIAN_SAVE_DIR=$(huggingface-cli download relaxml/${MODEL_NAME^}-${CALIBRATION_SIZE})
HESSIAN_SAVE_DIR="/media/hdd1/huggingface/hub/models--relaxml--Hessians-Llama-2-13b-6144/snapshots/1eb1423bd94e2fb664a80715efcda44ccde44e3e"

TOKEN="$HF_TOKEN"

QUANT_PARAMS="--Q_bits 2 \
    --compute_low_rank_factors true \
    --compute_quantized_component true \
    --L_bits $LORA_BITS \
    --R_bits $LORA_BITS \
    --lattice_quant_LR true \
    --rank $RANK \
    --activation_aware_LR true \
    --activation_aware_Q true \
    --hadamard_transform true \
    --iters 15 \
    --lplr_iters 10 \
    --rand_svd false \
    --update_order LR Q \
    --Q_hessian_downdate true \
    --ft_rank $FT_RANK"

python scripts/quantize_save_llama.py \
    --hessian_save_path $HESSIAN_SAVE_DIR \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --base_model $HF_MODEL \
    --token $TOKEN \
    $QUANT_PARAMS