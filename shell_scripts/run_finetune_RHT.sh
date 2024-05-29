#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"

python scripts/finetune_RHT.py \
    --base_model $BASE_MODEL \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --devset_size 256 \
    --ctx_size 512 \
    --device cuda:0 \
    --ft_bs 2 \
    --ft_valid_size 64 \
    --RHT_learning_rate 1e-3 \
    --epochs 5