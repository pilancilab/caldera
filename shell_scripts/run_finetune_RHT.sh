#!/bin/bash

python scripts/finetune_RHT.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --model_save_path "/media/hdd1/lplr-q-models/llama-2-7b/lplr-ldlq-16B-factors" \
    --devset_size 256 \
    --ctx_size 512 \
    --device cuda:4 \
    --ft_bs 2 \
    --ft_valid_size 64 \
    --RHT_lr 1e-3 \
    --epochs 5