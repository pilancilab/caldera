#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
HESSIAN_SAVE_DIR="YOUR DIRECTORY HERE"
TOKEN="YOUR HUGGINGFACE TOKEN HERE"

python scripts/save_llama_hessians.py \
    --base_model $BASE_MODEL \
    --token $TOKEN \
    --hessian_save_path $HESSIAN_SAVE_DIR \
    --devset rp1t \
    --context_length 4096 \
    --devset_size 256 \
    --chunk_size 64 \
    --batch_size 32 \
    --devices cuda:0 cuda:1 cuda:2 cuda:3