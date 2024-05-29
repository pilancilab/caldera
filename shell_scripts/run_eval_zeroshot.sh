#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
OUTPUT_FILENAME="YOUR FILE HERE"

python scripts/eval_zero_shot.py \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --device cuda:0 \
    --tasks winogrande rte piqa arc_easy arc_challenge \
    --base_model $BASE_MODEL \
    --batch_size 8 \
    --output_path $OUTPUT_FILENAME