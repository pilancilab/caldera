#!/bin/bash

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
OUTPUT_FILENAME="YOUR FILE HERE"
FINETUNE_CHECKPOINT_DIR="YOUR DIRECTORY HERE"

python scripts/eval_ppl.py \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --output_path  $OUTPUT_FILENAME \
    --device cuda:0 \
    --base_model $BASE_MODEL \
    --seed 0 \
    --seqlen 4096 \
    --datasets wikitext2 c4 \
    --finetune_save_dir $FINETUNE_CHECKPOINT_DIR