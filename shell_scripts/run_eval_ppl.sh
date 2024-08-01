#!/bin/bash

HF_MODEL="meta-llama/Llama-2-13b-hf"
MODEL_PATH="llama-2-13b/256R-4B-6144H"
CALDERA_MODEL_SAVE_DIR="/media/hdd1/caldera/quantized/$MODEL_PATH"
OUTPUT_FILENAME="/media/hdd1/caldera/$MODEL_PATH/ppl-eval.json"
FINETUNE_CHECKPOINT_DIR="$CALDERA_MODEL_SAVE_DIR"

CUDA_VISIBLE_DEVICES="0" python scripts/eval_ppl.py \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --output_path  $OUTPUT_FILENAME \
    --base_model $HF_MODEL \
    --seed 0 \
    --seqlen 4096 \
    --datasets wikitext2 c4 \
    --finetune_save_dir $FINETUNE_CHECKPOINT_DIR \
    --device cuda:0

# --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
