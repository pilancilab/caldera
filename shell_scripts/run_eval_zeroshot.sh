#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_PATH="PATH OF .pt FILE WITH MODEL"
DEVICE="cuda:0"
OUTPUT_FILENAME="YOUR FILE HERE"

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/eval_zeroshot.py. \
Please go into shell_scripts/run_eval_zeroshot.sh and replace BASE_MODEL, \
CALDERA_MODEL_SAVE_PATH, etc., and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

python scripts/eval_zero_shot.py \
    --model_save_path $CALDERA_MODEL_SAVE_PATH \
    --device $DEVICE \
    --tasks winogrande rte piqa arc_easy arc_challenge \
    --base_model $BASE_MODEL \
    --batch_size 8 \
    --output_path $OUTPUT_FILENAME