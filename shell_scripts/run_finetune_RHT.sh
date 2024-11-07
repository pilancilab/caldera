#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_PATH_NO_RHT_FT="PATH OF .pt FILE WITH MODEL TO FINETUNED"
CALDERA_MODEL_SAVE_PATH_WITH_RHT_FT="PATH OF .pt FILE TO SAVE FINETUNED MODEL"

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/finetune_RHT.py. \
Please go into shell_scripts/run_finetune_glue.sh and replace BASE_MODEL, \
CALDERA_MODEL_SAVE_PATH, etc., and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

python scripts/finetune_RHT.py \
    --base_model $BASE_MODEL \
    --model_path $CALDERA_MODEL_SAVE_PATH_NO_RHT_FT \
    --finetuned_save_path $CALDERA_MODEL_SAVE_PATH_WITH_RHT_FT \
    --devset_size 256 \
    --ctx_size 512 \
    --device cuda:0 \
    --ft_bs 2 \
    --ft_valid_size 64 \
    --RHT_learning_rate 1e-3 \
    --epochs 1