#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_PATH="PATH OF .pt FILE WITH MODEL"
OUTPUT_DIR="FINETUNING OUTPUT DIRECTORY"
BLOCK_SIZE=256

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/finetune_wikitext.py. \
Please go into shell_scripts/run_finetune_wikitext.sh and replace BASE_MODEL, \
CALDERA_MODEL_SAVE_PATH, etc., and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

accelerate launch --config_file shell_scripts/accelerate_config.yaml \
    scripts/finetune_wikitext.py \
    --model_save_path $CALDERA_MODEL_SAVE_PATH \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size $BLOCK_SIZE \
    --learning_rate 3e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.001 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type linear \
    --bf16 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --prediction_loss_only
