#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_PATH="PATH OF .pt FILE WITH MODEL"
OUTPUT_DIR="FINETUNING OUTPUT DIRECTORY"
GLUE_TASK="rte"

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/finetune_glue.py. \
Please go into shell_scripts/run_finetune_glue.sh and replace BASE_MODEL, \
CALDERA_MODEL_SAVE_PATH, etc., and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

accelerate launch --config_file shell_scripts/accelerate_config.yaml \
    scripts/finetune_glue.py \
    --task_name $GLUE_TASK \
    --model_name_or_path $CALDERA_MODEL_SAVE_PATH \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --learning_rate 3e-5 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0.01 \
    --num_warmup_steps 100 \
    --lr_scheduler_type linear \
    --report_to tensorboard \
    --with_tracking \
    --checkpointing_steps epoch \
    --pad_to_max_length \
    --seed 314