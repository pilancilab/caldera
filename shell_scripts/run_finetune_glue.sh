#!/bin/bash

DEVICES="0,1,2,3"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
OUTPUT_DIR="YOUR DIRECTORY HERE"
GLUE_TASK="rte"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file shell_scripts/accelerate_config.yaml \
    scripts/finetune_glue.py \
    --task_name $GLUE_TASK \
    --model_name_or_path $CALDERA_MODEL_SAVE_DIR \
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