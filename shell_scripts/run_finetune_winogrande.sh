#!/bin/bash

DEVICES="0,1,2,3"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
OUTPUT_DIR="YOUR DIRECTORY HERE"


CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file ./shell_scripts/accelerate_config.yaml \
    scripts/finetune_winogrande.py \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --learning_rate 1e-5  \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.033 \
    --num_warmup_steps 100 \
    --seed 202 \
    --max_seq_length 256 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_steps 200 \
    --bf16 \
    --ignore_rht_finetuning false \
    --with_tracking true \
    --report_to tensorboard