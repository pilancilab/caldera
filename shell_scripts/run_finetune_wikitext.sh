#!/bin/bash
DEVICES="0,1,2,3"
CALDERA_MODEL_SAVE_DIR="YOUR DIRECTORY HERE"
BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="YOUR DIRECTORY HERE"


CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file shell_scripts/accelerate_config.yaml \
    scripts/finetune_wikitext.py \
    --model_save_path $CALDERA_MODEL_SAVE_DIR \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 256 \
    --learning_rate 3e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.001 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type linear \
    --bf16 \
    --ignore_rht_finetuning false \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --prediction_loss_only
