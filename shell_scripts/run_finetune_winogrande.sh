#!/bin/bash

DEVICES="4,5,6,7"

MODELS=( "lplr-ldlq-rank-64-16B-factors-downdate" "lplr-ldlq-rank-64-4B-factors-downdate" "lplr-ldlq-rank-128-4B-factors-downdate" "lplr-ldlq-rank-256-4B-factors-downdate" )
for MODEL in "${MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file ./shell_scripts/accelerate_config.yaml \
        scripts/finetune_winogrande_seq_class.py \
        --model_name_or_path /media/hdd1/lplr-q-models/llama-3-8b/$MODEL \
        --output_dir /media/hdd1/lplr-q-finetune/llama-3-8b/winogrande-seq-class/$MODEL \
        --base_model meta-llama/Meta-Llama-3-8B \
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
done