#!/bin/bash
DEVICES="4,5,6,7"
MODELS=( "lplr-ldlq-4B-factors-downdate" "lplr-ldlq-rank-128-4B-factors" "lplr-ldlq-rank-128-4B-factors-downdate" "lplr-ldlq-rank-256-4B-factors-downdate" )

for MODEL in "${MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file shell_scripts/accelerate_config.yaml \
        scripts/finetune_wikitext.py \
        --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/$MODEL/ \
        --output_dir /media/hdd1/lplr-q-finetune/llama-2-7b/wikitext/$MODEL/ \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --block_size 512 \
        --learning_rate 3e-6 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --weight_decay 0.001 \
        --warmup_ratio 0.02 \
        --lr_scheduler_type linear \
        --bf16 \
        --report_to tensorboard \
        --logging_steps 10 \
        --save_steps 200
done