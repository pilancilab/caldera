#!/bin/bash
DEVICES="0,1,2,3"
# MODELS=( "lplr-ldlq-rank-64-16B-factors-downdate" "lplr-ldlq-rank-64-4B-factors-downdate" "lplr-ldlq-rank-128-4B-factors-downdate" "lplr-ldlq-rank-256-4B-factors-downdate" )
MODELS=( "lplr-ldlq-rank-256-4B-factors-downdate" "lplr-ldlq-rank-64-16B-factors-downdate" )

for MODEL in "${MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file shell_scripts/accelerate_config_2.yaml \
        scripts/finetune_wikitext.py \
        --model_save_path /media/hdd1/lplr-q-models/llama-3-8b/$MODEL/ \
        --output_dir /media/hdd1/lplr-q-finetune/llama-3-8b/wikitext/$MODEL-RHT-FT-2/ \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --base_model meta-llama/Meta-Llama-3-8B \
        --block_size 1024 \
        --learning_rate 1e-6 \
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