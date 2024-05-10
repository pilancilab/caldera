#!/bin/bash

MODELS=( "lplr-ldlq-4B-factors" )
#"lplr-ldlq-4B-factors-downdate" "lplr-ldlq-rank-128-4B-factors" "lplr-ldlq-rank-128-4B-factors-downdate" "lplr-ldlq-rank-256-4B-factors-downdate" )
# MODELS=( "lplr-ldlq-4B-factors-downdate" "lplr-ldlq-rank-128-4B-factors" "lplr-ldlq-rank-128-4B-factors-downdate" "lplr-ldlq-rank-256-4B-factors-downdate" )

for MODEL in "${MODELS[@]}"
do
    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file shell_scripts/accelerate_config.yaml \
        scripts/finetune_glue.py \
        --task_name rte \
        --model_name_or_path /media/hdd1/lplr-q-models/llama-2-7b/$MODEL/ \
        --base_model meta-llama/Llama-2-7b-hf \
        --output_dir /media/hdd1/lplr-q-finetune/llama-2-7b/glue-rte/$MODEL/ \
        --learning_rate 3e-5 \
        --num_train_epochs 15 \
        --per_device_train_batch_size 20 \
        --per_device_eval_batch_size 20 \
        --gradient_accumulation_steps 2 \
        --weight_decay 0.01 \
        --num_warmup_steps 100 \
        --lr_scheduler_type linear \
        --report_to tensorboard \
        --with_tracking \
        --checkpointing_steps epoch \
        --pad_to_max_length \
        --resume_from_checkpoint /media/hdd1/lplr-q-finetune/llama-2-7b/glue-rte/lplr-ldlq-4B-factors/epoch_9 \
        --seed 314
done

# MODELS=( "lplr-ldlq-rank-64-4B-factors-downdate" )

# for MODEL in "${MODELS[@]}"
# do
#     CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file shell_scripts/accelerate_config.yaml \
#         scripts/finetune_glue.py \
#         --task_name rte \
#         --model_name_or_path /media/hdd1/lplr-q-models/llama-3-8b/$MODEL/ \
#         --base_model meta-llama/Meta-Llama-3-8B \
#         --output_dir /media/hdd1/lplr-q-finetune/llama-3-8b/glue-rte/$MODEL/ \
#         --learning_rate 3e-5 \
#         --num_train_epochs 10 \
#         --per_device_train_batch_size 20 \
#         --per_device_eval_batch_size 20 \
#         --gradient_accumulation_steps 2 \
#         --weight_decay 0.01 \
#         --num_warmup_steps 100 \
#         --lr_scheduler_type linear \
#         --report_to tensorboard \
#         --with_tracking \
#         --checkpointing_steps epoch \
#         --pad_to_max_length \
#         --seed 314
# done