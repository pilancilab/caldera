#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
CALDERA_MODEL_SAVE_PATH="PATH OF .pt FILE WITH MODEL"
OUTPUT_DIR="FINETUNING OUTPUT DIRECTORY"
SEQ_LEN=256

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/finetune_winogrande.py. \
Please go into shell_scripts/run_finetune_winogrande.sh and replace BASE_MODEL, \
CALDERA_MODEL_SAVE_PATH, etc., and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi


CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --config_file ./shell_scripts/accelerate_config.yaml \
    scripts/finetune_winogrande.py \
    --model_name_or_path $CALDERA_MODEL_SAVE_PATH \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --learning_rate 1e-5  \
    --weight_decay 0.01 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.033 \
    --num_warmup_steps 100 \
    --seed 202 \
    --max_seq_length $SEQ_LEN \
    --num_train_epochs 1 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_steps 200 \
    --bf16 \
    --with_tracking true \
    --report_to tensorboard