#!/bin/bash

python scripts/quip_llama_zero_shot.py \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/lplr-ldlq-rank-256-4B-factors-downdate \
    --device cuda:6 \
    --tasks winogrande rte piqa arc_easy arc_challenge \
    --base_model meta-llama/Llama-2-7b-hf \
    --batch_size 8 \
    --output_path data/zeroshot/llama-2-7b/lplr-ldlq-rank-256-4B-factors-downdate

    # --finetine_save_dir /media/hdd1/lplr-q-finetune/llama-2-7b/quip-sharp/checkpoint-870 \
