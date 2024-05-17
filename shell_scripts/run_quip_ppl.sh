#!/bin/bash

i=200
MODEL="lplr-ldlq-rank-256-4B-factors-downdate"

for i in {100..900..100}
do
python scripts/quip_llama_ppl.py \
    --model_save_path /media/hdd1/lplr-q-models/llama-3-8b/$MODEL \
    --output_path data/ppl-temp/llama-3-8b/rht_ft_test_rank_256\
    --device cuda:2 \
    --base_model meta-llama/Meta-Llama-3-8B \
    --seed 0 \
    --seqlen 8192 \
    --datasets wikitext2 \
    --finetune_save_dir /media/hdd1/lplr-q-finetune/llama-3-8b/wikitext/$MODEL-RHT-FT-2/checkpoint-$i
done
