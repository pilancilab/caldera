#!/bin/bash

python scripts/quip_llama_ppl.py \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/lplr-ldlq-16B-factors-hessian-downdate \
    --finetune_save_dir /media/hdd1/lplr-q-finetune/llama-2-7b/wikitext/lplr-ldlq-16B-factors-hessian-downdate-2 \
    --output_path data/ppl-temp/llama-2-7b/lplr-ldlq-16B-factors-downdate-2 \
    --device cuda:7 \
    --base_model meta-llama/Llama-2-7b-hf \
    --seed 0 \
    --seqlen 4096 \
    --datasets wikitext2

for i in {200..2000..200}
do
    echo $i
    python scripts/quip_llama_ppl.py \
    --model_save_path /media/hdd1/lplr-q-models/llama-2-7b/lplr-ldlq-16B-factors-hessian-downdate \
    --finetune_save_dir /media/hdd1/lplr-q-finetune/llama-2-7b/wikitext/lplr-ldlq-16B-factors-hessian-downdate-2/checkpoint-$i\
    --output_path data/ppl-temp/llama-2-7b/lplr-ldlq-16B-factors-downdate-2-$i \
    --device cuda:7 \
    --base_model meta-llama/Llama-2-7b-hf \
    --seed 0 \
    --seqlen 4096 \
    --datasets wikitext2
done


# python scripts/quip_llama_ppl.py \
#     --model_save_path /media/hdd1/lplr-q-models/llama-3-8b/lplr-ldlq-rank-256-4B-factors-downdate \
#     --output_path data/ppl-temp/llama-3-8b/temp-test-256 \
#     --device cuda:7 \
#     --base_model meta-llama/Meta-Llama-3-8B \
#     --seed 0 \
#     --seqlen 8192