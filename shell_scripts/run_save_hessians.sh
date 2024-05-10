#!/bin/bash
TOKEN=$(cat hf_access_token.txt | xargs)
echo "Using HF token $TOKEN"
python scripts/save_llama_hessians.py \
    --base_model meta-llama/Llama-2-70b-hf \
    --token $TOKEN \
    --hessian_save_path data/hessians/llama-2-70b \
    --devset rp1t \
    --context_length 4096 \
    --devset_size 256 \
    --chunk_size 64 \
    --batch_size 32 \
    --devices cuda:4 cuda:5