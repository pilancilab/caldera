#!/bin/bash

SCRIPT_FILLED_IN=0

BASE_MODEL="meta-llama/Llama-2-7b-hf"
HESSIAN_SAVE_DIR="YOUR DIRECTORY HERE"
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/save_llama_hessians.py. \
Please go into shell_scripts/run_save_hessians.sh and replace BASE_MODEL, HESSIAN_SAVE_DIR, etc., \
and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

python scripts/save_llama_hessians.py \
    --base_model $BASE_MODEL \
    --hessian_save_path $HESSIAN_SAVE_DIR \
    --devset rp1t \
    --context_length 4096 \
    --devset_size 256 \
    --chunk_size 64 \
    --batch_size 32 \
    --devices $DEVICES