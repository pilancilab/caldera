#!/bin/bash

SCRIPT_FILLED_IN=0

# CALDERA PARAMETERS
RANK=256
FT_RANK=64
LR_BITS=4
CALDERA_ITERS=15
LPLR_ITERS=10
# END CALDERA PARAMETERS

BASE_MODEL="meta-llama/Llama-2-7b-hf"
DEVICES="cuda:0 cuda:1 cuda:2 cuda:3"

CALDERA_MODEL_SAVE_PATH="PATH OF .pt FILE TO SAVE QUANTIZED MODEL"
HESSIAN_SAVE_DIR="DIRECTORY WITH HESSIANS"

if [ $SCRIPT_FILLED_IN -eq 0 ]; then
    echo -e "This script is meant as a template for running scripts/quantize_save_llama.py. \
Please go into shell_scripts/run_quantize_save_caldera.sh and replace the CALDERA parameters, etc., \
and then set SCRIPT_FILLED_IN=1 at the top of the file."
  exit -0
fi

QUANT_PARAMS="--Q_bits 2 \
    --compute_low_rank_factors true \
    --compute_quantized_component true \
    --L_bits $LR_BITS \
    --R_bits $LR_BITS \
    --lattice_quant_LR true \
    --rank $RANK \
    --activation_aware_LR true \
    --activation_aware_Q true \
    --hadamard_transform true \
    --iters $CALDERA_ITERS \
    --lplr_iters $LPLR_ITERS \
    --rand_svd true \
    --update_order LR Q \
    --Q_hessian_downdate false \
    --ft_rank $FT_RANK"

python scripts/quantize_save_llama.py \
    --hessian_save_path $HESSIAN_SAVE_DIR \
    --model_save_path $CALDERA_MODEL_SAVE_PATH \
    --devices $DEVICES \
    --base_model $BASE_MODEL \
    $QUANT_PARAMS