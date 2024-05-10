# LPLR + Q

## Setup Instructions

1. Make a new [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) or conda environment with python `3.11.x` called `lplr`.

**Note**: for micromamba, make sure you're using bash. You may have some issues if you use zsh.

The included `lplr_env.yaml` file can be used for this as follows:
```
micromamba create -f lplr_env.yaml
```

2. Activate the environment: `micromamba activate lplr` or `conda activate lplr`.

3. Install the requirements. For micromamba:
```
pip install -r requirements.txt
pip install fast-hadamard-transform
```
For conda:
```
conda install pip
/anaconda/envs/lplr/bin/pip install -r requirements.txt
pip install fast-hadamard-transform
```

4. Install `lplr` as an editable submodule (named `lplr_llm`).
From the directory `lplr-q` (the home directory of this repo), run
```
pip install --editable .
```

5. Setup QuIP#:
```
./setup_quip_sharp.sh
```

6. If you have a Huggingface access token (e.g., for Llama), put that in the file `hf_access_token.txt`.

**Note**: make sure to not commit and push this file!

## Repo Structure

### `src/lplr_llm`
This folder contains the bulk of the code for LPLR-Q. Via step 4 above, everything in this folder is contained in the editable python package `lplr_llm`.

**`src/lplr_llm/utils`**: utils for LPLR-Q. Some relevant utils files are listed below:
- `enums.py`: `Enum` objects, e.g., for specifying transformer sublayers (query, key, etc.) and the name of the calibration dataset.
- `quantization.py`: our implementation of uniform and round to nearest quantizers.

**`src/lplr_llm/activation_aware`**: code for all of the activation-aware LPLR-Q decompositions (minimizing the activation aware norm $\lVert (W - Q - LR)X \rVert_F^2$, where $Q$ is quantized and $L$, $R$ are low-rank and potentially quantized).

- `dataclasses.py`: classes for storing parameters of the LPLR-Q algorithm, as well as information about quantized layers.
- `weight_compression.py`: code for the `ActivationAwareWeightCompressor` class. Unless Hessians have already been computed, this performs Hessian computation upon instantiation. The method `get_layer_quantizer`, called on a layer index, instantiates an `ActivationAwareLayerQuant` object.
- `layer_quantization.py`: code for the `ActivationAwareLayerQuant` class. The `compress_sublayer` compresses the specified sublayer.
There are also methods for plotting the data-aware error, saving errors and quantization parameters to a JSON file, and instantiating a quantized linear layer.
- `quantized_layer.py`: code for the `LPLRQuantizedLinear` class, which is a neural network module that computes $X^\top (Q + LR)^\top$ on layer input $X$, performing dequantization on the fly.

**`src/lplr_llm/data_agnostic`**: code for non-data-aware versions of LPLR-Q (ask nsagan for descriptions of this code).

**`src/lplr_llm/model`**: Llama2 model, with some minor modifications to work with LPLR-Q.

### `scripts`
This folder contains python scripts for running Frobenius norm, zero-shot, perplexity, and finetuning experiments on activation-aware LPLR-Q.

Parameters for all of these scripts are specified via command-line arguments, except for `lplr_ldlq_fro_norm_experiment.py`, where the experiment is specified directly in the code (under `if __name__ == "__main__"`).

### `shell_scripts`
These are Bash scripts for running experiments in the `scripts` folder with some reasonable parameters.

**Note**: all shell scripts are meant to be run from the root directory of this repo, i.e., `./shell_scripts/run_quip_ppl.py` instead of `cd shell_scripts && ./run_quip_ppl.py`.

### `quip_sharp`
This is the quip-sharp submodule, which pulls from the official quip-sharp repo.

### `naomi`, `rajsaha`
Miscellaneous notebooks, data, and code for testing/experiments.

## Example Experiment Workflow

**Note**: edit each script before running it to make sure that the parameters are what you want.

1. **Compute the Hessians** using `./shell_scripts/run_save_hessians.sh`, which will store Hessian matrices for each layer to files.

2. **Run Frobenius norm experiments** using `scripts/lplr_ldlq_fro_norm_experiment.py`. Experiment parameters are specified below the line `if __name__ == "__main__"`.

3. **Quantize the full model** using `shell_scripts/run_quantize_save_lplr_ldlq.sh` or `shell_scripts/run_quantize_save_quip_sharp.sh`, e.g. This stores each transformer layer, as well as embedding and model norm matrices.

4. **Run zero-shot/perplexity experiments** using `shell_scripts/run_quip_zeroshot.sh` or `shell_scripts/run_quip_ppl.sh`.

5. **Finetune** using `shell_scripts/run_finetune_wikitext.sh`

## Miscellaneous Notes
These notes might help with future development.

### List of "bugs"/misconceptions encountered in setting up finetuning
- Forgot to save the layer norm, embedding, and model norm matrices when storing the quantized model (these matrices are left in half precision, but their values need to be stored).
- In running the forward pass on the Q matrix, needed to make sure the quantization indices were on the same device as the input.
- Forgot to set `requires_grad` to `False` for the embedding and layer norm matrices when loading the quantized model.
- Forgot to wrap all tensor quantities in the quantized linear layer in `nn.Parameter` (correctly specifying `requires_grad`)

