# CALDERA (Calibration Aware Low-Precision DEcomposition with Low-Rank Adaptation)

CALDERA is a post-training compression method that represents the weights of LLM matrices via a low-rank, low-precision decomposition $\mathbf{W} \approx \mathbf{Q} + \mathbf{L} \mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are low-rank factors and $\mathbf{Q}, \mathbf{L}$ and $\mathbf{R}$ are all quantized to low-precision formats.
By formulating this decomposition as an optimization problem and solving it via alternating minimization, CALDERA outperforms existing compression techniques in the regime of less than 2.5 bits per parameter. 
To enhance performance on specific tasks, CALDERA also supports Low Rank Adaptation (LoRA) fine tuning ([Hu et al, 2021](https://arxiv.org/pdf/2106.09685)) of a portion of the low-rank factors.

🔗 Paper link: [Compressing Large Language Models using Low Rank
and Low Precision Decomposition](https://openreview.net/pdf?id=lkx3OpcqSZ)

## Setup Instructions

### Note on Python, CUDA, and PyTorch Versions
These setup instructions have been tested on Python 3.10 and 3.11, CUDA 12.1 and 12.2, and PyTorch 2.2.

In particular, the package `fast-hadamard-transform` lacks wheels for newer versions of these dependencies; the available wheels can be found [here](https://github.com/Dao-AILab/fast-hadamard-transform) (the wheel filenames are of the form `fast_hadamard_transform-1.0.4.post1+cu<CUDA_VERSION>torch<PYTORCH_VERSION>xx11abiTRUE-cp<PYTHON_VERSION>-cp<PYTHON_VERSION>-linux_x86_64.whl`).

### 🛠 Instructions
1. Install `caldera` as a submodule (named `caldera`).
From the home directory of this repository, run
```
pip install .
```
This will automatically install all dependencies, except `fast-hadamard-transform`, which has dependency issues.

2. While CALDERA can be used with any quantizer, we demonstrate the results using QuIP#'s quantizer. Setup the QuIP# ([Tseng et al, 2024](https://arxiv.org/pdf/2402.04396)) submodule:
```
./setup_quip_sharp.sh
```

This script first sets up the QuIP# Python library, and then builds the `quiptools` CUDA library, which provides dequantization kernels for inference.

QuIP# is used for the quantization of the $\mathbf{Q}$ matrix (backbone), and also provides useful subroutines for Hessian computation.

3. Install `fast-hadamard-transform`: `pip install fast-hadamard-transform`.

**Note**: If you get the error `package 'wheel' is not installed`, you can install it using `pip install wheel`.


## Repo structure

### `src/caldera`
This folder contains the bulk of the code for CALDERA. Via step 1 above, everything in this folder is contained in the editable python package `caldera`.

**`src/caldera/utils`**: utils for CALDERA. Some relevant utils files are listed below:
- `enums.py`: `Enum` objects, e.g., for specifying transformer sublayers (query, key, etc.) and the name of the calibration dataset.
- `quantization.py`: Uniform and Normal Float ([Dettmers et al, 2023](https://arxiv.org/pdf/2305.14314)) quantizers.
Generally, these are not recommended; E8 Lattice quantizers from QuIP# typically perform better.

**`src/caldera/decomposition`**: code for the CALDERA decomposition algorithm, as well as its application to transformer layers.

- `dataclasses.py`: classes for storing parameters of the CALDERA algorithm, as well as information about quantized layers.
- `weight_compression.py`: code for the `ActivationAwareWeightCompressor` class. Unless Hessians have already been computed, this performs Hessian computation upon instantiation. The method `get_layer_quantizer`, called on a layer index, instantiates an `ActivationAwareLayerQuant` object.
- `layer_quantization.py`: code for the `ActivationAwareLayerQuant` class. The `compress_sublayer` compresses the specified sublayer, calling the `caldera` method from `alg.py`.
There are also methods for plotting the data-aware error, saving errors and quantization parameters to a JSON file, and instantiating a quantized linear layer.
- `alg.py`: the CALDERA algorithm.
- `quantized_layer.py`: code for the `CalderaQuantizedLinear` class, which is a neural network module that computes $X^\top (Q + LR)^\top$ on layer input $X$, performing dequantization on the fly.


### `scripts`
This folder contains python scripts for running zero-shot, perplexity, and finetuning experiments.

Parameters for all of these scripts are specified via command-line arguments.

### `shell_scripts`
These are Bash scripts for running experiments in the `scripts` folder with some reasonable parameters.

Each shell script has variables at the top specifying, e.g., directories in which to save script outputs.
Make sure you set those variables as appropriate.

**Note**: all shell scripts are meant to be run from the root directory of this repo, i.e., `./shell_scripts/run_eval_ppl.py` instead of `cd shell_scripts && ./run_eval_ppl.py`.

### `quip_sharp`
This is the quip-sharp submodule, which is initialized in step 2 of the setup instructions.

### `notebooks`

- `test_caldera.ipynb` to obtain the quickly try out CALDERA decomposition on a random matrix. 
- `eval_throughput.ipynb` obtains the autoregressive generation throughput of the model.

## 🚀 Trying out CALDERA: Example experiment workflow

**Note**: Edit each script before running it to make sure desired parameters are used.

1. **Compute the Hessians** using `./shell_scripts/run_save_hessians.sh`, which will store Hessian matrices for each layer to files.

2. **Quantize the full model** using `shell_scripts/run_quantize_save_caldera.sh`. This stores each quantized transformer layer.
The quantized model can later be loaded in using the `load_quantized_model` function in `scripts/quantize_save_llama.py`.

3. **Run zero-shot/perplexity experiments** using `shell_scripts/run_eval_zeroshot.sh` or `shell_scripts/run_eval_ppl.sh`.

4. **Finetune** using, e.g., `shell_scripts/run_finetune_wikitext.sh`

## Citation
If you find our work useful, consider citing it as:
```bibtex
@inproceedings{
    saha2024compressing,
    title={Compressing Large Language Models using Low Rank and Low Precision Decomposition},
    author={Rajarshi Saha and Naomi Sagan and Varun Srivastava and Andrea Goldsmith and Mert Pilanci},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=lkx3OpcqSZ}
}
```
