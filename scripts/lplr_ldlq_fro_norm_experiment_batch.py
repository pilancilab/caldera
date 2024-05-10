import os
from lplr_llm.activation_aware.layer_quantization import *
from lplr_llm.activation_aware.weight_compression import *
import glog

def get_filename(layer, sublayer, experiment):
    sublayer = str(sublayer)
    if "." in sublayer:
        sublayer = sublayer.split(".")[1]
    return f"errors_layer_{layer}_{sublayer}_proj_{experiment}"

def fro_norm_experiment(
    layer: int,
    sublayer: int,
    weight_compressors: dict[str, ActivationAwareWeightCompressor],
    json_save_dir: str,
    plot_save_dir: str
):
    for experiment in weight_compressors:
        print(f"Quantizing layer {layer}, sublayer {sublayer} using {experiment}")
        layer_quant = weight_compressors[experiment].get_layer_quantizer(
            layer_idx=layer, label=experiment)
        layer_quant.compress_sublayer(sublayer)

        # Plot errors
        filename = get_filename(layer, sublayer, experiment)
        # layer_quant.plot_errors(sublayer, plot_first_iter=False,
        #                         savefile=f"{plot_save_dir}/{filename}.png")
        print(f"The minimum error is {layer_quant.min_error(sublayer)}")

        # Save to JSON
        layer_quant.export_errors_json(
            sublayer, f"{json_save_dir}/{filename}"
        )

if __name__ == "__main__":
    # START EXPERIMENT PARAMETERS ##########

    # Model to quantize, as a Huggingface (or local) path
    BASE_MODEL = "meta-llama/Llama-2-7b-hf"

    # Huggingface token, if the model is private
    with open("hf_access_token.txt") as f:
        HF_TOKEN = f.read().strip()
    if not HF_TOKEN or HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        HF_TOKEN = None

    # Directory in which the Hessians were saved
    HESSIAN_SAVE_PATH = "data/hessians/llama-2-7b"

    # Directory in which to store Frobenius norm errors in JSON format
    JSON_SAVE_DIR = "data/frobenius_errors/llama-2-7b"

    # Directory in which to store plot images
    PLOT_SAVE_DIR = "data/plots/llama-2-7b"

    # Device to use for quantization
    DEVICE = "cuda:4"

    # Number of iterations of 
    QLR_ITERS = 50
    LPLR_ITERS = 10

    glog.setLevel("WARN")
    os.makedirs(JSON_SAVE_DIR, exist_ok=True)
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

    def get_weight_compressor(quant_params):
        return ActivationAwareWeightCompressor(
            model_params=ModelParameters(
                base_model=BASE_MODEL,
                token=HF_TOKEN
            ),
            data_params=DataParameters(),
            hessian_save_path=HESSIAN_SAVE_PATH,
            quant_params=quant_params,
            compute_hessians=False,
            quant_device=DEVICE,
        )

    # for RANK in [64, 128, 256]:
    for RANK in [256]:
        # ADD TO THIS DICTIONARY TO ADD MORE EXPERIMENTS
        weight_compressors = {
            f"quip_sharp_rank_{RANK}": get_weight_compressor(
                ActivationAwareQuantParams(
                    Q_bits=2,
                    compute_low_rank_factors=False,
                    compute_quantized_component=True,
                    iters=2,
                    quip_args=QuIPArgs(
                        lora_rank=RANK
                    ),
                    full_quip_sharp=True
                )
            ),
            f"lplr_ldlq_16B_factors_rank_{RANK}": get_weight_compressor(
                ActivationAwareQuantParams(
                    Q_bits=2,
                    L_bits=16, R_bits=16,
                    lattice_quant_LR=False,
                    quant_factory_LR=QuantizerFactory(method="normal"),
                    rank=RANK,
                    activation_aware_Q=True,
                    activation_aware_LR=True,
                    hadamard_transform=True,
                    iters=QLR_ITERS,
                    lplr_iters=LPLR_ITERS,
                    rand_svd=False,
                    update_order=["LR", "Q"]
                )
            ),
            f"lplr_ldlq_16B_factors_hessian_downdate_rank_{RANK}": get_weight_compressor(
                ActivationAwareQuantParams(
                    Q_bits=2,
                    L_bits=16, R_bits=16,
                    lattice_quant_LR=False,
                    quant_factory_LR=QuantizerFactory(method="normal"),
                    rank=RANK,
                    activation_aware_Q=True,
                    activation_aware_LR=True,
                    hadamard_transform=True,
                    iters=4,
                    lplr_iters=LPLR_ITERS,
                    rand_svd=False,
                    Q_hessian_downdate=True,
                    update_order=["LR", "Q"]
                )
            ),
            f"lplr_lattice_quant_16B_factors_rank_{RANK}": get_weight_compressor(
                ActivationAwareQuantParams(
                    Q_bits=2,
                    L_bits=16, R_bits=16,
                    lattice_quant_LR=False,
                    quant_factory_LR=QuantizerFactory(method="normal"),
                    rank=RANK,
                    activation_aware_Q=False,
                    activation_aware_LR=True,
                    hadamard_transform=True,
                    iters=QLR_ITERS,
                    lplr_iters=LPLR_ITERS,
                    rand_svd=False,
                    update_order=["LR", "Q"]
                )
            ),
            f"lplr_ldlq_4B_factors_rank_{RANK}": get_weight_compressor(
                ActivationAwareQuantParams(
                    Q_bits=2,
                    L_bits=4, R_bits=4,
                    lattice_quant_LR=False,
                    quant_factory_LR=QuantizerFactory(method="normal"),
                    rank=RANK,
                    activation_aware_Q=True,
                    activation_aware_LR=True,
                    hadamard_transform=True,
                    iters=QLR_ITERS,
                    lplr_iters=LPLR_ITERS,
                    rand_svd=False,
                    update_order=["LR", "Q"]
                )
            )
        }

        for LAYER in [31]:
            for SUBLAYER in [
                    TransformerSubLayers.QUERY, TransformerSubLayers.KEY,
                    TransformerSubLayers.VALUE, TransformerSubLayers.O,
                    TransformerSubLayers.GATE, TransformerSubLayers.UP,
                    TransformerSubLayers.DOWN]:
            # for SUBLAYER in [
            #         TransformerSubLayers.VALUE, TransformerSubLayers.O,
            #         TransformerSubLayers.GATE, TransformerSubLayers.UP,
            #         TransformerSubLayers.DOWN]:

                fro_norm_experiment(
                    layer=LAYER, sublayer=SUBLAYER,
                    weight_compressors=weight_compressors,
                    json_save_dir=JSON_SAVE_DIR,
                    plot_save_dir=PLOT_SAVE_DIR
                )