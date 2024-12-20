import sys
from caldera.decomposition.dataclasses import *
from caldera.decomposition.weight_compression import ActivationAwareWeightCompressor
from caldera.utils.enums import TransformerSubLayers
import gc
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoModelForSequenceClassification,
    LlamaForCausalLM
)
import torch
import os
import torch.multiprocessing as mp
import glog
from lib.utils import graph_wrapper
import shutil

SUBLAYER_TO_STRING = {
    TransformerSubLayers.KEY: "Key Projection (attn)",
    TransformerSubLayers.QUERY: "Query Projection (attn)",
    TransformerSubLayers.VALUE: "Value Projection (attn)",
    TransformerSubLayers.O: "O Projection (attn)",
    TransformerSubLayers.GATE: "Gate Projection (mlp)",
    TransformerSubLayers.UP: "Up Projection (mlp)",
    TransformerSubLayers.DOWN: "Down Projection (mlp)",
}


@dataclass
class Arguments:
    hessian_save_path: str = field(
        metadata={"help": "Path in which the Hessians were stored"}
    )
    model_save_path: str = field(
        metadata={"help": ("Path in which to save the quantized model, e.g., artifacts/model.pt")}
    )
    base_model: str = field(
        metadata={
            "help": (
                "Path of the model that is being quantized, as "
                "either a local or a Huggingface path"
            )
        }
    )
    devices: list[str] = field(
        metadata={
            "help": (
                "List of devices to use for quantization, e.g. "
                '"cuda:0 cuda:1 cuda:2 cuda:3"'
            )
        }
    )
    ft_rank: int = field(
        default=64,
        metadata={
            "help": (
                "Number of columns of L and rows of R, in the decomposition"
                "W approx. Q + LR to finetune. The remaining columns will "
                "remain fixed."
            )
        },
    )
    token: str = field(
        default="", metadata={"help": "Huggingface token for private models."}
    )
    start_layer: int = field(
        default=0,
        metadata={
            "help": "Layer index to start quantizing from (to resume quantization from an interrupt)"
        },
    )
    stop_layer: int = field(
        default=int(sys.maxsize),
        metadata={
            "help": "Layer index to stop quantizing at (to resume quantization from an interrupt)"
        },
    )


def quant_layer(
    in_q,
    model_save_path,
    base_model,
    config,
    ft_rank,
    grad_ckpt,
    device,
    data_params,
    quant_params,
    hessian_save_path,
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", low_cpu_mem_usage=True
    ).cpu()

    while True:
        layer_idx = in_q.get()

        if layer_idx is None:
            return

        weight_compressor = ActivationAwareWeightCompressor(
            model_params=ModelParameters(base_model),
            data_params=data_params,
            hessian_save_path=hessian_save_path,
            quant_params=quant_params,
            compute_hessians=False,
        )
        layer_quant = weight_compressor.get_layer_quantizer(layer_idx, device)

        with torch.no_grad():
            layer = model.model.layers[layer_idx]

            for sublayer in layer_quant.sublayer_info.keys():
                print(f"Quantizing layer {layer_idx}, {SUBLAYER_TO_STRING[sublayer]}")
                layer_quant.compress_sublayer(sublayer)

                attr_names = layer_quant.sublayer_info[sublayer].out_key.split(".")
                setattr(
                    getattr(layer, attr_names[0]),
                    attr_names[1],
                    layer_quant.get_quantized_linear_layer(
                        sublayer, ft_rank, grad_ckpt
                    ),
                )
                layer_quant.clean_up_sublayer(sublayer)
            layer = layer.cpu()
            torch.save(layer, f"{model_save_path}/layers/quant_layer_{layer_idx}.pt")
            del layer_quant
            gc.collect()
            torch.cuda.empty_cache()


def quantize_save_llama(
    base_model: str,
    hessian_save_path: str,
    model_save_path: str,
    token: str = "",
    ft_rank: int = 64,
    grad_ckpt: bool = True,
    data_params: DataParameters = DataParameters(),
    quant_params: CalderaParams = CalderaParams(),
    quant_devices=["cuda"],
    start_layer=0,
    stop_layer=int(sys.maxsize),
):
    os.makedirs(f"{model_save_path}/layers", exist_ok=True)
    mp.set_start_method("spawn")

    if token:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", low_cpu_mem_usage=True, token=token
        ).cpu()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype="auto", low_cpu_mem_usage=True
        ).cpu()

    model_config = model.config
    n_layers = len(model.model.layers)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    manager = mp.get_context("spawn").Manager()
    in_q = manager.Queue()
    quant_procs = []

    for device in quant_devices:
        p = mp.Process(
            target=quant_layer,
            args=(
                in_q,
                model_save_path,
                base_model,
                model_config,
                ft_rank,
                grad_ckpt,
                device,
                data_params,
                quant_params,
                hessian_save_path,
            ),
        )
        p.start()
        quant_procs.append(p)

    stop_layer: int = min(stop_layer, n_layers)
    for layer_idx in range(start_layer, stop_layer):
        in_q.put(layer_idx)

    for _ in quant_devices:
        in_q.put(None)

    for p in quant_procs:
        p.join()

    # now save the full model
    model = load_layers_cpu(model_save_path, base_model)
    shutil.rmtree(f'{model_save_path}/') 
    torch.save(model, f"{model_save_path}")

def load_layers_cpu(
    model_save_path,
    base_model,
):
    model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype='auto', device_map="cpu", low_cpu_mem_usage=True
    ).cpu()
    
    for layer_idx in range(len(model.model.layers)):
        layer = torch.load(
            f"{model_save_path}/layers/quant_layer_{layer_idx}.pt",
            map_location="cpu"
        )
        layer.post_attention_layernorm.weight.requires_grad = False
        layer.input_layernorm.weight.requires_grad = False

        for sublayer in [
            layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj,
            layer.self_attn.o_proj, layer.mlp.gate_proj, layer.mlp.up_proj,
            layer.mlp.down_proj
        ]:
            if sublayer.ft_rank > 0:
                sublayer.L_ft = torch.nn.Parameter(sublayer.L_ft.contiguous(), requires_grad=True)
                sublayer.R_ft = torch.nn.Parameter(sublayer.R_ft.contiguous(), requires_grad=True)

        model.model.layers[layer_idx] = layer

    return model

def load_quantized_model(
    model_save_path,
    base_model,
    device,
    sequence_classification=False,
    seq_class_num_labels=2,
    cuda_graph=False,
):
    model = torch.load(model_save_path, map_location=device).to(device)
    if cuda_graph:
        graph_model = graph_wrapper.get_graph_wrapper(AutoModelForCausalLM, device="cpu").from_pretrained(
            base_model, torch_dtype='auto', device_map="cpu", low_cpu_mem_usage=True,
            use_flash_attention_2=True
        ).to("cpu")
        for i in range(len(graph_model.model.layers)):
            graph_model.model.layers[i].self_attn.q_proj = model.model.layers[i].self_attn.q_proj
            graph_model.model.layers[i].self_attn.k_proj = model.model.layers[i].self_attn.k_proj
            graph_model.model.layers[i].self_attn.v_proj = model.model.layers[i].self_attn.v_proj
            graph_model.model.layers[i].self_attn.o_proj = model.model.layers[i].self_attn.o_proj
            graph_model.model.layers[i].mlp = model.model.layers[i].mlp
            graph_model.model.layers[i].post_attention_layernorm = graph_model.model.layers[i].post_attention_layernorm.to(device)
            graph_model.model.layers[i].input_layernorm = graph_model.model.layers[i].input_layernorm.to(device)
        graph_model.model.norm = graph_model.model.norm.to(device)
        graph_model.model.embed_tokens = graph_model.model.embed_tokens.to(device)
        graph_model.lm_head = graph_model.lm_head.to(device)
        graph_model.graph_device = device
        model = graph_model.to(device )
    
    elif sequence_classification:
        seq_model = AutoModelForSequenceClassification.from_pretrained(
            base_model, torch_dtype='auto', device_map="cpu", low_cpu_mem_usage=True, num_labels=seq_class_num_labels
        ).cpu()
        seq_model.score = seq_model.score.to(device)
        seq_model.score.weight.requires_grad = True
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        seq_model.model.layers = model.model.layers
        model = seq_model.to(device)

    if not sequence_classification:
        model.lm_head.weight.requires_grad = False
    model.model.embed_tokens.weight.requires_grad = False
    model.model.norm.weight.requires_grad = False
    for layer in model.model.layers:
        layer.post_attention_layernorm.weight.requires_grad = False
        layer.input_layernorm.weight.requires_grad = False

    return model


if __name__ == "__main__":
    glog.setLevel("WARN")

    parser = HfArgumentParser([Arguments, CalderaParams, QuIPArgs])

    args, quant_params, quip_args = parser.parse_args_into_dataclasses()
    quant_params.quip_args = quip_args
    quantize_save_llama(
        base_model=args.base_model,
        hessian_save_path=args.hessian_save_path,
        model_save_path=args.model_save_path,
        token=args.token,
        ft_rank=args.ft_rank,
        grad_ckpt=False,
        data_params=DataParameters(),
        quant_params=quant_params,
        quant_devices=args.devices,
        start_layer=args.start_layer,
        stop_layer=args.stop_layer,
    )
