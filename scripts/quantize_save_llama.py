from lplr_llm.activation_aware.dataclasses import *
from lplr_llm.utils.enums import TransformerSubLayers
from lplr_llm.activation_aware.weight_compression import \
    ActivationAwareWeightCompressor
import gc
from transformers import AutoModelForCausalLM, HfArgumentParser, AutoModelForSequenceClassification
import torch
import os
import torch.multiprocessing as mp
import glog
from safetensors.torch import load_model


@dataclass
class Arguments:
    hessian_save_path: str = field(metadata={
        "help": "Path in which the Hessians were stored"
    })
    model_save_path: str = field(metadata={
        "help": ("Path in which to save the quantized model.")
    })
    devices: list[str] = field(metadata={
        "help": ("List of devices to use for quantization, e.g. "
                 "\"cuda:0 cuda:1 cuda:2 cuda:3\"")
    })
    ft_rank: int = field(default=64, metadata={
        "help": ("Number of columns of L and rows of R, in the decomposition"
                 "W approx. Q + LR to finetune. The remaining columns will "
                 "remain fixed.")
    })
    base_model: str = field(default="meta-llama/Llama-2-7b-hf", metadata={
        "help": ("Path of the model that is being quantized, as "
                 "either a local or a Huggingface path")
    })
    token: str = field(default="", metadata={
        "help": "Huggingface token for private models."
    })


def quant_layer(in_q, model_save_path, base_model, config, ft_rank, grad_ckpt, device,
                data_params, quant_params, hessian_save_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype='auto', low_cpu_mem_usage=True
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
            compute_hessians=False
        )
        layer_quant = weight_compressor.get_layer_quantizer(layer_idx, device)

        with torch.no_grad():
            layer = model.model.layers[layer_idx]

            for sublayer in layer_quant.sublayer_info.keys():
                print(f"Quantizing layer {layer_idx}, sublayer {sublayer}")
                layer_quant.compress_sublayer(sublayer)

                attr_names = layer_quant.sublayer_info[sublayer].out_key.split('.')
                setattr(
                    getattr(layer, attr_names[0]), attr_names[1],
                    layer_quant.get_quantized_linear_layer(
                        sublayer, ft_rank, grad_ckpt
                    )
                )
                layer_quant.clean_up_sublayer(sublayer)
            layer = layer.cpu()
            torch.save(
                layer,
                f"{model_save_path}/quant_layer_{layer_idx}.pt"
            )
            del layer_quant
            gc.collect()
            torch.cuda.empty_cache()


def quantize_save_llama(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    hessian_save_path: str = "./hessians/llama-2-7b",
    model_save_path: str = "./models/llama-2-7b",
    token: str = "",
    ft_rank: int = 64,
    grad_ckpt: bool = True,
    data_params: DataParameters = DataParameters(),
    quant_params: ActivationAwareQuantParams = ActivationAwareQuantParams(),
    quant_devices=["cuda"]
):

    os.makedirs(model_save_path, exist_ok=True)
    mp.set_start_method('spawn')

    if token:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype='auto', low_cpu_mem_usage=True, token=token
        ).cpu()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype='auto', low_cpu_mem_usage=True
        ).cpu()

    model_config = model.config
    n_layers = len(model.model.layers)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    manager = mp.get_context('spawn').Manager()
    in_q = manager.Queue()
    quant_procs = []

    for device in quant_devices:
        p = mp.Process(
            target=quant_layer,
            args=(in_q, model_save_path, base_model, 
                  model_config, ft_rank, grad_ckpt, device,
                  data_params, quant_params, hessian_save_path)
        )
        p.start()
        quant_procs.append(p)

    for layer_idx in range(n_layers):
        in_q.put(layer_idx)

    for _ in quant_devices:
        in_q.put(None)

    for p in quant_procs:
        p.join()


def load_quantized_model(
    model_save_path,
    base_model,
    device,
    sequence_classification=False
):
    if not sequence_classification:
        model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype='auto', device_map=device, low_cpu_mem_usage=True
        ).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model, torch_dtype='auto', device_map=device, low_cpu_mem_usage=True,
        ).to(device)
    
    if not sequence_classification:\
        model.lm_head.weight.requires_grad = False
    else:
        model.score.weight.requires_grad = True

    model.model.embed_tokens.weight.requires_grad = False
    model.model.norm.weight.requires_grad = False
    for layer_idx in range(len(model.model.layers)):
        layer = torch.load(
            f"{model_save_path}/quant_layer_{layer_idx}.pt",
            map_location=device
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
    
    if os.path.isfile(model_save_path + "/RHT_ft_model.safetensors"):
        load_model(model, model_save_path + "/RHT_ft_model.safetensors", strict=False)

    return model
        
if __name__ == '__main__':
    glog.setLevel("WARN")

    parser = HfArgumentParser([
        Arguments, ActivationAwareQuantParams, QuIPArgs])

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
        quant_devices=args.devices
    )
