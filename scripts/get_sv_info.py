import torch
from transformers import AutoModelForCausalLM, HfArgumentParser
import gc
from tqdm import tqdm
from dataclasses import dataclass, field
import os


@dataclass
class Arguments:
    base_model: str = field(default="meta-llama/Llama-2-7b-hf", metadata={
        "help": ("Path of the original model, as either a local or a "
                 "Huggingface path")
    })
    device: str = field(default="cuda:0", metadata={
        "help": "Device on which to run evaluation."
    })
    save_dir: str = field(default="./data")

SUBLAYERS = {
    "q_proj": lambda layer: layer.self_attn.q_proj,
    "k_proj": lambda layer: layer.self_attn.k_proj,
    "v_proj": lambda layer: layer.self_attn.v_proj,
    "o_proj": lambda layer: layer.self_attn.o_proj,
    "up_proj": lambda layer: layer.mlp.up_proj,
    "gate_proj": lambda layer: layer.mlp.gate_proj,
    "down_proj": lambda layer: layer.mlp.down_proj
}

def main(base_model, device, save_dir):
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype='auto', low_cpu_mem_usage=True
    ).cpu()

    for label in SUBLAYERS:
        A = SUBLAYERS[label](model.model.layers[0]).weight
        n = min(A.shape[0], A.shape[1])
        SVs = torch.zeros(n, 32, requires_grad=False)
        for i, layer in tqdm(enumerate(model.model.layers)):
            A = SUBLAYERS[label](layer).weight.to(device).float().detach()
            _, S, _ = torch.linalg.svd(A)
            S = S.cpu()
            del A
            gc.collect()
            torch.cuda.empty_cache()

            SVs[:, i] = S

        torch.save({
            "SV_data": SVs.detach(),
            "means": torch.mean(SVs.detach(), dim=1),
            "stdevs": torch.std(SVs.detach(), dim=1)
        }, f"{save_dir}/{label}_sv_info.pt")

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    os.makedirs(os.path.dirname(args.save_dir), exist_ok=True)
    main(args.base_model, args.device, args.save_dir)

