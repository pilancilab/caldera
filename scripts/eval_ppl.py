import torch
from lib.utils import gptq_data_utils
from tqdm import tqdm
from quantize_save_llama import load_quantized_model
from dataclasses import dataclass, field
import transformers
import os
import json
import glob

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp

@dataclass
class Arguments:
    model_save_path: str = field(metadata={
        "help": ("Path of the .pt file in which the model can be found.")
    })
    finetune_save_dir: str = field(default=None, metadata={
        "help": ("If using a finetuned model, the directory in which the "
                 "model.safetensors file is stored")
    })
    output_path: str = field(default=None, metadata={
        "help": ("Path in which to save a JSON file with zero-shot results.")
    })
    base_model: str = field(default="meta-llama/Llama-2-7b-hf", metadata={
        "help": ("Path of the original model, as either a local or a "
                 "Huggingface path")
    })
    seed: int = field(default=0, metadata={
        "help": "Random seed for selecting test points from the dataset"
    })
    seqlen: int = field(default=4096, metadata={
        "help": "Sequence length of model inputs"
    })
    device: str = field(default="cuda:0", metadata={
        "help": "Device on which to run evaluation."
    })
    datasets: list[str] = field(default_factory=list, metadata={
        "help": ("Which datasets, out of \"wikitext2\" and \"c4\" to compute "
                 "perplexity. Defaults to both datasets")})
    cuda_graph: bool = field(default=False, metadata={
        "help": "Whether to use CUDA graphs and flash attention to speed up evaluation."
    })


def eval_ppl(args: Arguments):
    
    with torch.no_grad():
        model = load_quantized_model(args.model_save_path, args.base_model, args.device, cuda_graph=args.cuda_graph)

        if args.finetune_save_dir is not None:
            from safetensors.torch import load_model
            for safetensor_file in glob.glob(args.finetune_save_dir + "/model*.safetensors"):
                print("Loading ", safetensor_file)
                load_model(model, safetensor_file, strict=False)

            
        if not args.datasets:
            args.datasets = ["wikitext2", "c4"]

        ppls = {}

        for dataset in args.datasets:
            input_tok = gptq_data_utils.get_test_tokens(
                dataset, seed=args.seed, seqlen=args.seqlen, model=args.base_model)
            nsamples = input_tok.numel() // args.seqlen
            input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
                nsamples, args.seqlen)
        
            loss_fct = torch.nn.CrossEntropyLoss().cuda()
            acc_loss = 0.0
            progress = tqdm(range(nsamples))
            for ii in progress:
                input = input_tok[ii, :].to(args.device).view(1, -1)
                output = model(input,
                            use_cache=False,
                            output_hidden_states=False,
                            output_attentions=False)[0]

                shift_logits = output[:, :-1, :].contiguous()
                shift_labels = input[:, 1:]
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1))
                acc_loss += loss.item()
                progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

            avg_loss = acc_loss / nsamples

            ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f'{dataset} perplexity: {ppl}')

            ppls[dataset] = ppl

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path + ".json", "w") as f:
                json.dump(ppls, f, indent=2)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    eval_ppl(args)