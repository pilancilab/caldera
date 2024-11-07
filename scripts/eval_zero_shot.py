from transformers import AutoTokenizer
from lib.utils import LMEvalAdaptor
from lm_eval import evaluator
import os
import json
from quantize_save_llama import load_quantized_model
from dataclasses import field, dataclass
import transformers
import glob

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp

class LMEvalAdaptorWithDevice(LMEvalAdaptor):
    def __init__(self,
                 model_name,
                 model,
                 tokenizer,
                 batch_size=1,
                 max_length=-1,
                 device="cuda"):
        super().__init__(
            model_name, model, tokenizer, batch_size, max_length
        )
        self._device = device

    @property
    def device(self):
        return self._device


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
    tasks: list[str] = field(default_factory=list, metadata={
        "help": ("Task on which to measure zero-shot accuracy, e.g."
                 "wingorande, piqa, arc_easy, arc_challenge, rte, cola...")
    })
    batch_size: int = field(default=1, metadata={
        "help": "Number of datapoints processed at once"
    })
    device: str = field(default="cuda:0", metadata={
        "help": "Device on which to run evaluation."
    })
    cuda_graph: bool = field(default=False, metadata={
        "help": "Whether to use CUDA graphs and flash attention to speed up evaluation."
    })


def eval_zero_shot(args: Arguments):
    print(args.base_model)
    model = load_quantized_model(args.model_save_path, args.base_model, args.device, cuda_graph=args.cuda_graph)
    model = model.to(args.device)
    if args.finetune_save_dir is not None:
            from safetensors.torch import load_model
            for safetensor_file in glob.glob(args.finetune_save_dir + "/model*.safetensors"):
                print("Loading ", safetensor_file)
                load_model(model, safetensor_file, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    print("Loaded model!")

    tokenizer.pad_token = tokenizer.eos_token

    lm_eval_model = LMEvalAdaptorWithDevice(
        args.base_model, model, tokenizer, args.batch_size, device=args.device)
    lm_eval_model.device
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=args.tasks,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=0,
        device=args.device
    )

    print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.base_model
        with open(args.output_path + ".json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([Arguments])
    args = parser.parse_args_into_dataclasses()[0]
    eval_zero_shot(args)