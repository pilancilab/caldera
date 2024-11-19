import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from lib.utils import sample_rp1t
from lib.utils.data_utils import split_data
import gc
from quantize_save_llama import load_quantized_model
from dataclasses import dataclass, field
from tqdm import tqdm
from safetensors.torch import save_model, load_model
import os

# Adapted from https://github.com/Cornell-RelaxML/quip-sharp


@dataclass
class Arguments:
    base_model: str = field(metadata={
        "help": ("Path of the original model, as either a local or a "
                 "Huggingface path")
    })
    model_path: str = field(metadata={
        "help": ("Path of the .pt file in which the model can be found.")
    })
    finetuned_save_path: str = field(metadata={
        "help": ("Path in which to save the final finetuned model")
    })
    devset_size: int = field(default=256, metadata={
        "help": ("Number of datapoints to sample from the calibration set "
                 "for finetuning.")
    })
    ctx_size: int = field(default=512, metadata={
        "help": ("Length of each input data sequence.")
    })
    device: str = field(default="cuda", metadata={
        "help": "Device to use for finetuning."
    })
    ft_bs: int = field(default=2, metadata={
        "help": "Batch size for finetuning."
    })
    ft_valid_size: int = field(default=64, metadata={
        "help": ("Number of datapoints to set aside for validation. "
                 "The number of training datapoints is devset_size, minus "
                 "ft_valid_size.")
    })
    finetune_factors: bool = field(default=False, metadata={
        "help": ("Whether to finetune L and R in addition to the randomized "
                 "Hadamard transform diagonal matrices.")
    })
    RHT_learning_rate: float = field(default=1e-3, metadata={
        "help": "Learning rate for the randomized Hadamard transform parameters."
    })
    factors_learning_rate: float = field(default=1e-4, metadata={
        "help": "Learning rate for L andsR, if finetune_factors is set True."
    })
    epochs: int = field(default=5, metadata={
        "help": "Number of epochs of finetuning."
    })


def main(args: Arguments):
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    devset = sample_rp1t(tokenizer, args.devset_size, args.ctx_size, 1)

    # Get the logits for the calibration set from the original model. The loss
    # function for finetuning will be the cross-entropy loss between the quantized
    # model logits and these logits.
    orig_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype='auto', device_map=args.device, low_cpu_mem_usage=True
    ).to(args.device)
    
    orig_logits = torch.zeros(args.devset_size, args.ctx_size, orig_model.config.vocab_size)
    for i in range(args.devset_size):
        input = devset[i:i+1].to(args.device)
        output = orig_model(input).logits.cpu()
        orig_logits[i:i+1, :, :] = output

    orig_logits = orig_logits[:, :-1].contiguous().softmax(dim=-1).float() 
    del orig_model  
    gc.collect()
    torch.cuda.empty_cache()

    torch.set_grad_enabled(True)
    quant_model = load_quantized_model(
        args.model_path, args.base_model, args.device
    ).to(args.device).float()

    factor_params = []
    RHT_params = []
    
    for name, param in quant_model.named_parameters():
        if 'L_ft' in name or 'R_ft' in name:
            if not args.finetune_factors:
                param.requires_grad = False
            else:
                factor_params.append(param)
        elif 'SU' in name or 'SV' in name:
            RHT_params.append(param)
    train_dataloader, valid_dataloader = split_data(devset, orig_logits, args)
    
    adam_params = [{
        'params': RHT_params,
        'lr': args.RHT_learning_rate
    }]
    if args.finetune_factors:
        adam_params.append({
            'params': factor_params,
            'lr': args.factors_learning_rate
        })
    optim = torch.optim.AdamW(adam_params)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    quant_model.eval()
    print("Running eval.")
    with torch.no_grad():
        val_loss = 0
        for _, (input, targets) in enumerate(valid_dataloader):
            input, targets = input.to(args.device), targets.to(args.device)
            output = quant_model(input).logits[:, :-1].contiguous()

            val_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                targets.view(-1, targets.shape[-1])).item()
        val_loss /= len(valid_dataloader)
        print("Validation loss: ", val_loss)
        best_val_loss = val_loss
        save_model(quant_model, args.finetuned_save_path)

    progress_bar = tqdm(range(len(train_dataloader)*args.epochs))
    for _ in range(args.epochs):
        for _, (input, targets) in enumerate(train_dataloader):
            input, targets = input.to(args.device), targets.to(args.device)
            output = quant_model(input).logits[:, :-1].contiguous()

            loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                        targets.view(-1, targets.shape[-1]))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            progress_bar.update(1)
        
        # validation
        quant_model.eval()
        print("Running eval.")
        with torch.no_grad():
            val_loss = 0
            for _, (input, targets) in enumerate(valid_dataloader):
                input, targets = input.to(args.device), targets.to(args.device)
                output = quant_model(input).logits[:, :-1].contiguous()

                val_loss += nn.CrossEntropyLoss()(
                    output.view(-1, output.shape[-1]),
                    targets.view(-1, targets.shape[-1])).item()
            val_loss /= len(valid_dataloader)
            print("Validation loss: ", val_loss)
            if val_loss < best_val_loss:
                save_model(quant_model, args.finetuned_save_path)
                best_val_loss = val_loss
        quant_model.train()

    quant_model = load_quantized_model(
        args.model_path, args.base_model, args.device
    ).to(args.device)
    load_model(quant_model, args.finetuned_save_path)
    torch.save(quant_model, args.finetuned_save_path)

if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args()
    main(args)