import glog
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from lib.utils import sample_rp1t
from lib.utils.data_utils import split_data
import gc
from quantize_save_llama import load_quantized_model
from dataclasses import dataclass, field
from tqdm import tqdm
from safetensors.torch import save_model


@dataclass
class Arguments:
    base_model: str = field()
    model_save_path: str = field()
    devset_size: int = field(default=256)
    ctx_size: int = field(default=512)
    device: str = field(default="cuda")
    ft_bs: int = field(default=2)
    ft_valid_size: int = field(default=64)
    finetune_LR: bool = field(default=False)
    RHT_lr: float = field(default=1e-3)
    factors_lr: float = field(default=1e-4)
    epochs: int = field(default=5)


def main(args: Arguments):
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    devset = sample_rp1t(tokenizer, args.devset_size, args.ctx_size, 1)

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
        args.model_save_path, args.base_model, 
        args.device
    ).to(args.device).float()

    factor_params = []
    RHT_params = []
    
    for name, param in quant_model.named_parameters():
        if 'L_ft' in name or 'R_ft' in name:
            if not args.finetune_LR:
                param.requires_grad = False
            else:
                factor_params.append(param)
        elif 'SU' in name or 'SV' in name:
            RHT_params.append(param)
    # emb = quant_model.model.embed_tokens(devset)

    train_dataloader, valid_dataloader = split_data(devset, orig_logits, args)

    # position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
    #     torch.zeros(args.ft_bs, args.ctx_size, dtype=torch.int32)
    # position_ids = position_ids.to(args.device)
    # attention_mask = _prepare_4d_causal_attention_mask(
    #     None, (args.ft_bs, args.ctx_size), emb[:args.ft_bs], 0).to(args.device)
    
    adam_params = [{
        'params': RHT_params,
        'lr': args.RHT_lr
    }]
    if args.finetune_LR:
        adam_params.append({
            'params': factor_params,
            'lr': args.factors_lr
        })
    optim = torch.optim.AdamW(adam_params)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    quant_model.eval()
    print("Running eval.")
    with torch.no_grad():
        val_loss = 0
        for idx, (input, targets) in enumerate(valid_dataloader):
            input, targets = input.to(args.device), targets.to(args.device)
            output = quant_model(input).logits[:, :-1].contiguous()

            val_loss += nn.CrossEntropyLoss()(
                output.view(-1, output.shape[-1]),
                targets.view(-1, targets.shape[-1])).item()
        val_loss /= len(valid_dataloader)
        print("Validation loss: ", val_loss)
        best_val_loss = val_loss
        save_model(quant_model, args.model_save_path + "/RHT_ft_model.safetensors")

    progress_bar = tqdm(range(len(train_dataloader)*args.epochs))
    for epoch in range(args.epochs):
        for idx, (input, targets) in enumerate(train_dataloader):
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
            for idx, (input, targets) in enumerate(valid_dataloader):
                input, targets = input.to(args.device), targets.to(args.device)
                output = quant_model(input).logits[:, :-1].contiguous()

                val_loss += nn.CrossEntropyLoss()(
                    output.view(-1, output.shape[-1]),
                    targets.view(-1, targets.shape[-1])).item()
            val_loss /= len(valid_dataloader)
            print("Validation loss: ", val_loss)
            if val_loss < best_val_loss:
                save_model(quant_model, args.model_save_path + "/RHT_ft_model.safetensors")
                best_val_loss = val_loss
        quant_model.train()


if __name__ == "__main__":
    parser = HfArgumentParser([Arguments])
    args = parser.parse_args()
    main(args)