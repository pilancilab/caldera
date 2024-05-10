import torch
from lib.utils import clean, dtype_from_str, get_hadK
from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda
from tqdm import tqdm
from copy import deepcopy
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask
from lplr_llm.activation_aware.dataclasses import DataParameters
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils.data_utils import sample_rp1t, sample_falcon_refinedweb
from lplr_llm.utils.enums import DevSet, TransformerSubLayers
import gc
from collections import namedtuple


def get_sublayer_input(
    layer_idx: int,
    sublayer: str,
    base_model: str,
    data_params: DataParameters = DataParameters(),
    n_sample_proc: int = 4,
    device: str = "cuda",
    layer_input: torch.Tensor=None,
    attention_mask: torch.Tensor = None,
    position_ids: torch.Tensor = None,
):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", low_cpu_mem_usage=True)

    # Get input to this layer, which is the output of the previous layer
    if layer_input is None:
        # Calibration dataset
        if data_params.devset == DevSet.RP1T:
            devset = sample_rp1t(
                tokenizer, data_params.devset_size,
                data_params.context_length, nproc=n_sample_proc)
            dev_emb = model.model.embed_tokens(devset)
        elif data_params.devset == DevSet.FALCON:
            devset = sample_falcon_refinedweb(
                tokenizer, data_params.devset_size,
                data_params.context_length,
                nproc=n_sample_proc
            )
            dev_emb = model.model.embed_tokens(devset)
        else:
            raise NotImplementedError("Dataset not implemented yet")
        
        position_ids = torch.arange(
            data_params.context_length, dtype=torch.int64
        )[None, :] + torch.zeros(
            data_params.batch_size, data_params.context_length,
            dtype=torch.int64
        )

        if hasattr(model.config, 'sliding_window'):
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (data_params.batch_size, data_params.context_length),
                dev_emb[0:data_params.batch_size], 0,
                sliding_window=model.config.sliding_window
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (data_params.batch_size, data_params.context_length),
                dev_emb[0:data_params.batch_size], 0
            )

        position_ids = position_ids.to(device)
        attention_mask = attention_mask.to(device)

        for n in range(layer_idx):
            for i in tqdm(range(0, dev_emb.shape[0], data_params.batch_size)):
                layer = model.model.layers[n].to(device)
                input = dev_emb[i:i+data_params.batch_size].to(device)
                dev_emb[i:i+data_params.batch_size] = layer(
                    input, position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False)[0].cpu()

                del layer
                del input
                gc.collect()
                torch.cuda.empty_cache()

    else:
        dev_emb = layer_input

    # Get the input to the particular sublayer
    layer = model.model.layers[layer_idx].to(device)
    captured_inputs = None
    def hook(_, x):
        nonlocal captured_inputs
        captured_inputs = x[0].cpu()

    attr_names = sublayer.split('.')

    sublayer_obj = getattr(
        getattr(layer, attr_names[0]), attr_names[1])
    handle = sublayer_obj.register_forward_pre_hook(hook)

    input = torch.zeros(
        dev_emb.shape[0], dev_emb.shape[1], sublayer_obj.weight.shape[0],
        dtype=dev_emb.dtype)
    for i in tqdm(range(0, dev_emb.shape[0], data_params.batch_size)):
        layer(
            dev_emb[i:i+data_params.batch_size].to(device),
            position_ids=position_ids, attention_mask=attention_mask,
            use_cache=False, output_attentions=False)
        input[i:i+data_params.batch_size] = captured_inputs
        gc.collect()
        torch.cuda.empty_cache()

    handle.remove()

    ret_type = namedtuple(
        'LayerInput',
        ['layer_in', 'sublayer_in', 'attention_mask', 'position_ids'])
    return ret_type(dev_emb, input, attention_mask, position_ids)