from quantize_llama.hessian_offline_llama import forward_layer, accumulate
from lib.utils.data_utils import sample_rp1t, sample_falcon_refinedweb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

import torch
import torch.multiprocessing as mp
import gc

from caldera.decomposition.dataclasses import *
from caldera.utils.enums import DevSet
from caldera.decomposition.layer_quantization import \
    ActivationAwareLayerQuant

from caldera.utils.enums import DevSet
import os


# Partially adapted from https://github.com/Cornell-RelaxML/quip-sharp


class ActivationAwareWeightCompressor:
    """
    Sets up the framework for activation aware weight compression: loads in
    the model and calibration dataset, and then computes the inputs to each
    layer and the corresponding Hessian matrix (the second moment of the
    inputs). The inputs and Hessians are stored in files at
    `hessian_save_path`.

    Yor can instantiate an `ActivationAwareLayerQuant` for a given layer by
    calling the `get_layer_quantizer` method. See `ActivationAwareLayerQuant`
    for more usage details.

    Note: if you have already done Hessian computation and the data is stored
    in the appropriate files w.r.t. `Hessian_save_path`, you can pass in
    `compute_Hessians=False` to skip the data sampling and Hessian computation.
    """
    def __init__(
            self,
            model_params: ModelParameters = ModelParameters(),
            data_params: DataParameters = DataParameters(),
            quant_params: CalderaParams = CalderaParams(),
            quant_device: str = "cuda",
            hessian_save_path: str = "",
            start_at_layer: int = 0,
            stop_at_layer: int = None,
            n_sample_proc: int = 4,
            compute_hessians: bool = True
    ):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            print(("Warning: an exception occured when setting the "
                   "multiprocessing context. If you have previously "
                   "instantiated an ActivationAwareWeightCompressor, you can "
                   "ignore this."))

        self.compute_hessians = compute_hessians

        torch.set_grad_enabled(False)
        os.makedirs(hessian_save_path, exist_ok=True)

        # passed into `ActivationAwareLayerQuant` when `get_layer_quantizer`
        # is called:
        self.hessian_save_path = hessian_save_path
        self.quant_params = quant_params
        self.quant_device = quant_device
        self.data_params = data_params

        self._setup_model_and_data(
            model_params,
            data_params,
            n_sample_proc
        )

        if stop_at_layer is None:
            stop_at_layer = float('inf')

        if self.compute_hessians:
            # Loop through transformer layers and comput + save the Hessians
            for transformer_layer_index, transformer_layer \
                    in enumerate(self.model.model.layers):
                if transformer_layer_index < start_at_layer:
                    continue
                if transformer_layer_index >= stop_at_layer:
                    break
                self._process_layer(
                    transformer_layer_index=transformer_layer_index,
                    transformer_layer=transformer_layer,
                    data_params=data_params,
                    hessian_save_path=hessian_save_path
                )

    def get_layer_quantizer(
        self,
        layer_idx: int,
        device: str = None,
        label: str = None
    ):
        """
        Instantiates an `ActivationAwareLayerQuant` object for a given
        transformer layer.
        """
        assert layer_idx >= 0 and layer_idx < len(self.model.model.layers)

        if device is None:
            device = self.quant_device

        return ActivationAwareLayerQuant(
            layer=self.model.model.layers[layer_idx],
            layer_idx=layer_idx,
            hessian_save_path=self.hessian_save_path,
            quant_params=self.quant_params,
            device=device,
            label=label
        )

    def _setup_model_and_data(
            self,
            model_params: ModelParameters,
            data_params: DataParameters,
            n_sample_proc: int  # Number of processes used to sample the
                                # calibration dataset. Unrelated to Hessian
                                # computation.
    ):
        """
        Loads in the model and calibration dataset.
        """
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_params.base_model, torch_dtype="auto", low_cpu_mem_usage=True,
            token=model_params.token
        )

        if self.compute_hessians:
            # Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_params.base_model, use_fast=True,
                token=model_params.token
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Calibration dataset
            if data_params.devset == DevSet.RP1T:
                self.devset = sample_rp1t(tokenizer,
                                          data_params.devset_size,
                                          data_params.context_length,
                                          nproc=n_sample_proc)
                self.dev_emb = self.model.model.embed_tokens(self.devset)
            elif data_params.devset == DevSet.FALCON:
                self.devset = sample_falcon_refinedweb(
                    tokenizer, data_params.devset_size,
                    data_params.context_length,
                    nproc=n_sample_proc
                )
                self.dev_emb = self.model.model.embed_tokens(self.devset)
            else:
                raise NotImplementedError("Dataset not implemented yet")
            self.dev_emb.share_memory_()

            # Attention mask and position IDs
            self.position_ids = torch.arange(
                    data_params.context_length, dtype=torch.int64
                )[None, :] + torch.zeros(
                    data_params.batch_size,
                    data_params.context_length,
                    dtype=torch.int64
                )

            if hasattr(self.model.config, 'sliding_window'):
                self.attention_mask = _prepare_4d_causal_attention_mask(
                    None, (data_params.batch_size, data_params.context_length),
                    self.dev_emb[0:data_params.batch_size], 0,
                    sliding_window=self.model.config.sliding_window
                )
            else:
                self.attention_mask = _prepare_4d_causal_attention_mask(
                    None, (data_params.batch_size, data_params.context_length),
                    self.dev_emb[0:data_params.batch_size], 0
                )

    def _process_layer(
        self,
        transformer_layer_index: int,
        transformer_layer: torch.nn.Module,
        data_params: DataParameters,
        hessian_save_path: str
    ):
        """
        Compute the layer inputs and Hessians via the same process as
        quip-sharp/quantize_llama/hessian_offline_llama.py.
        """
        # Check that there are four layers (QKV + 4 MLP), as expected
        assert (len([
            m for m in transformer_layer.modules()
            if isinstance(m, torch.nn.Linear)
        ]) == 7)

        chunk_size = min(data_params.chunk_size, len(self.dev_emb))

        devices_available = data_params.devices if \
            data_params.devices is not None else \
            range(torch.cuda.device_count())
        ngpus = min(len(devices_available), len(self.dev_emb) // chunk_size)
        devices = devices_available[:ngpus]
        print(f"Computing hessians on {devices}")

        manager = mp.get_context('spawn').Manager()
        in_q = manager.Queue()
        out_q = manager.Queue()

        accumulate_proc = mp.Process(
            target=accumulate,
            args=(
                out_q, None, ngpus,
                AccumulatorArgs(save_path=hessian_save_path),
                transformer_layer_index
            )
        )
        accumulate_proc.start()

        forward_procs = []
        for device in devices:
            p = mp.Process(
                target=forward_layer,
                args=(
                    transformer_layer,
                    self.position_ids,
                    self.attention_mask,
                    data_params.batch_size,
                    device, in_q, out_q
                )
            )
            p.start()
            forward_procs.append(p)

        assert len(self.dev_emb) % data_params.batch_size == 0 and \
            chunk_size % data_params.batch_size == 0
        i = 0
        while i < len(self.dev_emb):
            next = min(i + chunk_size, len(self.dev_emb))
            in_q.put(self.dev_emb[i:next])
            i = next

        for device in devices:
            in_q.put(None)

        for p in forward_procs:
            p.join()

        accumulate_proc.join()

        transformer_layer.cpu()
        # self.model.model.layers[transformer_layer_index] = None
        gc.collect()
        torch.cuda.empty_cache()

        print(f"done processing layer {transformer_layer_index}")
