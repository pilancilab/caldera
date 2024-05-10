from lib.utils.data_utils import flat_to_sym
from lib.utils.math_utils import regularize_H

import torch
import matplotlib.pyplot as plt
import gc

from lplr_llm.activation_aware.dataclasses import *
from lplr_llm.utils.enums import TransformerSubLayers
from lplr_llm.activation_aware.lplr_q import LPLRq

from dataclasses import asdict
import json
import datetime


# Maps number of bits to name of the QuIP# lattice quantizer
BITS_TO_CODEBOOK = {
    2: 'E8P12',
    3: 'E8P12RVQ3B',
    4: 'E8P12RVQ4B'
}


class ActivationAwareLayerQuant:
    """
    For a given transformer layer, decomposes each sublayer (i.e., one of
    {query, key, value, out, gate, up, down}) into Q + LR, where Q is a
    matrix quantized according to QuIP# and L, R are low-rank and quantized
    factors via an iterative, activation-aware procedure.

    This class is instantiated by `ActivationAwareWeightCompressor` upon
    calling the method `get_layer_quantizer`.

    Usage example:
    ```
    # Instantiate ActivationAwareWeightCompressor. This will automatically
    # compute all the Hessians upon initialization, unless you pass in the
    # `stop_at_layer` keyword argument
    weight_compressor = ActivationAwareWeightCompressor(
        model_params=ModelParameters(
            base_model="meta-llama/Llama-2-7b-hf"
        ),
        data_params=DataParameters(
            devset=DevSet.FALCON,
            devset_size=128,
            context_length=4096,
            batch_size=2,
            devices=["cuda", "cuda:2"]
        ),
        hessian_save_path="./data/hessians/llama-7b",
        quant_params=ActivationAwareQuantParams(
            Q_bits=2,
            L_bits=3,
            R_bits=4,
            rank=256,
            iters=25,
            lplr_iters=3
        ),
        quant_device="cuda:2"
    )

    # Instantiate ActivationAwareLayerQuant for a layer
    layer_quant = weight_compressor.get_layer_quantizer(layer_idx=12)

    # Quantize one sub-layer using QuIP + LPLR
    layer_quant.compress_sublayer(TransformerSubLayers.VALUE)

    # Plot the quantization error
    best_error = layer_quant.min_error(TransformerSubLayers.VALUE)
    layer_quant.plot_errors(TransformerSubLayers.VALUE)

    # Delete Q, L, and R matrices to free GPU memory before quantizing another
    # sublayer (optional)
    layer_quant.clean_up_sublayer(TransformerSubLayers.VALUE)
    ```
    """
    def __init__(
        self,
        layer: torch.nn.Module,
        layer_idx: int,
        hessian_save_path: str = "",
        quant_params: ActivationAwareQuantParams = ActivationAwareQuantParams(),
        label: str = "LPLR-LDLQ",
        device: str = "cuda",
    ):
        self.hessian_save_path = hessian_save_path
        self.layer = layer.to(device)
        self.label = label
        if label is None:
            self.label = "LPLR-LDLQ"

        self.layer_idx = layer_idx
        self.quant_params = quant_params
        if not self.quant_params.update_order:
            if not self.quant_params.compute_low_rank_factors:
                self.quant_params.update_order = ["Q"]
            elif not self.quant_params.compute_quantized_component:
                self.quant_params.update_order = ["LR"]
            else:
                self.quant_params.update_order = ["LR", "Q"]
        if not self.quant_params.compute_low_rank_factors:
            self.quant_params.rank = 0
        self.device = device

        self._set_sublayer_weights_and_info()

    def compress_sublayer(self, sublayer):
        """
        Decomposes a sublayer (e.g., query, key, value, etc.) by alternating
        between QuIP# and LPLR.

        The sublayer argument must be a member of the TransformerSubLayers
        enum.
        """
        assert sublayer in self.sublayer_info.keys(), \
            ("Invalid sublayer! Please use a member of the "
             "TransformerSubLayers enum.")
        sublayer_info = self.sublayer_info[sublayer]
        sublayer_info.started_quant = True

        sublayer_info.lplr_q.W = sublayer_info.sublayer.weight.to(self.device).float()
        

        W = sublayer_info.lplr_q.W
        H = self._get_H(sublayer)

        sublayer_info.lplr_q = LPLRq(self.quant_params, W, H, self.device)

    def get_quantized_linear_layer(self, sublayer, ft_rank, grad_ckpt=True):
        from lplr_llm.activation_aware.quantized_layer import \
            LPLRQuantizedLinear

        sublayer_info = self._get_sublayer_info_and_check_sublayer(sublayer)

        have_L_codebook = self.quant_params.lattice_quant_LR and self.quant_params.compute_low_rank_factors and \
            self.quant_params.L_bits < 16
        have_R_codebook = self.quant_params.lattice_quant_LR and self.quant_params.compute_low_rank_factors and \
            self.quant_params.R_bits < 16

        L_codebook_version = BITS_TO_CODEBOOK[self.quant_params.L_bits] if have_L_codebook else None

        R_codebook_version = BITS_TO_CODEBOOK[self.quant_params.R_bits] if have_R_codebook else None

        if ft_rank < self.quant_params.rank and self.quant_params.compute_low_rank_factors and \
                ((L_codebook_version is None and self.quant_params.L_bits < 16)
                or (R_codebook_version is None and
                    self.quant_params.R_bits < 16)):
            raise NotImplementedError(
                "Only lattice quantization for L and R implemented so far"
            )

        return LPLRQuantizedLinear(
            # Dimensions
            in_features=sublayer_info.lplr_q.W.shape[1],
            out_features=sublayer_info.lplr_q.W.shape[0],
            # Codebooks
            Q_codebook_version=BITS_TO_CODEBOOK[self.quant_params.Q_bits],
            L_codebook_version=L_codebook_version,
            R_codebook_version=R_codebook_version,
            # L and R
            L=sublayer_info.lplr_q.L,
            R=sublayer_info.lplr_q.R,
            # Quantized idxs
            L_idxs=sublayer_info.lplr_q.L_idxs,
            R_idxs=sublayer_info.lplr_q.R_idxs,
            Q_idxs=sublayer_info.lplr_q.Q_idxs,
            # Scaling
            L_scale=sublayer_info.lplr_q.L_scale,
            R_scale=sublayer_info.lplr_q.R_scale,
            Q_scale=sublayer_info.lplr_q.Q_scale,
            global_scale=sublayer_info.lplr_q.global_scale,
            scaleWH=sublayer_info.lplr_q.scaleWH,
            # Hadamard
            hadamard=(self.quant_params.hadamard_transform
                      or self.quant_params.full_quip_sharp),
            # SU and SV
            SU=sublayer_info.lplr_q.SU,
            SV=sublayer_info.lplr_q.SV,
            # Rank and fine-tuning
            rank=max(self.quant_params.rank,
                     self.quant_params.quip_args.lora_rank),
            ft_rank=ft_rank,
            grad_ckpt=grad_ckpt
        )

    def plot_errors(self, sublayer, plot_first_iter=True, savefile=None):
        """
        Plot the per-iteration approximation errors for a given sublayer
        (i.e., a member of the TransformerSubLayers enum).
        """
        sublayer_info = self._get_sublayer_info_and_check_sublayer(sublayer)
        self._plot(sublayer_info.lplr_q.errors, plot_first_iter, savefile=savefile)

    def _plot(self, errors, plot_first_iter, savefile=None):
        COLORS = ['b', 'r', 'm']
        plt.figure(figsize=(12, 4))
        title = f"Activation-Aware Error per iteration: {self.label}"
        plt.title(title)
        for i, key in enumerate(errors.keys()):
            if plot_first_iter or len(errors[key] )== 1:
                plt.plot(range(len(errors[key])), errors[key],
                         marker='o', linestyle='-', color=COLORS[i], label=key)
            else:
                plt.plot(range(1, len(errors[key])), errors[key][1:],
                         marker='o', linestyle='-', color=COLORS[i], label=key)
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        if savefile is not None:
            plt.savefig(savefile)
            plt.close()
        else:
            plt.show()

    def plot_errors_json(self, file, plot_first_iter=True, savefile=None):
        with open(file, "r") as infile:
            json_str = infile.read()
        data = json.loads(json_str)
        self._plot(data["per_iter_errors"], plot_first_iter, savefile=savefile)

    def min_error(self, sublayer):
        """
        Returns the minimum value of the activation-aware loss for a given
        sublayer (i.e., a member of the TransformerSubLayers enum).
        """
        sublayer_info = self._get_sublayer_info_and_check_sublayer(sublayer)
        min_error = float('inf')
        for key in sublayer_info.lplr_q.errors:
            min_error = min(min_error, min(sublayer_info.lplr_q.errors[key]))
        return min_error

    def export_errors_json(self, sublayer, savefile):
        sublayer_info = self._get_sublayer_info_and_check_sublayer(sublayer)
        param_dict = asdict(self.quant_params)
        del param_dict["quant_factory_Q"]
        del param_dict["quant_factory_LR"]

        now = datetime.datetime.now()

        data = {
            "per_iter_errors": sublayer_info.lplr_q.errors,
            "layer_idx": self.layer_idx,
            "sublayer": sublayer_info.key,
            "datetime": str(now),
            "timestamp": datetime.datetime.timestamp(now),
            "params": param_dict
        }
        json_object = json.dumps(data)
        with open(savefile + ".json", "w") as out:
            out.write(json_object)

    def clean_up_sublayer(self, sublayer):
        """
        Delete Q, L, and R matrices for a sublayer to free GPU memory.
        """
        sublayer_info = self._get_sublayer_info_and_check_sublayer(sublayer)

        self.sublayer_info[sublayer] = SubLayerInfo(
            sublayer=sublayer_info.sublayer, key=sublayer_info.key
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _set_sublayer_weights_and_info(self):
        """
        Initializes a SubLayerInfo object for each of the seven transformer
        sublayers. Called upon instantiation.
        """
        self.sublayer_info = {
            TransformerSubLayers.KEY: SubLayerInfo(
                sublayer=self.layer.self_attn.k_proj, key="qkv",
                out_key="self_attn.k_proj"),
            TransformerSubLayers.QUERY: SubLayerInfo(
                sublayer=self.layer.self_attn.q_proj, key="qkv",
                out_key="self_attn.q_proj"),
            TransformerSubLayers.VALUE: SubLayerInfo(
                sublayer=self.layer.self_attn.v_proj, key="qkv",
                out_key="self_attn.v_proj"),
            TransformerSubLayers.O: SubLayerInfo(
                sublayer=self.layer.self_attn.o_proj, key="o",
                out_key="self_attn.o_proj"),
            TransformerSubLayers.UP: SubLayerInfo(
                sublayer=self.layer.mlp.up_proj, key="up",
                out_key="mlp.up_proj"),
            TransformerSubLayers.GATE: SubLayerInfo(
                sublayer=self.layer.mlp.gate_proj, key="up",
                out_key="mlp.gate_proj"),
            TransformerSubLayers.DOWN: SubLayerInfo(
                sublayer=self.layer.mlp.down_proj, key="down",
                out_key="mlp.down_proj")
        }

    def _get_H(self, sublayer):
        """
        Reads the Hessian (sum X_i X_i^T) for a specific sublayer from the
        corresponding file (in which it was saved by
        ActivationAwareWeightCompressor).
        """
        sublayer_key = self.sublayer_info[sublayer].key
        H_data = torch.load(
            f'{self.hessian_save_path}/{self.layer_idx}_{sublayer_key}.pt',
            map_location=torch.device(self.device),
        )
        H = flat_to_sym(H_data['flatH'], H_data['n'])

        # Add back in the mean
        mu = H_data['mu']
        H.add_(mu[None, :] * mu[:, None])
        H.div_(torch.diag(H).mean())
        H = regularize_H(H, H_data['n'], self.quant_params.quip_args.sigma_reg)

        return H

    def _get_sublayer_info_and_check_sublayer(self, sublayer):
        assert sublayer in self.sublayer_info.keys(), \
            ("Invalid sublayer! Please use a member of the "
             "TransformerSubLayers enum.")

        sublayer_info = self.sublayer_info[sublayer]
        assert sublayer_info.started_quant, \
               "Sublayer has't been quantized yet!"
        return sublayer_info
