from caldera.utils.enums import DevSet
from caldera.utils.quantization import QuantizerFactory, \
    AbstractQuantizer, LowMemoryQuantizer

from dataclasses import field, dataclass
import torch


@dataclass
class DataParameters:
    """
    Parameters for loading the calibration dataset and computing the
    inputs to each layer.
    """
    devset: int = field(
        default=DevSet.RP1T, metadata={"help": (
            "Calibration dataset, as a member of the DevSet enum."
        )}
    )
    devset_size: int = field(
        default=256, metadata={"help": (
            "Number of calibration samples to use."
        )}
    )
    context_length: int = field(
        default=4096, metadata={"help": (
            "Length of context window."
        )}
    )
    batch_size: int = field(
        default=2, metadata={"help": (
            "Number of datapoints to pass into the model at once."
        )}
    )
    chunk_size: int = field(
        default=256, metadata={"help": (
            "Number of datapoints sent to each GPU at a time. "
            "Must be a multiple of batch_size"
        )}
    )
    devices: list[str] = field(
        default=None, metadata={"help": (
            "Specific CUDA devices to use for Hessian computation. Defaults "
            "to None, which means that all available devices are used."
        )}
    )


@dataclass
class ModelParameters:
    """
    Parameters for loading in a transformer model and simulating forward
    passes.
    """
    base_model: str = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": (
            "Model to quantize."
        )}
    )
    token: str = field(default=None, metadata={
        "help": "Huggingface access token for private models."
    })


@dataclass
class AccumulatorArgs:
    """
    These arguments will be passed into the `accumulator` function in
    quip-sharp/quantize_llama/hessian_offline_llama.py. This class will be
    automatically instantiated by ActivationAwareWeightCompressor.
    """
    scratch_path = None
    save_path: str = field(default="./hessians/")


@dataclass
class QuIPArgs:
    """
    Parameters for QuIP. See the documentation of the quip-sharp repository
    for descriptions of these (or some of these) parameters.
    """
    lora_rank: int = field(default=0)
    full_svd: bool = field(default=False)
    use_fp64: bool = False
    lowmem_ldlq: bool = field(default=False)
    scale_override: float = field(default=-1)
    resid_scale_override: float = field(default=-1)
    no_use_buffered: bool = field(default=False)
    sigma_reg: float = field(default=1e-2)
    sigma_reg2: float = field(default=1e-2)
    incoh_mode: str = field(default="had", metadata={
        "help": ("Which form of incoherence processing to use. Either \"had\""
                 "for a randomized Hadamard transform, or \"kron\" for a "
                 "randomized Kronecker product of 2x2 matrices.")
    })
    rescale_WH: bool = field(default=False)
    quip_tune_iters: int = field(default=10, metadata={
        "help": ("Number of iterations in the LDLQ step.")
    })


@dataclass
class CalderaParams:
    """
    Parameters for the CALDERA decomposition.
    """
    quip_args: QuIPArgs = field(default_factory=QuIPArgs)
    compute_quantized_component: bool = field(
        default=True, metadata={"help": (
            "Whether the decomposition should include a quantized full-size"
            "component (denoted Q)."
        )}
    )
    compute_low_rank_factors: bool = field(
        default=True, metadata={"help": (
            "Whether the decomposition should include low-rank factors (L, R)."
        )}
    )
    Q_bits: int = field(default=2, metadata={
        "help": "Either 2, 3, or 4 bit lattice quantization"
    })
    L_bits: int = field(default=2, metadata={
        "help": "Either 2, 3, or 4 bit lattice quantization"
    })
    R_bits: int = field(default=2, metadata={
        "help": "Either 2, 3, or 4 bit lattice quantization"
    })
    rank: int = field(default=64, metadata={
        "help": "Rank of L and R factors"
    })
    iters: int = field(default=20)
    lplr_iters: int = field(default=5)
    activation_aware_Q: bool = field(default=True, metadata={
        "help": ("Use QuIP# activation-aware quantization for Q, as opposed "
                 "to naive quantization.")
    })
    activation_aware_LR: bool = field(default=True, metadata={
        "help": "Use activation-aware LPLR for computing the factors."
    })
    lattice_quant_Q: bool = field(default=True, metadata={
        "help": ("If Q is not data-aware, this determines whether to use "
                 "lattice quantization, as opposed to unif/normal float quantization "
                 "implementations.")
    })
    lattice_quant_LR: bool = field(default=True, metadata={
        "help": ("Use lattice quantization from the QuIP# codebase, as opposed"
                 " to uniform or normal float, for L and R")
    })
    hadamard_transform: bool = field(default=False, metadata={
        "help": ("Whether to perform a randomized Hadamard transform on W "
                 "before computing the decomposition W = Q + LR.")
    })
    full_quip_sharp: bool = field(default=False, metadata={
        "help": ("If Q is activation-aware and this parameter is True, then "
                 "Q is computed using the full quip-sharp algorithm. "
                 "Otherwise, we only use LDLQ.")
    })
    update_order: list[str] = field(default_factory=list, metadata={
        "help": ("List specifying whether to update the \"LR\" factors before "
                 "\"q\" or vice versa. The default is [\"LR\", \"Q\"]; pass "
                 "in [\"Q\", \"LR\"] to swap the update order.")
    })
    quant_factory_Q: QuantizerFactory = field(
        default_factory=QuantizerFactory, metadata={"help": (
            "(Non-data-aware only) QuantizerFactory (from caldera.utils.quantizers)"
            "  object used to instantiate quantizer for Q. Only used if "
            "activation_aware_Q is False."
        )}
    )
    quant_factory_LR: QuantizerFactory = field(
        default_factory=QuantizerFactory, metadata={"help": (
            "(Non-lattice quant only) QuantizerFactory (from "
            "caldera.utils.quantizers) object used to instantiate quantizer for L "
            "and R. Only used if lattice_quant_LR is False."
        )}
    )
    rand_svd: bool = field(default=True, metadata={
        "help": "Whether to use randomized SVD for LPLR initialization"
    })
    Q_hessian_downdate: bool = field(default=False, metadata={
        "help": ("Whether to do quip-sharp's heuristic Hessian correction"
                 "via Cholesky downdating before updating Q.")
    })
    lattice_quant_block_size: int = field(default=32000, metadata={
        "help": ("For lattice quantization, quantize parameters in groups of "
                 "(codesize * lattice_quant_block_size) to reduce memory "
                 "usage")
    })

@dataclass
class CalderaDecomposition:
    Q: torch.Tensor = field(default=None)
    L: torch.Tensor = field(default=None)
    R: torch.Tensor = field(default=None)
    W: torch.Tensor = field(default=None)
    Q_idxs: torch.Tensor = field(default=None)
    L_idxs: torch.Tensor = field(default=None)
    R_idxs: torch.Tensor = field(default=None)
    Q_scale: float = field(default=1)
    L_scale: float = field(default=1)
    R_scale: float = field(default=1)
    global_scale: float = field(default=1)
    SU: torch.Tensor = field(default=None)
    SV: torch.Tensor = field(default=None)
    scaleWH: torch.Tensor = field(default=None)
    errors: dict[str,list[float]] = field(default_factory=dict)


@dataclass
class SubLayerInfo:
    """
    Class for storing information about a transformer sub-layer (i.e., one of
    {query, key, value, out, gate, up, down}), including the computed
    decomposition Q + LR, and the activation-aware error at each iteration of
    the CALDERA algorithm.
    """
    sublayer: torch.nn.Module = field(default=None)
    key: str = field(default="")
    out_key: str = field(default="")
    started_quant: bool = field(default=False)
    caldera: CalderaDecomposition = field(default_factory=CalderaDecomposition)


@dataclass
class QuantInfo:
    """
    Stores information necessary for quantizing a specific matrix:
        1. Whether to use lattice quantization (QuIP#) or Unif./NormalFloat
            quantization.
        2. If lattice quantization is used, the codebook.
        3. If our quantization methods are used, the quantizer object.
    """
    lattice_quant: bool = field(default=True)
    lattice_cb: torch.nn.Module = field(default=None)
    quant: AbstractQuantizer = field(default_factory=LowMemoryQuantizer)

