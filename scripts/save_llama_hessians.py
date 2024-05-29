import transformers
from caldera.decomposition.weight_compression import *


@dataclass
class Arguments:
    base_model: str = field(default="meta-llama/Llama-2-7b-hf", metadata={
        "help": ("Path of the model that will eventually be quantized, as "
                 "either a local or a Huggingface path.")
    })
    token: str = field(default=None, metadata={
        "help": "Huggingface access token for private models."
    })
    hessian_save_path: str = field(
        default="./data/hessians/llama-2-7b", metadata={
            "help": ("Directory in which to save Hessians.")
        })
    n_sample_proc: int = field(default=4, metadata={
        "help": "Number of processes used to sample calibration data."
    })


@dataclass
class DataParametersCommandLine:
    """
    Parameters for loading the calibration dataset and computing the
    inputs to each layer.
    """
    devset: str = field(
        default="rp1t", metadata={"help": (
            "Calibration dataset; either rp1t or falcon"
        ), "choices": ["rp1t", "falcon"]}
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
            "Number of batches sent to each GPU at a time."
        )}
    )
    devices: list[str] = field(
        default=None, metadata={"help": (
            "Specific CUDA devices to use for Hessian computation. Defaults "
            "to None, which means that all available devices are used."
        )}
    )


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([Arguments, DataParametersCommandLine])
    args, data_params_command_line = parser.parse_args_into_dataclasses()

    devset = DevSet.RP1T if data_params_command_line.devset == "rp1t" \
             else DevSet.FALCON

    data_params = DataParameters(
        devset=devset,
        devset_size=data_params_command_line.devset_size,
        context_length=data_params_command_line.context_length,
        batch_size=data_params_command_line.batch_size,
        chunk_size=data_params_command_line.chunk_size,
        devices=data_params_command_line.devices
    )

    ActivationAwareWeightCompressor(
        model_params=ModelParameters(
            base_model=args.base_model,
            token=args.token
        ),
        data_params=data_params,
        hessian_save_path=args.hessian_save_path,
        quant_device="cuda",
        n_sample_proc=args.n_sample_proc,
        compute_hessians=True
    )
