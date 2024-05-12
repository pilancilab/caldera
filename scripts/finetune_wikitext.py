from argparse_dataclass import ArgumentParser
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    default_data_collator,
    TrainingArguments
)
from transformers.testing_utils import CaptureLogger
import math
import logging
from itertools import chain
import evaluate
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers.utils.versions import require_version
from quantize_save_llama import load_quantized_model
from accelerate import Accelerator
import os


@dataclass
class Arguments:
    model_save_path: str = field(metadata={
        "help": ("Path in which the quantized model was saved via "
                 "quantize_save_llama.py")
    })
    base_model: str = field(metadata={
        "help": ("Path of the original model, as either a local or a "
                 "Huggingface path")
    })
    ignore_rht_finetuning: bool = field(default=False, metadata={
        "help": "If RHT finetuning has been performed, do *not* use the RHT-finetuned model."
    })

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main(
    model_save_path: str,
    base_model: str,
    output_dir: str,
    ignore_rht_finetuning: bool,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
):
    os.makedirs(output_dir, exist_ok=True)
    accelerator = Accelerator()
    model = load_quantized_model(model_save_path, base_model, accelerator.device,
                                 include_rht_finetuning=not ignore_rht_finetuning)
    for name, param in model.named_parameters():
        if 'SU' in name or 'SV' in name:
            param.requires_grad = False
    
    logger = logging.getLogger(__name__)

    config = AutoConfig.from_pretrained(base_model)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        streaming=data_args.streaming,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True)
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long "
                "input will be chunked into smaller bits before being passed "
                "to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                "The tokenizer picked seems to have a very large "
                f"`model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. "
                "You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger "
                "than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using "
                f"block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from 
    # our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k]))
                                 for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size
        # we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop,
        # you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] 
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
    eval_dataset = lm_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    train_dataset = lm_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra
            # tensors, like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has
        # been calculated by preprocess_logits_for_metrics but we need to
        # shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # train?
    trainer.train()
    trainer.save_model()

    metrics = trainer.evaluate()

    max_eval_samples = len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    print(f"perplexity: {perplexity}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser([
        Arguments, DataTrainingArguments, TrainingArguments])
    args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(
        args.model_save_path,
        args.base_model,
        training_args.output_dir,
        ignore_rht_finetuning=args.ignore_rht_finetuning,
        data_args=data_args,
        training_args=training_args
    )
