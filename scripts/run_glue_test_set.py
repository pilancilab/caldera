import argparse
import json
import logging
import math
import os
import random
import numpy as np


import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name#, send_example_telemetry
from quantize_save_llama import load_quantized_model
from dataclasses import dataclass, field
from safetensors.torch import load_model
import glob


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_metrics = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}

@dataclass
class Arguments:
    task_name: str = field(metadata={
        "help": "The name of the glue task to train on.",
        "choices": list(task_to_keys.keys())
    })
    model_name_or_path: str = field(metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models."
    })
    finetune_save_dir: str = field(metadata={
        "help": "Directory with checkpoint (e.g., model.safetensors) to load"
    })
    base_model: str = field(metadata={
        "help": "Huggingface identifier of original model"
    })
    max_length: int = field(default=128, metadata={
        "help": (
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        )
    })
    pad_to_max_length: bool = field(default=True, metadata={
        "help": "pad all samples to `max_length`. Otherwise, dynamic padding is used."
    })
    batch_size: int = field(default=8, metadata={
        "help": "Batch size for testing."
    })
    output_path: str = field(default=None, metadata={
        "help": "Filename to store a JSON file with the result."
    })
    device: str = field(default="cuda")

def run_eval(eval_dataloader, model, is_regression, metric):
    samples_seen = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                # batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                predictions = model(**batch)
        
        references = batch["labels"]
        
        # print(predictions.logits)
        if predictions.logits.isnan().any():
            print("WARNING NaN OUTPUT LOGITS")
        predictions = predictions.logits.argmax(dim=-1) if not is_regression else predictions.logits.squeeze()
        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(eval_dataloader) - 1:
            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
            references = references[: len(eval_dataloader.dataset) - samples_seen]
        else:
            samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    return metric.compute()

def main(args: Arguments):
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.base_model,
            trust_remote_code=args.trust_remote_code,
        )
        
    # Model and tokenizer
    model = load_quantized_model(
        args.model_name_or_path, args.base_model, 
        args.device, sequence_classification=True
    )
    model.eval()
    for safetensor_file in glob.glob(args.finetune_save_dir + "/model*.safetensors"):
        load_model(model, safetensor_file, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=not args.use_slow_tokenizer,
    )        

    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.truncation_side = "left"

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(args.device)
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            print(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_datasets["test"]
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    test_metric = run_eval(test_dataloader, model, is_regression, metric)
    print("Test accuracy: ", test_metric)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path + ".json", "w") as f:
                json.dump({"test_accuracy": test_metric,
                           "task": args.task_name}, f, indent=2)