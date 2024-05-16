# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import default_data_collator, DataCollatorWithPadding, get_scheduler

from datasets import load_dataset, Dataset
from quantize_save_llama import load_quantized_model
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import datasets

import math
import evaluate
from tqdm import tqdm


logger = get_logger(__name__)


@dataclass
class ModelArguments:
    base_model: str = field()
    model_name_or_path: str = field()
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    ignore_rht_finetuning: bool = field(default=False, metadata={
        "help": "If RHT finetuning has been performed, do *not* use the RHT-finetuned model."
    })


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_seq_length: int = field(default=128, metadata={
        "help": ("The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.")
    })
    with_tracking: bool = field(default=False, metadata={
        "help": "Whether to report eval accuracies to, e.g., tensorboard."
    })
    num_warmup_steps: int = field(default=0)

def run_eval(eval_dataloader, model, accelerator, metric):
    samples_seen = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                # batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                predictions = model(**batch)
        
        references = batch["labels"]
        predictions, references = accelerator.gather_for_metrics((predictions, references))
        
        # print(predictions.logits)
        if predictions.logits.isnan().any():
            print("WARNING NaN OUTPUT LOGITS")
        predictions = predictions.logits.argmax(dim=-1)
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


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    transformers.utils.send_example_telemetry("run_clm_no_trainer", training_args)

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        f"ep_{int(training_args.num_train_epochs)}_lr_{training_args.learning_rate}_seed_{training_args.seed}"
    )

    accelerator_log_kwargs = {}

    if training_args.with_tracking:
        accelerator_log_kwargs["log_with"] = training_args.report_to
        accelerator_log_kwargs["project_dir"] = training_args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        **accelerator_log_kwargs)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if accelerator.is_main_process:
        if training_args.seed is not None:
            set_seed(training_args.seed)

        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
        # writer = SummaryWriter(args.output_dir)
    accelerator.wait_for_everyone()

    model = load_quantized_model(
        model_args.model_name_or_path, model_args.base_model,
        accelerator.device, sequence_classification=True,
        include_rht_finetuning=not model_args.ignore_rht_finetuning
    ).to(torch.bfloat16)

    for name, param in model.named_parameters():
        if 'SU' in name or 'SV' in name:
            param.requires_grad = False
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.base_model,
        token=model_args.token,
        use_fast=True,
    )

    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.truncation_side = "left"

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    train_data = load_dataset("winogrande", "winogrande_xl", split="train").to_pandas()
    eval_data = load_dataset("winogrande", "winogrande_xl", split="validation").to_pandas()

    def preprocess(data):
        data["text1"] = data.apply(lambda x: x["sentence"].replace("_", x["option1"]), axis=1)
        data["text2"] = data.apply(lambda x: x["sentence"].replace("_", x["option2"]), axis=1)
        data["label"] = data.apply(lambda x: int(x["answer"]) - 1, axis=1)
        return Dataset.from_pandas(data)

    def tokenize(sample):
        model_inps =  tokenizer(sample["text1"], sample["text2"], padding="max_length",
                                truncation=True, max_length=training_args.max_seq_length)
        model_inps["labels"] = sample["label"]
        return model_inps

    with accelerator.main_process_first():
        train_data = preprocess(train_data)
        eval_data = preprocess(eval_data)
        tokenized_train_data = train_data.map(tokenize, batched=True, desc="Tokenizing training data",
                                              remove_columns=train_data.column_names)
        tokenized_eval_data = eval_data.map(tokenize, batched=True, desc="Tokenizing eval data",
                                            remove_columns=train_data.column_names)

    print(tokenized_train_data)
    # print(tokenized_train_data["labels"])

    train_dataloader = DataLoader(
        tokenized_train_data, collate_fn=default_data_collator,
        batch_size=training_args.per_device_train_batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        tokenized_eval_data, collate_fn=default_data_collator,
        batch_size=training_args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    save_steps = training_args.save_steps
    if save_steps is not None and isinstance(save_steps, str) and save_steps.isdigit():
        save_steps = int(save_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if training_args.with_tracking:
        experiment_config = vars(training_args)
        # TensorBoard cannot log Enums, need the raw value
        # experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("winogrande_no_trainer", {})

    metric = evaluate.load("accuracy")
    max_train_steps = int(max_train_steps)

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(tokenized_train_data)}")
    print(f"  Num Epochs = {training_args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = starting_epoch * len(train_dataloader) + resume_step
            progress_bar.update(completed_steps)

    performace_dict = {}
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        if training_args.with_tracking:
            total_loss = 0
        # if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
        #     # We skip the first `n` batches in the dataloader when resuming from a checkpoint
        #     train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        
        for step, batch in enumerate(train_dataloader):
            # print(batch["attention_mask"])
            # # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    # completed_steps += 1
                    continue

            # print(batch["labels"], batch["labels"].shape)
            with accelerator.accumulate(model):
                # print(batch)
                # return
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if training_args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                if completed_steps % 50:
                    accelerator.print(f"Epoch: {epoch} | Step: {completed_steps} | Loss: {loss}")
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
            if isinstance(save_steps, int):
                if completed_steps % save_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps % 200 == 0:
                model.eval()
                eval_metric = run_eval(eval_dataloader, model, accelerator, metric)

                # for k, v in eval_metric.items():
                #     writer.add_scalar(f"eval/{args.output_dir}/{k}", v, global_step=completed_steps)
                logger.info(
                    f"seed {training_args.seed} learning rate {training_args.learning_rate} "
                    + f"epoch {epoch}: {eval_metric}")
                performace_dict[completed_steps]=eval_metric["accuracy"]

                if training_args.with_tracking and total_loss != 0:
                    accelerator.log(
                        {
                            "accuracy": eval_metric,
                            "train_loss": total_loss.item() / len(train_dataloader),
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

            if completed_steps >= max_train_steps:
                break

            if completed_steps % 500 == 0 and step % training_args.gradient_accumulation_steps == 0 :
                logger.info(f"The current loss is {loss}")

                
        if training_args.save_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)

            accelerator.save_state(output_dir)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                from safetensors.torch import save_file
                save_file(accelerator.get_state_dict(model), output_dir + "/model.safetensors")
                print("Saved checkpoint in ", output_dir + "/model.safetensors")
            accelerator.wait_for_everyone()     

        model.eval()
        eval_metric = run_eval(eval_dataloader, model, accelerator, metric)

        # for k, v in eval_metric.items():
        #     writer.add_scalar(f"eval/{args.output_dir}/{k}", v, global_step=epoch)
        logger.info(f"{training_args.output_dir} | epoch {epoch}: {eval_metric}")
        if training_args.with_tracking and total_loss != 0:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        performace_dict[epoch] = eval_metric["accuracy"]

        # torch.save(model.state_dict(), args.output_dir + f"/{completed_steps}.bin")

    model.eval()
    eval_metric = run_eval(eval_dataloader, model, accelerator, metric)

    # for k, v in eval_metric.items():
    #     writer.add_scalar(f"eval/{args.output_dir}/{k}", v, global_step=completed_steps)
    print(f"{training_args.output_dir} | step {completed_steps}: {eval_metric}")
    if not eval:
        best_performance = max(performace_dict.values())
        max_keys = [k for k, v in performace_dict.items() if
                    v == best_performance]  # getting all keys containing the `maximum`
        print(f"seed {training_args.seed} learning rate {args.learning_rate} "
              + f"The best performance is at {max_keys[0]} with {best_performance}")


if __name__ == "__main__":
    train()
