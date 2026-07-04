import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import glob
import math

from tokenizer import get_instruct_tokenizer
from config_models import MainConfig, FinetuneConfig


SAVE_PATH = "/kaggle/working/models/v1/finetune_instruct"
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN", "")

CACHE_DIR = "/kaggle/working/datasets_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    assistant_sequence: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = input_ids[..., 1:].contiguous()

    loss_mask = torch.zeros_like(shift_targets, dtype=torch.bool)

    seq_len = assistant_sequence.shape[0]
    batch_size = input_ids.shape[0]

    for i in range(batch_size):
        sample_ids = input_ids[i]

        match_idx = -1
        for idx in range(len(sample_ids) - seq_len + 1):
            if torch.equal(sample_ids[idx : idx + seq_len], assistant_sequence):
                match_idx = idx
                break

        if match_idx != -1:
            start_calculating_idx = match_idx + seq_len - 1
            loss_mask[i, start_calculating_idx:] = True

    is_not_pad = shift_targets != pad_token_id
    loss_mask = loss_mask & is_not_pad

    shift_targets = torch.where(
        loss_mask, shift_targets, torch.tensor(-100, device=shift_targets.device)
    )

    B, S, V = shift_logits.shape
    return criterion(shift_logits.view(B * S, V), shift_targets.view(B * S))


def evaluate(
    criterion: nn.CrossEntropyLoss,
    model: AutoModelForCausalLM,
    dataloader: torch.utils.data.DataLoader,
    start_token_ids: torch.Tensor,
    pad_token_id: int,
    max_eval_steps: int = 100,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    steps_run = 0

    with torch.no_grad():
        for batch in dataloader:
            if steps_run >= max_eval_steps:
                break

            outputs = model(batch["input_ids"]).logits
            loss = compute_loss(
                criterion, outputs, batch["input_ids"], start_token_ids, pad_token_id
            )

            total_loss += loss.detach().item()
            steps_run += 1

        avg_eval_loss = total_loss / max(steps_run, 1)

        try:
            perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            perplexity = float("inf")

    return avg_eval_loss, perplexity


def main(
    save_path: str,
    model_name: str,
) -> None:
    set_seed(1337)
    num_workers = 2

    config = MainConfig.from_yaml("configs_kaggle.yaml")

    tokenizer = get_instruct_tokenizer(config.tokenizer)
    tokenizer.padding_side = "right"

    with open(f"{save_path}/config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.finetune.gradient_accumulation_steps,
        mixed_precision="no",
    )

    loss_start_token_ids = torch.tensor(
        tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
        device=accelerator.device,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )

    tokenized_train_ds = load_from_disk(config.finetune.tokenized_ds_dir).with_format(
        "torch"
    )

    # REMOVE LONG SEQUENCES
    # lengths = [len(x) for x in tokenized_train_ds["input_ids"]]
    # max_len_95 = int(torch.quantile(torch.tensor(lengths, dtype=torch.float32), 0.95).item())

    max_len_95 = 1024

    tokenized_train_ds = tokenized_train_ds.filter(
        lambda x: len(x["input_ids"]) <= max_len_95,
        cache_file_name=os.path.join(CACHE_DIR, "filtered_train.cache"),
    )

    split_ds = tokenized_train_ds.train_test_split(
        test_size=0.01,
        seed=6767,
        train_indices_cache_file_name=os.path.join(CACHE_DIR, "train_indices.cache"),
        test_indices_cache_file_name=os.path.join(CACHE_DIR, "test_indices.cache"),
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
    )

    train_dataloader = torch.utils.data.DataLoader(
        split_ds["train"],  # type: ignore
        batch_size=config.finetune.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        split_ds["test"],  # type: ignore
        batch_size=config.finetune.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    model = prepare_model_for_kbit_training(model)
    # model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=config.finetune.lr,
        weight_decay=config.finetune.weight_decay,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Trainable parameters: {trainable_params:,}")

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    total_steps = (
        len(train_dataloader) * config.finetune.num_epochs
    ) // config.finetune.gradient_accumulation_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.finetune.lr_warmup),
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # TRAIN LOOP
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    running_loss = 0.0
    running_steps = 0

    for epoch in range(config.finetune.num_epochs):
        model.train()

        train_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in train_bar:
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"]).logits

                loss = compute_loss(
                    criterion,
                    outputs,
                    batch["input_ids"],
                    loss_start_token_ids,
                    tokenizer.pad_token_id,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    gathered_loss = accelerator.gather(loss.detach())
                    running_loss += gathered_loss.mean().item()
                    running_steps += 1

                if step % 150 == 0:
                    if running_steps != 0:
                        avg_train_loss = running_loss / running_steps
                    else:
                        avg_train_loss = accelerator.gather(loss.detach()).mean().item()
                    accelerator.print(
                        f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {avg_train_loss:.4f}"
                    )
                    running_loss = 0.0
                    running_steps = 0

        if running_steps != 0:
            avg_train_loss = running_loss / running_steps
        else:
            avg_train_loss = accelerator.gather(loss.detach()).mean().item()
        eval_loss, perplexity = evaluate(
            criterion,
            model,
            eval_dataloader,
            loss_start_token_ids,
            pad_token_id=tokenizer.pad_token_id,
            max_eval_steps=100,
        )
        model.train()

        accelerator.print(
            f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {avg_train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}"
        )

        running_loss = 0.0
        running_steps = 0

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            os.makedirs(f"{save_path}/epoch_{epoch + 1}", exist_ok=True)

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                f"{save_path}/epoch_{epoch + 1}", safe_serialization=True
            )

            with open(f"{save_path}/epoch_{epoch + 1}/train_state.json", "w") as f:
                json.dump(
                    {"train_loss": avg_train_loss, "eval_loss": eval_loss}, f, indent=4
                )


if __name__ == "__main__":
    main(
        save_path=SAVE_PATH,
        model_name=MODEL_NAME,
    )
