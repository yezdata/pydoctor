import os
import json
import torch
from transformers import AutoModelForCausalLM
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import glob
import math
from dotenv import load_dotenv

from src.utils.tokenizer import get_finetune_tokenizer
from src.utils.config_models import TokenizerConfig
from src.utils.config_models import FinetuneConfig
from src.model.decoder_arch import DecoderModel


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")


def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    start_token_ids: list[int],
    pad_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = input_ids[..., 1:]

    start_mask = torch.isin(
        input_ids[..., :-1], torch.tensor(start_token_ids, device=input_ids.device)
    )

    is_docstring = torch.cumsum(start_mask, dim=-1) > 0

    is_not_pad = shift_targets != pad_token_id

    loss_mask = is_docstring & is_not_pad

    shift_targets = torch.where(loss_mask, shift_targets, -100)

    B, S, V = shift_logits.shape
    return criterion(shift_logits.view(B * S, V), shift_targets.view(B * S))


def evaluate(
    criterion: nn.CrossEntropyLoss,
    model: DecoderModel,
    dataloader: torch.utils.data.DataLoader,
    start_token_ids: list[int],
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
    num_workers = min(16, os.cpu_count() or 1)

    tokenizer_config = TokenizerConfig(name=model_name)  # type: ignore
    tokenizer = get_finetune_tokenizer(tokenizer_config)
    tokenizer.padding_side = "right"
    spec_tokens = tokenizer.encode(
        list(tokenizer_config.spec_tokens.model_dump().values())
    )

    train_config = FinetuneConfig(tokenized_ds_dir="TEST-gemma/data/tokenized_finetune")  # type:ignore

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )

    # account for docstring spec tokens
    model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Trainable parameters: {trainable_params:,}")

    tokenized_train_ds = load_from_disk(train_config.tokenized_ds_dir).with_format(
        "torch"
    )

    # REMOVE LONG SEQUENCES
    lengths = [len(x) for x in tokenized_train_ds["input_ids"]]
    max_len_95 = int(
        torch.quantile(torch.tensor(lengths, dtype=torch.float32), 0.95).item()
    )

    tokenized_train_ds = tokenized_train_ds.filter(
        lambda x: len(x["input_ids"]) <= max_len_95
    )

    split_ds = tokenized_train_ds.train_test_split(test_size=0.01, seed=6767)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
    )

    train_dataloader = torch.utils.data.DataLoader(
        split_ds["train"],  # type: ignore
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        split_ds["test"],  # type: ignore
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=data_collator,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        fused=True,
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    total_steps = (
        len(train_dataloader) * train_config.num_epochs
    ) // train_config.gradient_accumulation_steps

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * train_config.lr_warmup),
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # TRAIN LOOP
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    running_loss = 0.0
    running_steps = 0

    for epoch in range(train_config.num_epochs):
        model.train()

        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with accelerator.accumulate(model):
                outputs = model(batch["input_ids"]).logits

                loss = compute_loss(
                    criterion,
                    outputs,
                    batch["input_ids"],
                    spec_tokens,
                    tokenizer.pad_token_id,
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item()
                running_steps += 1

            if step % 150 == 0:
                avg_train_loss = running_loss / running_steps
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {avg_train_loss:.4f}"
                )
                running_loss = 0.0
                running_steps = 0

        avg_train_loss = running_loss / running_steps
        eval_loss, perplexity = evaluate(
            criterion,
            model,
            eval_dataloader,
            spec_tokens,
            pad_token_id=tokenizer.pad_token_id,
            max_eval_steps=100,
        )
        accelerator.print(
            f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {avg_train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}"
        )

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            f"{save_path}/epoch_{epoch + 1}", safe_serialization=True
        )

        with open(f"{save_path}/epoch_{epoch + 1}/train_state.json", "w") as f:
            json.dump(
                {"train_loss": avg_train_loss, "eval_loss": eval_loss}, f, indent=4
            )

        with open(f"{save_path}/epoch_{epoch + 1}/train_config.json", "w") as f:
            f.write(train_config.model_dump_json(indent=4))

        running_loss = 0.0
        running_steps = 0

        model.train()


if __name__ == "__main__":
    SAVE_PATH = "models/v1/finetune"
    MODEL_NAME = "google/gemma-3-270m"

    main(
        save_path=SAVE_PATH,
        model_name=MODEL_NAME,
    )
