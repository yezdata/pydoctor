import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding
from safetensors.torch import load_file
from datasets import load_from_disk
from tqdm import tqdm
import math

from pydoctor_model_pipeline.utils.tokenizer import get_finetune_tokenizer
from pydoctor_model_pipeline.utils.config_models import MainConfig, DecoderConfig
from pydoctor_model_pipeline.model.decoder_arch import DecoderModel
from pydoctor_model_pipeline.utils.save_model import save_decoder_model


SAVE_PATH = "models/v2/finetune"
BASE_MODEL_PATH = "models/v1/pretrain/epoch_1/step_280000"


class DataCollatorForCausalLMWithCustomLabels(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):

        super().__init__(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features):

        labels = [f.pop("labels") if "labels" in f else None for f in features]

        batch = super().__call__(features)

        if labels[0] is not None:
            max_label_length = batch["input_ids"].shape[1]
            padded_labels = []
            for l in labels:
                l_list = l.tolist() if isinstance(l, torch.Tensor) else list(l)
                remainder = max_label_length - len(l_list)

                padded_labels.append(l_list + [-100] * remainder)

            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch


def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = labels[..., 1:].contiguous()

    B, S, V = shift_logits.shape
    return criterion(shift_logits.view(B * S, V), shift_targets.view(B * S))


def evaluate(
    criterion: nn.CrossEntropyLoss,
    model: DecoderModel,
    dataloader: torch.utils.data.DataLoader,
    max_eval_steps: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    steps_run = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_eval_steps is not None and steps_run >= max_eval_steps:
                break

            logits = model(input_ids=batch["input_ids"])
            loss = compute_loss(criterion, logits, batch["labels"])

            total_loss += loss.detach().item()
            steps_run += 1

        avg_eval_loss = total_loss / max(steps_run, 1)

        try:
            perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            perplexity = float("inf")

    return avg_eval_loss, perplexity


def main() -> None:
    set_seed(1337)
    num_workers = min(16, os.cpu_count() or 1)

    config = MainConfig.from_yaml("configs.yaml")
    tokenizer = get_finetune_tokenizer(
        config.tokenizer,
    )

    with open(f"{BASE_MODEL_PATH}/config.json", "r") as f:
        model_config = DecoderConfig.model_validate_json(f.read())

    config.decoder = model_config

    with open(f"{SAVE_PATH}/config.json", "w") as f:
        f.write(config.model_dump_json(indent=4))

    accelerator = Accelerator(
        gradient_accumulation_steps=config.finetune.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    model = DecoderModel(
        config=model_config,
        eos_token_id=tokenizer.eos_token_id,
    )

    state_dict = load_file(
        f"{BASE_MODEL_PATH}/model.safetensors", device=str(accelerator.device)
    )

    if "token_embedding.weight" not in state_dict and "head.weight" in state_dict:
        state_dict["token_embedding.weight"] = state_dict["head.weight"]
    elif "head.weight" not in state_dict and "token_embedding.weight" in state_dict:
        state_dict["head.weight"] = state_dict["token_embedding.weight"]

    model.load_state_dict(strict=True, state_dict=state_dict)

    # account for docstring spec tokens
    model.resize_token_embeddings(len(tokenizer))

    config.decoder.vocab_size = len(tokenizer)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Trainable parameters: {trainable_params:,}")

    tokenized_train_ds = load_from_disk(config.finetune.tokenized_ds_dir).with_format(
        type="torch", columns=["input_ids", "labels"]
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

    data_collator = DataCollatorForCausalLMWithCustomLabels(
        tokenizer=tokenizer, pad_to_multiple_of=8
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        fused=True,
        lr=config.finetune.lr,
        weight_decay=config.finetune.weight_decay,
    )

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

        for step, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
            with accelerator.accumulate(model):
                logits = model(input_ids=batch["input_ids"])

                loss = compute_loss(
                    criterion,
                    logits,
                    batch["labels"],
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item()
                running_steps += 1

            if step % 150 == 0:
                if running_steps != 0:
                    avg_train_loss = running_loss / running_steps
                else:
                    avg_train_loss = loss.detach().item()
                    accelerator.print(
                        f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {avg_train_loss:.4f}"
                    )
                running_loss = 0.0
                running_steps = 0

        if running_steps != 0:
            avg_train_loss = running_loss / running_steps
        else:
            avg_train_loss = loss.detach().item()
        eval_loss, perplexity = evaluate(
            criterion,
            model,
            eval_dataloader,
            max_eval_steps=None,
        )
        model.train()

        accelerator.print(
            f"FINISH EPOCH {epoch + 1} | TRAIN Loss: {avg_train_loss:.4f} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}"
        )

        save_decoder_model(
            accelerator,
            model,
            f"{SAVE_PATH}/epoch_{epoch + 1}",
            avg_train_loss,
            eval_loss,
        )
        running_loss = 0.0
        running_steps = 0
