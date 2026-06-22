import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
from functools import partial
import glob
import math
import os

from src.utils.config_models import TokenizerConfig
from src.utils.config_models import DecoderConfig, PretrainConfig
from src.utils.save_load import save_decoder_model
from src.model.init_weights import init_weights_modern
from src.model.transformer_blocks import construct_block_diagonal_mask
from src.model.decoder_arch import DecoderModel
from src.utils.save_tokenizer import get_pretrain_tokenizer


def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = input_ids[..., 1:].clone()

    input_eos_mask = input_ids[..., :-1] == eos_token_id

    shift_targets = torch.where(input_eos_mask, torch.tensor(-100), shift_targets)

    B, S, V = shift_logits.shape

    return criterion(shift_logits.view(B * S, V), shift_targets.view(B * S))


def evaluate(
    criterion: nn.CrossEntropyLoss,
    model: DecoderModel,
    dataloader: torch.utils.data.DataLoader,
    eos_token_id: int,
    max_eval_steps: int = 100,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    steps_run = 0

    with torch.no_grad():
        for batch in dataloader:
            if steps_run >= max_eval_steps:
                break

            outputs = model(
                batch["input_ids"], batch["attention_mask"], batch["input_pos"]
            )
            loss = compute_loss(criterion, outputs, batch["input_ids"], eos_token_id)

            total_loss += loss.detach().item()
            steps_run += 1

        avg_eval_loss = total_loss / max(steps_run, 1)

        try:
            perplexity = math.exp(avg_eval_loss)
        except OverflowError:
            perplexity = float("inf")

    return avg_eval_loss, perplexity


# Custom collate_fn pro DataLoader
def packed_collate_fn(batch: list[dict], eos_token_id: int) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])  # (B, S)
    input_pos = torch.stack([item["input_pos"] for item in batch])  # (B, S)

    is_eos = input_ids == eos_token_id

    mask = construct_block_diagonal_mask(is_eos)  # (B, 1, S, S)

    return {"input_ids": input_ids, "attention_mask": mask, "input_pos": input_pos}


def main(save_dir: str) -> None:
    set_seed(6767)
    num_workers = min(16, os.cpu_count() or 1)

    tokenizer_config = TokenizerConfig()  # type: ignore

    tokenizer = get_pretrain_tokenizer(
        tokenizer_config,
    )

    model_config = DecoderConfig(vocab_size=len(tokenizer))  # type: ignore

    train_config = PretrainConfig()  # type: ignore
    train_config.tokenized_ds_dir = (
        f"{train_config.tokenized_ds_dir}{train_config.max_seq_len}"
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps
    )

    model = DecoderModel(
        config=model_config,
        eos_token_id=tokenizer.eos_token_id,
        expected_max_seq_len=train_config.max_seq_len,
    )

    init_fn = partial(init_weights_modern, n_layers=model_config.n_layers)
    model.apply(init_fn)

    train_search_path = os.path.join(train_config.tokenized_ds_dir, "*", "train")
    train_folders = [f for f in glob.glob(train_search_path) if os.path.isdir(f)]
    if not train_folders:
        raise FileNotFoundError("Did not find any train data dirs")

    eval_search_path = os.path.join(train_config.tokenized_ds_dir, "*", "eval")
    eval_folders = [f for f in glob.glob(eval_search_path) if os.path.isdir(f)]
    if not eval_folders:
        raise FileNotFoundError("Did not find any eval data dirs")

    tokenized_train_ds = concatenate_datasets(
        [load_from_disk(folder) for folder in train_folders]
    ).with_format("torch")
    tokenized_eval_ds = concatenate_datasets(
        [load_from_disk(folder) for folder in eval_folders]
    ).with_format("torch")

    collator = partial(packed_collate_fn, eos_token_id=tokenizer.eos_token_id)

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_train_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    accelerator.print(f"Total parameters: {total_params:,}")
    accelerator.print(f"Trainable parameters: {trainable_params:,}")

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
                outputs = model(
                    batch["input_ids"], batch["attention_mask"], batch["input_pos"]
                )
                loss = compute_loss(
                    criterion, outputs, batch["input_ids"], tokenizer.eos_token_id
                )

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                running_loss += loss.detach().item()
                running_steps += 1

            if step % 1000 == 0 and step > 0:
                avg_train_loss = running_loss / running_steps
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step} | TRAIN  Loss: {avg_train_loss:.4f}"
                )
                running_loss = 0.0
                running_steps = 0

            if step % 10000 == 0 and step > 0:
                eval_loss, perplexity = evaluate(
                    criterion,
                    model,
                    eval_dataloader,
                    tokenizer.eos_token_id,
                    max_eval_steps=100,
                )
                accelerator.print(
                    f"CHECKPOINT {step} | TRAIN Loss: {loss.item():.4f} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}"
                )

                step_path = f"{save_dir}/epoch_{epoch + 1}/step_{step}"

                save_decoder_model(
                    accelerator,
                    model,
                    step_path,
                    model_config,
                    train_config,
                    loss.item(),
                    eval_loss,
                )

                model.train()

        epoch_path = f"{save_dir}/epoch_{epoch + 1}"
        save_decoder_model(accelerator, model, epoch_path, model_config, train_config)

        eval_loss, perplexity = evaluate(
            criterion,
            model,
            eval_dataloader,
            tokenizer.eos_token_id,
            max_eval_steps=len(eval_dataloader),
        )
        accelerator.print(
            f"FINISH EPOCH {epoch + 1} | Eval Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}"
        )


if __name__ == "__main__":
    SAVE_DIR = "models/v0/pretrain"

    main(SAVE_DIR)
