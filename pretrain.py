import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_from_disk
from tqdm import tqdm
import math

from src.utils.config_models import DecoderConfig, PretrainConfig
from src.utils.save_load import save_decoder_model
from src.utils.init_weights import init_weights_modern
from src.utils.transformer_blocks import construct_block_diagonal_mask
from src.model.decoder_arch import DecoderModel


def compute_loss(
    criterion: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = input_ids[..., 1:].contiguous()

    input_eos_mask = input_ids[..., :-1] == eos_token_id

    shift_targets[input_eos_mask] = -100

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

            input_pos = get_input_positions(batch["input_ids"], eos_token_id)
            outputs = model(batch["input_ids"], batch["attention_mask"], input_pos)
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
    B, S = input_ids.shape

    is_eos = input_ids == eos_token_id

    mask = construct_block_diagonal_mask(is_eos)  # (B, 1, S, S)

    return {"input_ids": input_ids, "attention_mask": mask}


def get_input_positions(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    is_eos = input_ids == eos_token_id

    B, S = is_eos.shape
    device = is_eos.device

    global_indices = torch.arange(S, device=device).unsqueeze(0).expand(B, S)

    start_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
    start_mask = torch.cat([start_mask, is_eos[:, :-1]], dim=-1)

    offset_markers = torch.where(
        start_mask, global_indices, torch.zeros_like(global_indices)
    )

    offsets, _ = torch.cummax(offset_markers, dim=-1)

    return global_indices - offsets


def main(save_dir: str) -> None:
    set_seed(6767)

    tokenizer = AutoTokenizer.from_pretrained(
        "tokenizers/Qwen3-Coder-Next", local_files_only=True
    )

    model_config = DecoderConfig(vocab_size=len(tokenizer))  # type: ignore

    train_config = PretrainConfig()  # type: ignore

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps
    )

    model = DecoderModel(
        config=model_config,
        eos_token_id=tokenizer.eos_token_id,
        expected_max_seq_len=train_config.max_seq_len,
    )

    model.apply(lambda m: init_weights_modern(m, model_config.n_layers))

    tokenized_train_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/train")
    tokenized_eval_ds = load_from_disk(f"{train_config.tokenized_ds_dir}/eval")

    train_dataloader = torch.utils.data.DataLoader(
        tokenized_train_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=lambda batch: packed_collate_fn(batch, tokenizer.eos_token_id),
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval_ds,  # type: ignore
        batch_size=train_config.batch_size,
        collate_fn=lambda batch: packed_collate_fn(batch, tokenizer.eos_token_id),
        shuffle=False,
        num_workers=8,
        pin_memory=True,
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
                input_pos = get_input_positions(
                    batch["input_ids"], tokenizer.eos_token_id
                )

                outputs = model(batch["input_ids"], batch["attention_mask"], input_pos)
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
