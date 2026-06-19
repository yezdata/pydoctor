from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset, Features, Value, Sequence
from typing import Generator
import torch

import os


def get_input_positions(input_ids: torch.Tensor, eos_token_id: int) -> list[int]:
    is_eos = input_ids == eos_token_id

    S = is_eos.shape[0]
    device = is_eos.device

    global_indices = torch.arange(S, device=device)

    start_mask = torch.zeros((1,), dtype=torch.bool, device=device)
    start_mask = torch.cat([start_mask, is_eos[:-1]], dim=0)

    offset_markers = torch.where(
        start_mask, global_indices, torch.zeros_like(global_indices)
    )

    offsets, _ = torch.cummax(offset_markers, dim=0)
    input_pos = global_indices - offsets

    return input_pos.tolist()


def packing_generator(
    dataset: Dataset, max_seq_len: int, eos_token_id: int
) -> Generator[dict[str, list[int]]]:
    current_chunk = []
    current_len = 0

    for sample in dataset:
        ids = sample["input_ids"]  # type: ignore
        current_chunk.extend(ids)
        current_len += len(ids)

        while current_len >= max_seq_len:
            curr_input_ids = current_chunk[:max_seq_len]

            curr_tensor = torch.tensor(curr_input_ids, dtype=torch.long)
            input_pos = get_input_positions(curr_tensor, eos_token_id)

            yield {"input_ids": curr_input_ids, "input_pos": input_pos}
            current_chunk = current_chunk[max_seq_len:]
            current_len -= max_seq_len


def tokenize_batch(batch: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    return tokenizer(
        [file_txt + tokenizer.eos_token for file_txt in batch["text"]],
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )


def tokenize_ds(
    tokenizer: PreTrainedTokenizerFast,
    ds: Dataset,
    max_seq_len: int,
    eos_token_id: int,
    packing: bool = False,
    num_workers: int = 1,
) -> Dataset:
    tokenized_ds = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=15000,
        num_proc=num_workers,
        remove_columns=["text"],
    )

    if not packing:
        return tokenized_ds

    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int32")),
            "input_pos": Sequence(feature=Value(dtype="int16")),
        }
    )

    final_ds = Dataset.from_generator(
        lambda: packing_generator(tokenized_ds, max_seq_len, eos_token_id),
        features=features,
    )
    return final_ds
