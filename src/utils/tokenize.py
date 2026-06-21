from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset, Features, Value, Sequence
from typing import Generator
import torch
from collections import deque


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
    dataset: Dataset, max_seq_len: int, tokenizer: PreTrainedTokenizerFast
) -> Generator[dict[str, list[int]]]:
    buffer = deque()

    for sample in dataset:
        tokens = tokenize_examples(f"{sample['text']}{tokenizer.eos_token}", tokenizer)[
            "input_ids"
        ]  # type: ignore

        buffer.extend(tokens)

        while len(buffer) >= max_seq_len:
            chunk = [buffer.popleft() for _ in range(max_seq_len)]
            curr_tensor = torch.tensor(chunk, dtype=torch.long)
            yield {
                "input_ids": chunk,
                "input_pos": get_input_positions(curr_tensor, tokenizer.eos_token_id),
            }


def tokenize_examples(
    example: str, tokenizer: PreTrainedTokenizerFast
) -> BatchEncoding:
    return tokenizer(
        example,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )


def tokenize_ds(
    tokenizer: PreTrainedTokenizerFast,
    ds: Dataset,
    max_seq_len: int,
    packing: bool = False,
    num_workers: int | None = None,
    batch_size: int | None = None,
) -> Dataset:
    """num_workers and batch_size are only used when packing=False"""
    if packing:
        features = Features(
            {
                "input_ids": Sequence(feature=Value(dtype="int32")),
                "input_pos": Sequence(feature=Value(dtype="int16")),
            }
        )

        final_ds = Dataset.from_generator(
            lambda: packing_generator(ds, max_seq_len, tokenizer),
            features=features,
            keep_in_memory=True,
        )
        return final_ds
    else:
        tokenized_ds = ds.map(
            lambda batch: tokenize_examples(batch["text"], tokenizer),
            batched=True,
            batch_size=batch_size,
            num_proc=num_workers,
            remove_columns=["text"],
            keep_in_memory=True,
        )
        return tokenized_ds
