from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset
from typing import Generator


def packing_generator(
    dataset: Dataset, max_seq_len: int
) -> Generator[dict[str, list[int]]]:
    buffer = []
    for sample in dataset:
        buffer.extend(sample["input_ids"])  # type: ignore
        while len(buffer) >= max_seq_len:
            curr_input_ids = buffer[:max_seq_len]
            yield {"input_ids": curr_input_ids}
            buffer = buffer[max_seq_len:]


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
    packing: bool = False,
) -> Dataset:
    tokenized_ds = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=["text"],
    )

    if not packing:
        return tokenized_ds

    final_ds = Dataset.from_generator(
        lambda: packing_generator(tokenized_ds, max_seq_len)
    )
    return final_ds
