from transformers import AutoTokenizer
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


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    return tokenizer(
        [file_txt + tokenizer.eos_token for file_txt in batch["text"]],  # type: ignore
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )  # type: ignore


def tokenize_ds(tokenizer: AutoTokenizer, ds: Dataset, max_seq_len: int) -> Dataset:
    tokenized_ds = ds.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        batch_size=1000,
        num_proc=8,
        remove_columns=["text"],
    )

    final_ds = Dataset.from_generator(
        lambda: packing_generator(tokenized_ds, max_seq_len)
    )
    return final_ds
