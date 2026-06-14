from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next")
    max_seq_len = 512

    ds = load_dataset(
        "json",
        data_files={
            "train": "data/goemotions/train_data.jsonl",
            "eval": "data/goemotions/val_data.jsonl",
            "test": "data/goemotions/test_data.jsonl",
        },
    )

    tokenized_ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            # truncation=True,
            # max_length=max_seq_len,
        ),
        batched=True,
        num_proc=16,
        remove_columns=["text"],
    )

    tokenized_ds.set_format("torch")

    tokenized_ds.save_to_disk(f"data/goemotions_v2_seq{max_seq_len}")


if __name__ == "__main__":
    main()
