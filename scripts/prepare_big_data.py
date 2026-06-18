from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast
from dotenv import load_dotenv
import os

from src.utils.tokenize import tokenize_ds
from src.utils.config_models import TokenizerConfig

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MAX_SEQ_LEN = 2048
PRETRAIN_DIR = f"data/tokenized_pretrain_seq{MAX_SEQ_LEN}"


def filter_empty_texts(x):
    return x["text"] is not None and len(x["text"].strip()) > 0



config = TokenizerConfig()  # type: ignore

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    f"tokenizers/{config.name}", local_files_only=True
)


ds1 = load_dataset("angie-chen55/python-github-code")
ds2 = load_dataset("codeparrot/codeparrot-clean")
ds3 = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", token=HF_TOKEN)

ds = concatenate_datasets([ds1, ds2, ds3])

ds = ds.filter(
    filter_empty_texts,
    num_proc=16,
)

print(len(ds))

ds = ds.shuffle(seed=100)
ds = ds.flatten_indices()

pretrain_ds_tokenized = tokenize_ds(tokenizer, ds, MAX_SEQ_LEN, packing=True)

split_ds = ds.train_test_split(test_size=0.01, seed=42)
 

data_path = f"{PRETRAIN_DIR}/big_data"
os.makedirs(data_path, exist_ok=True)

split_ds["train"].save_to_disk(f"{data_path}/train")
split_ds["test"].save_to_disk(f"{data_path}/eval")