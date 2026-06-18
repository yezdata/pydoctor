from datasets import load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizerFast
from dotenv import load_dotenv
import os

from src.utils.tokenize import tokenize_ds
from src.utils.preprocessing import passes_quality_filter
from src.utils.config_models import TokenizerConfig

load_dotenv()

MAX_SEQ_LEN = int(os.getenv("PRETRAIN_MAX_SEQ_LEN"))
HF_TOKEN = os.getenv("HF_TOKEN")
pretrain_dir = os.getenv("PRETRAIN_TOKENIZED_DS_DIR")
PRETRAIN_DIR = f"{pretrain_dir}{MAX_SEQ_LEN}"



def main():
    config = TokenizerConfig()  # type: ignore
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        f"tokenizers/{config.name}", local_files_only=True
    )
    
    
    # ds1 = load_dataset("angie-chen55/python-github-code", split="train")
    # ds2 = load_dataset("codeparrot/codeparrot-clean", split="train")
    # ds3 = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", token=HF_TOKEN, split="train")
    
    # ds1 = ds1.select_columns(["code"]).rename_column("code", "text")
    # ds2 = ds2.select_columns(["content"]).rename_column("content", "text")
    # ds3 = ds3.select_columns(["content"]).rename_column("content", "text")
    
    # ds = concatenate_datasets([ds1, ds2, ds3])
    
    # del ds1, ds2, ds3
    ds = load_dataset("codeparrot/codeparrot-clean", split="train")
    ds = ds.select_columns(["content"]).rename_column("content", "text")
    
    num_workers = min(16, os.cpu_count() or 1)
    ds = ds.filter(
        passes_quality_filter,
        num_proc=num_workers,
    )
    
   
    ds = tokenize_ds(tokenizer, ds, MAX_SEQ_LEN, tokenizer.eos_token_id, packing=True, num_workers=num_workers)
    
    split_ds = ds.train_test_split(test_size=0.01, seed=42)
     
    del ds
    
    data_path = f"{PRETRAIN_DIR}/big_data"
    os.makedirs(data_path, exist_ok=True)
    
    split_ds["train"].save_to_disk(f"{data_path}/train")
    split_ds["test"].save_to_disk(f"{data_path}/eval")
    
    
if __name__ == "__main__":
    main()