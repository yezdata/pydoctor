import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--lora_model_path", type=str)
    parser.add_argument("--merged_model_path", type=str)

    args = parser.parse_args()

    merged_model_path = args.merged_model_path

    base = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, args.lora_model_path)
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", use_fast=False
    )

    tokenizer.save_pretrained(merged_model_path)
