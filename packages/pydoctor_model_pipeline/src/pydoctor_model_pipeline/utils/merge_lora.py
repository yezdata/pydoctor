import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main() -> None:
    merged_model_path = "models/smollm2_1_7b_instruct_merged"

    base = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype=torch.float32
    )
    model = PeftModel.from_pretrained(
        base, "models/smollm2_1_7b_instruct/finetune_instruct/epoch_3"
    )
    model = model.merge_and_unload()
    model.save_pretrained(merged_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct", use_fast=False
    )

    tokenizer.save_pretrained(merged_model_path)
