from transformers import PreTrainedTokenizerFast
import os

from src.utils.config_models import TokenizerConfig


def get_pretrain_tokenizer(
    config: TokenizerConfig, hf_token: str | None = None
) -> PreTrainedTokenizerFast:
    save_path = f"tokenizers/pretrain/{config.name}"

    if os.path.exists(save_path):
        return PreTrainedTokenizerFast.from_pretrained(save_path, local_files_only=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.name, token=hf_token)
    tokenizer.eos_token = config.eos_token

    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_finetune_tokenizer(
    config: TokenizerConfig, hf_token: str | None = None
) -> PreTrainedTokenizerFast:
    pretrain_tokenizer_path = f"tokenizers/pretrain/{config.name}"
    save_path = f"tokenizers/finetune/{config.name}"

    if os.path.exists(save_path):
        return PreTrainedTokenizerFast.from_pretrained(save_path, local_files_only=True)

    if os.path.exists(pretrain_tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            pretrain_tokenizer_path, local_files_only=True
        )
    else:
        tokenizer = get_pretrain_tokenizer(config, hf_token=hf_token)

    spec_tokens = {
        "extra_special_tokens": list(config.spec_tokens.model_dump().values())
    }
    tokenizer.add_special_tokens(spec_tokens, replace_extra_special_tokens=False)  # type: ignore

    tokenizer.eos_token = config.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_instruct_tokenizer(
    config: TokenizerConfig, hf_token: str | None = None
) -> PreTrainedTokenizerFast:
    save_path = f"tokenizers/instruct_finetune/{config.name}"

    if os.path.exists(save_path):
        return PreTrainedTokenizerFast.from_pretrained(save_path, local_files_only=True)
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.name, token=hf_token)

    tokenizer.eos_token = config.eos_token
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(save_path)
    return tokenizer
