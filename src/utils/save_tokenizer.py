from transformers import PreTrainedTokenizerFast
import os

from src.utils.config_models import TokenizerConfig


def get_pretrain_tokenizer(
    config: TokenizerConfig,
) -> PreTrainedTokenizerFast:
    save_path = f"tokenizers/pretrain/{config.name}"

    if os.path.exists(save_path):
        return PreTrainedTokenizerFast.from_pretrained(save_path, local_files_only=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.name)

    eos_token = config.eos_token
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    tokenizer.save_pretrained(save_path)
    return tokenizer


def get_finetune_tokenizer(config: TokenizerConfig):
    pretrain_tokenizer_path = f"tokenizers/pretrain/{config.name}"
    save_path = f"tokenizers/finetune/{config.name}"

    if os.path.exists(save_path):
        return PreTrainedTokenizerFast.from_pretrained(
            f"{save_path}{config.name}", local_files_only=True
        )

    if not os.path.exists(pretrain_tokenizer_path):
        raise FileNotFoundError(
            f"Pretrain tokenizer not found at {pretrain_tokenizer_path}{config.name}"
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        pretrain_tokenizer_path, local_files_only=True
    )

    spec_tokens = {"extra_special_tokens": [config.sod_token, config.eod_token]}
    tokenizer.add_special_tokens(spec_tokens)
    tokenizer.add_special_tokens(spec_tokens, replace_existing_added_tokens=False)

    tokenizer.save_pretrained(save_path)
    return tokenizer
