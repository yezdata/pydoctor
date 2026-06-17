from transformers import PreTrainedTokenizerFast

from src.utils.config_models import TokenizerConfig

config = TokenizerConfig()  # type: ignore

tokenizer = PreTrainedTokenizerFast.from_pretrained(config.name)

eos_token = config.eos_token
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

tokenizer.save_pretrained(f"tokenizers/{config.name}")
