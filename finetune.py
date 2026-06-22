# TODO: add sod / eod tokens
# TODO: resize embedding matrix
# TODO: save tokenizer with new tokens to models/finetune/
from src.utils.save_tokenizer import get_finetune_tokenizer
from src.utils.config_models import TokenizerConfig


tokenizer_config = TokenizerConfig()  # type: ignore
tokenizer = get_finetune_tokenizer(tokenizer_config)
