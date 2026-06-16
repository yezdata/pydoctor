from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-Next")

eos_token = "<|endoftext|>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

tokenizer.save_pretrained("tokenizers/Qwen3-Coder-Next")
