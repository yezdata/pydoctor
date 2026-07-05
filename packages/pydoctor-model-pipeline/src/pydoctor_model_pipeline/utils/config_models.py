from pydantic import model_validator, BaseModel
from pathlib import Path
import yaml


class SpecialTokens(BaseModel):
    docstring_placeholder_token: str
    docstring_start_token: str | None


class TokenizerConfig(BaseModel):
    name: str
    eos_token: str
    spec_tokens: SpecialTokens


class DecoderConfig(BaseModel):
    vocab_size: int = -1

    d_model: int
    n_head: int
    n_layers: int
    d_ffn: int

    dropout: float

    @model_validator(mode="after")
    def check_architecture_consistency(self):
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )
        return self


class PretrainConfig(BaseModel):
    tokenized_ds_dir: str

    max_seq_len: int

    lr: float
    lr_warmup: float

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int


class FinetuneConfig(BaseModel):
    tokenized_ds_dir: str

    lr: float
    lr_warmup: float

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int


class MainConfig(BaseModel):
    tokenizer: TokenizerConfig
    decoder: DecoderConfig | None = None
    pretrain: PretrainConfig | None = None
    finetune: FinetuneConfig

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "MainConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
