from pydantic import model_validator, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class SpecialTokens(BaseModel):
    sofd_token: str
    socd_token: str
    somd_token: str


class TokenizerConfig(BaseSettings):
    name: str
    spec_tokens: SpecialTokens
    eos_token: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="TOKENIZER_",
        env_nested_delimiter="__",
    )


class DecoderConfig(BaseSettings):
    vocab_size: int

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

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="DECODER_", extra="ignore"
    )


class PretrainConfig(BaseSettings):
    tokenized_ds_dir: str

    max_seq_len: int

    lr: float
    lr_warmup: float

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="PRETRAIN_", extra="ignore"
    )


class FinetuneConfig(BaseSettings):
    tokenized_ds_dir: str

    lr: float
    lr_warmup: float

    weight_decay: float

    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="FINETUNE_", extra="ignore"
    )
