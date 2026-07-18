import torch
import torch.nn as nn
import torch.nn.functional as F

from pydoctor_model_pipeline.model.init_weights import init_weights_modern
from pydoctor_model_pipeline.utils.config_models import DecoderConfig
from pydoctor_model_pipeline.model.transformer_blocks import RMSNorm, SwiGLU
from pydoctor_model_pipeline.model.rope import RotaryPositionalEmbeddings


class DecoderLayer(nn.Module):
    """Custom Pre-LN Transformer Decoder Layer with RoPE and FlashAttention."""

    def __init__(self, config: DecoderConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.rope = rope

        # Attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.ffn = SwiGLU(config.d_model, config.d_ffn)

        self.dropout = nn.Dropout(config.dropout)

        # mark for initialization
        self.out_proj._is_residual = True
        self.ffn.wo._is_residual = True

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # MULTI-HEAD ATTENTION
        residual = x
        nx = self.norm1(x)
        B, S, _ = nx.shape

        # Projections -> (B, S, N_heads, D_head)
        q = self.q_proj(nx).view(B, S, self.n_head, self.d_head)
        k = self.k_proj(nx).view(B, S, self.n_head, self.d_head)
        v = self.v_proj(nx).view(B, S, self.n_head, self.d_head)

        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True if mask is None else False,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.dropout.training else 0.0,
        )

        # Join heads -> (B, S, D_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        x = residual + self.dropout(self.out_proj(attn_out))

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderModel(nn.Module):
    """Decoder-only Transformer model architecture"""

    def __init__(
        self,
        config: DecoderConfig,
        eos_token_id: int,
        expected_max_seq_len: int = 4096,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        self.rope = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_head, max_seq_len=expected_max_seq_len
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        In-place resize of token embeddings and head weights for weight tying
        ! new_vocab_size > old_vocab_size

        """
        old_embeddings = self.token_embedding

        if new_vocab_size == old_embeddings.num_embeddings:
            return

        assert new_vocab_size > old_embeddings.num_embeddings

        new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
        init_weights_modern(new_embeddings)

        with torch.no_grad():
            new_embeddings.weight.data[: old_embeddings.num_embeddings] = (
                old_embeddings.weight.data
            )
        self.token_embedding = new_embeddings

        new_head = nn.Linear(old_embeddings.embedding_dim, new_vocab_size, bias=False)
        new_head.weight = self.token_embedding.weight
        self.head = new_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        input_ids = self.token_embedding(input_ids)

        for layer in self.layers:
            input_ids = layer(input_ids, attention_mask, input_pos)

        x_norm = self.final_norm(input_ids)

        return self.head(x_norm)
