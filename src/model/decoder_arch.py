import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_models import DecoderConfig
from src.utils.transformer_blocks import RMSNorm, SwiGLU
from src.utils.rope import RotaryPositionalEmbeddings


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
        expected_max_seq_len: int,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.rope = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_head, max_seq_len=expected_max_seq_len
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x, mask, input_pos)

        return self.head(self.final_norm(x))
