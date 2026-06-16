import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.wo = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.wi(x).chunk(2, dim=-1)
        return self.wo(x1 * F.silu(x2))


class GeGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.wi = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.wo = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.wi(x).chunk(2, dim=-1)
        return self.wo(x1 * F.gelu(x2))


def construct_block_diagonal_mask(
    is_eos: torch.Tensor,
) -> torch.Tensor:
    B, S = is_eos.shape

    block_ids = torch.cumsum(is_eos, dim=-1)

    zeros = torch.zeros((B, 1), dtype=block_ids.dtype)
    block_ids = torch.cat([zeros, block_ids[:, :-1]], dim=-1)

    block_mask = block_ids.unsqueeze(-1) == block_ids.unsqueeze(-2)

    causal_mask = torch.ones(S, S, dtype=torch.bool).tril()

    return (causal_mask & block_mask).unsqueeze(1)
