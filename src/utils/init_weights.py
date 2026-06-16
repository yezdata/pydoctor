import torch.nn as nn
from src.utils.transformer_blocks import RMSNorm


def init_weights_modern(module: nn.Module, n_layers: int):
    """Modern initialization for Transformers (RMSNorm, GeGLU / SwiGLU)."""

    if isinstance(module, nn.Linear):
        # scale down the init for residual connections
        if getattr(module, "_is_residual", False):
            std = 0.02 / ((2 * n_layers) ** 0.5)
        else:
            std = 0.02

        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)

    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)


def init_weights(module: nn.Module):
    """Standard initialization for Transformers (LayerNorm, Linear)."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
