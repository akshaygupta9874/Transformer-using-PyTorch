"""Backward-compatible shim for modern Transformer model.

Canonical location: models/modern.py
Import as:
    from models.modern import build_modern_transformer, ModernTransformer
    # or
    from models import build_modern_transformer, ModernTransformer
"""

from models.modern import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    KVCache,
    ModernMultiHeadAttention,
    ModernFeedForward,
    ModernEncoderBlock,
    ModernEncoder,
    ModernDecoderBlock,
    ModernDecoder,
    ModernTransformer,
    build_modern_transformer,
)

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLU",
    "KVCache",
    "ModernMultiHeadAttention",
    "ModernFeedForward",
    "ModernEncoderBlock",
    "ModernEncoder",
    "ModernDecoderBlock",
    "ModernDecoder",
    "ModernTransformer",
    "build_modern_transformer",
]
