"""Models package for English-to-Hindi Neural Machine Translation.

Provides both:
1. Basic / Vanilla Transformer (Vaswani et al., 2017)
2. Modern SOTA Transformer (RMSNorm, RoPE, SwiGLU, KV Caching, Gradient Checkpointing)
"""

from .basic import (
    LayerNormalization,
    FeedForwardBlock,
    InputEmbedding,
    PositionalEncoding,
    ResidualConnection,
    MultiHeadAttentionBlock,
    EncoderBlock,
    Encoder,
    DecoderBlock,
    Decoder,
    ProjectionLayer,
    Transformer,
    build_transformer,
)

from .modern import (
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
    # Basic / Vanilla
    "LayerNormalization",
    "FeedForwardBlock",
    "InputEmbedding",
    "PositionalEncoding",
    "ResidualConnection",
    "MultiHeadAttentionBlock",
    "EncoderBlock",
    "Encoder",
    "DecoderBlock",
    "Decoder",
    "ProjectionLayer",
    "Transformer",
    "build_transformer",
    # Modern
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
