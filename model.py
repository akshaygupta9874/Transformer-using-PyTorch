"""Backward-compatible shim for vanilla Transformer model.

Canonical location: models/basic.py
Import as:
    from models.basic import build_transformer, Transformer
    # or
    from models import build_transformer, Transformer
"""

from models.basic import (
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

__all__ = [
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
]
