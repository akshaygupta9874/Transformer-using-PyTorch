"""Modern SOTA Transformer Architecture: Pre-LN, RMSNorm, RoPE, SwiGLU, and KV Caching (Part A).

Implements modern best practices (as seen in LLaMA, Gemma, Mistral, and modern NMT):
- RMSNorm: 10-20% faster than LayerNorm with superior BF16 numerical stability
- RoPE (Rotary Position Embeddings): Relative position encoding with natural distance decay
- SwiGLU: Gated feed-forward network for improved expressiveness per parameter
- Incremental KV Caching: Cuts autoregressive decoding from O(L^2) to O(L)
- Gradient Checkpointing: Trades minimal compute to fit larger models/batches in VRAM
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    
    Faster than LayerNorm by omitting mean subtraction; stabilizes deep networks.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE, Su et al., 2021).
    
    Applies position-dependent rotation to Q and K representations.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency inverse table
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos and sin caches
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (max_seq_len, dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the vector."""
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors sliced for the current sequence length and offset."""
        seq_len = x.size(-2)
        cos = self.cos_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[offset : offset + seq_len, :].unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        return cos.to(x.dtype), sin.to(x.dtype)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to Q and K tensors."""
        cos, sin = self.forward(q, offset=offset)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """Swish Gated Linear Unit Feed-Forward Network (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


class KVCache:
    """Manages Key-Value states for linear O(L) autoregressive inference."""

    def __init__(self):
        self.self_keys: Dict[int, torch.Tensor] = {}
        self.self_values: Dict[int, torch.Tensor] = {}
        self.cross_keys: Dict[int, torch.Tensor] = {}
        self.cross_values: Dict[int, torch.Tensor] = {}

    def update_self(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append newly generated K and V to cached self-attention keys and values."""
        if layer_idx not in self.self_keys:
            self.self_keys[layer_idx] = k
            self.self_values[layer_idx] = v
        else:
            self.self_keys[layer_idx] = torch.cat([self.self_keys[layer_idx], k], dim=2)
            self.self_values[layer_idx] = torch.cat([self.self_values[layer_idx], v], dim=2)
        return self.self_keys[layer_idx], self.self_values[layer_idx]

    def set_cross(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store precomputed cross-attention keys and values from encoder output."""
        self.cross_keys[layer_idx] = k
        self.cross_values[layer_idx] = v

    def get_cross(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if layer_idx in self.cross_keys:
            return self.cross_keys[layer_idx], self.cross_values[layer_idx]
        return None

    def get_seq_len(self, layer_idx: int = 0) -> int:
        if layer_idx in self.self_keys:
            return self.self_keys[layer_idx].size(2)
        return 0

    def clear(self):
        self.self_keys.clear()
        self.self_values.clear()
        self.cross_keys.clear()
        self.cross_values.clear()


class ModernAttention(nn.Module):
    """Multi-head attention supporting RoPE, PyTorch SDPA, and KV Caching."""

    def __init__(self, d_model: int, heads: int, dropout: float = 0.1, is_cross: bool = False):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.d_k = d_model // heads
        self.is_cross = is_cross

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        batch_size, q_len, _ = q.shape

        # Linear projections
        query = self.w_q(q).view(batch_size, q_len, self.heads, self.d_k).transpose(1, 2)

        if self.is_cross and kv_cache is not None:
            cached_cross = kv_cache.get_cross(layer_idx)
            if cached_cross is not None:
                key, value = cached_cross
            else:
                k_len = k.size(1)
                key = self.w_k(k).view(batch_size, k_len, self.heads, self.d_k).transpose(1, 2)
                value = self.w_v(v).view(batch_size, k_len, self.heads, self.d_k).transpose(1, 2)
                kv_cache.set_cross(layer_idx, key, value)
        else:
            k_len = k.size(1)
            key = self.w_k(k).view(batch_size, k_len, self.heads, self.d_k).transpose(1, 2)
            value = self.w_v(v).view(batch_size, k_len, self.heads, self.d_k).transpose(1, 2)

            # Apply RoPE to self-attention Q and K
            if rope is not None:
                offset = kv_cache.get_seq_len(layer_idx) if kv_cache is not None else 0
                query, key = rope.apply_rope(query, key, offset=offset)

            # Update KV cache for self-attention
            if kv_cache is not None and not self.is_cross:
                key, value = kv_cache.update_self(layer_idx, key, value)

        is_causal = False
        if mask is not None and mask.dtype != torch.bool:
            mask = mask.bool()
        elif mask is None and not self.is_cross and q_len > 1 and q_len == key.size(2):
            is_causal = True

        # Scaled dot product attention
        attn_out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = attn_out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.w_o(out)


# Backward-compatible aliases
ModernMultiHeadAttention = ModernAttention
ModernFeedForward = SwiGLU


class ModernEncoderBlock(nn.Module):
    """Pre-RMSNorm Encoder Block with RoPE and SwiGLU."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(d_model, heads, dropout=dropout, is_cross=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], rope: RotaryEmbedding) -> torch.Tensor:
        # Pre-LN: x + Sublayer(Norm(x))
        norm_x = self.norm1(x)
        attn = self.self_attn(norm_x, norm_x, norm_x, mask=mask, rope=rope)
        x = x + self.dropout(attn)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ModernDecoderBlock(nn.Module):
    """Pre-RMSNorm Decoder Block with RoPE, Cross-Attention, SwiGLU, and KV Caching."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float = 0.1, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(d_model, heads, dropout=dropout, is_cross=False)
        
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = ModernAttention(d_model, heads, dropout=dropout, is_cross=True)

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
        rope: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Self Attention with RoPE and KV cache
        norm_x = self.norm1(x)
        attn = self.self_attn(
            norm_x, norm_x, norm_x, mask=tgt_mask, rope=rope, kv_cache=kv_cache, layer_idx=self.layer_idx
        )
        x = x + self.dropout(attn)

        # Cross Attention
        norm_x = self.norm2(x)
        cross_out = self.cross_attn(
            norm_x, encoder_output, encoder_output, mask=src_mask, kv_cache=kv_cache, layer_idx=self.layer_idx
        )
        x = x + self.dropout(cross_out)

        # SwiGLU FFN
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class ModernEncoder(nn.Module):
    """Stack of ModernEncoderBlocks with final RMSNorm."""

    def __init__(self, layers: nn.ModuleList, norm: RMSNorm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: RotaryEmbedding,
        use_checkpointing: bool = False,
    ) -> torch.Tensor:
        for block in self.layers:
            if use_checkpointing and self.training:
                x = checkpoint(block, x, mask, rope, use_reentrant=False)
            else:
                x = block(x, mask, rope)
        return self.norm(x)


class ModernDecoder(nn.Module):
    """Stack of ModernDecoderBlocks with final RMSNorm."""

    def __init__(self, layers: nn.ModuleList, norm: RMSNorm):
        super().__init__()
        self.layers = layers
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt_mask: Optional[torch.Tensor],
        rope: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        for block in self.layers:
            x = block(x, encoder_output, src_mask, tgt_mask, rope, kv_cache=kv_cache)
        return self.norm(x)


class ModernTransformer(nn.Module):
    """Modern Encoder-Decoder Transformer with RMSNorm, RoPE, SwiGLU, and KV Caching."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 4,
        heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token Embeddings scaled by sqrt(d_model)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)

        # Shared Rotary Position Embedding
        self.rope = RotaryEmbedding(dim=d_model // heads, max_seq_len=max_seq_len)

        # Encoder stack
        self.encoder_blocks = nn.ModuleList([
            ModernEncoderBlock(d_model, heads, d_ff, dropout) for _ in range(encoder_layers)
        ])
        self.encoder_final_norm = RMSNorm(d_model)

        # Decoder stack
        self.decoder_blocks = nn.ModuleList([
            ModernDecoderBlock(d_model, heads, d_ff, dropout, layer_idx=i) for i in range(decoder_layers)
        ])
        self.decoder_final_norm = RMSNorm(d_model)

        # Final projection layer to vocabulary
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize weights with standard normal (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.src_embed(src) * self.embed_scale
        for block in self.encoder_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, src_mask, self.rope, use_reentrant=False)
            else:
                x = block(x, src_mask, self.rope)
        return self.encoder_final_norm(x)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * self.embed_scale
        for block in self.decoder_blocks:
            if self.use_gradient_checkpointing and self.training and kv_cache is None:
                x = checkpoint(block, x, encoder_output, src_mask, tgt_mask, self.rope, None, use_reentrant=False)
            else:
                x = block(x, encoder_output, src_mask, tgt_mask, self.rope, kv_cache=kv_cache)
        return self.decoder_final_norm(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probabilities over target vocabulary."""
        return F.log_softmax(self.proj(x), dim=-1)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask)
        return self.project(dec_out)


def build_modern_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    encoder_layers: int = 6,
    decoder_layers: int = 4,
    heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    max_seq_len: int = 2048,
    use_gradient_checkpointing: bool = False,
) -> ModernTransformer:
    """Convenience factory for ModernTransformer."""
    return ModernTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        heads=heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
