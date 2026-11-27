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

    def _ensure_cache(self, need_len: int) -> None:
        if need_len > self.cos_cached.size(0):
            t = torch.arange(need_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin tensors sliced for offset or gathered for per-token pos (B, T)."""
        if pos is not None:
            need_len = int(pos.max().item()) + 1 if pos.numel() > 0 else 1
            self._ensure_cache(need_len)
            cos = self.cos_cached[pos].unsqueeze(1)  # (B, 1, seq_len, dim)
            sin = self.sin_cached[pos].unsqueeze(1)  # (B, 1, seq_len, dim)
            return cos.to(x.dtype), sin.to(x.dtype)

        seq_len = x.size(-2)
        need_len = offset + seq_len
        self._ensure_cache(need_len)
        cos = self.cos_cached[offset:need_len, :].unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[offset:need_len, :].unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, dim)
        return cos.to(x.dtype), sin.to(x.dtype)

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to Q and K tensors."""
        cos, sin = self.forward(q, offset=offset, pos=pos)
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

    def __init__(self, record_cross_attn: bool = False):
        self.self_keys: Dict[int, torch.Tensor] = {}
        self.self_values: Dict[int, torch.Tensor] = {}
        self.cross_keys: Dict[int, torch.Tensor] = {}
        self.cross_values: Dict[int, torch.Tensor] = {}
        self.record_cross_attn = record_cross_attn
        self.last_cross_attn: Dict[int, torch.Tensor] = {}

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
        self.last_cross_attn.clear()


class ModernAttention(nn.Module):
    """Multi-head / Grouped-Query attention supporting QK-Norm, RoPE (self-only), PyTorch SDPA, and KV Caching."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float = 0.1,
        is_cross: bool = False,
        use_qk_norm: bool = True,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        kv_heads = heads if kv_heads is None else int(kv_heads)
        assert heads % kv_heads == 0, "heads must be divisible by kv_heads"
        self.d_model = d_model
        self.heads = heads
        self.kv_heads = kv_heads
        self.d_k = d_model // heads
        self.is_cross = is_cross
        self.use_qk_norm = use_qk_norm

        self.w_q = nn.Linear(d_model, heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(heads * self.d_k, d_model, bias=False)
        self.q_norm = RMSNorm(self.d_k) if use_qk_norm else None
        self.k_norm = RMSNorm(self.d_k) if use_qk_norm else None
        self.dropout = dropout

    def _expand_kv(self, t: torch.Tensor) -> torch.Tensor:
        """Expand (B, kv_heads, S, d_k) to (B, heads, S, d_k) for GQA."""
        g = self.heads // self.kv_heads
        if g == 1:
            return t
        b, _, s, d = t.shape
        return t[:, :, None].expand(b, self.kv_heads, g, s, d).reshape(b, self.heads, s, d)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryEmbedding] = None,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = q.shape

        # Linear projections
        query = self.w_q(q).view(batch_size, q_len, self.heads, self.d_k).transpose(1, 2)
        if self.q_norm is not None:
            query = self.q_norm(query)

        if self.is_cross and kv_cache is not None:
            cached_cross = kv_cache.get_cross(layer_idx)
            if cached_cross is not None:
                key, value = cached_cross
            else:
                k_len = k.size(1)
                key = self.w_k(k).view(batch_size, k_len, self.kv_heads, self.d_k).transpose(1, 2)
                if self.k_norm is not None:
                    key = self.k_norm(key)
                value = self.w_v(v).view(batch_size, k_len, self.kv_heads, self.d_k).transpose(1, 2)
                kv_cache.set_cross(layer_idx, key, value)
        else:
            k_len = k.size(1)
            key = self.w_k(k).view(batch_size, k_len, self.kv_heads, self.d_k).transpose(1, 2)
            if self.k_norm is not None:
                key = self.k_norm(key)
            value = self.w_v(v).view(batch_size, k_len, self.kv_heads, self.d_k).transpose(1, 2)

            # Apply RoPE to self-attention Q and K only (cross-attention omits RoPE)
            if rope is not None:
                offset = kv_cache.get_seq_len(layer_idx) if kv_cache is not None else 0
                query, key = rope.apply_rope(query, key, offset=offset, pos=pos)

            # Update KV cache for self-attention at kv_heads size
            if kv_cache is not None and not self.is_cross:
                key, value = kv_cache.update_self(layer_idx, key, value)

        key = self._expand_kv(key)
        value = self._expand_kv(value)

        is_causal = False
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        elif not self.is_cross and q_len > 1 and q_len == key.size(2):
            is_causal = True

        # Capture cross-attention weights only when requested (e.g. beam coverage penalty)
        if self.is_cross and kv_cache is not None and getattr(kv_cache, "record_cross_attn", False):
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
            kv_cache.last_cross_attn[layer_idx] = attn_weights.mean(dim=1)[:, -1, :].detach()
            attn_out = torch.matmul(attn_weights, value)
        else:
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
    """Pre-RMSNorm Encoder Block with QK-Norm, RoPE, GQA option, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(
            d_model, heads, dropout=dropout, is_cross=False, use_qk_norm=use_qk_norm, kv_heads=kv_heads
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: RotaryEmbedding,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN: x + Sublayer(Norm(x))
        norm_x = self.norm1(x)
        attn = self.self_attn(norm_x, norm_x, norm_x, mask=mask, rope=rope, pos=pos)
        x = x + self.dropout(attn)

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ModernDecoderBlock(nn.Module):
    """Pre-RMSNorm Decoder Block with QK-Norm, RoPE, Cross-Attention, GQA option, SwiGLU, and KV Caching."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        d_ff: int,
        dropout: float = 0.1,
        layer_idx: int = 0,
        use_qk_norm: bool = True,
        kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(d_model)
        self.self_attn = ModernAttention(
            d_model, heads, dropout=dropout, is_cross=False, use_qk_norm=use_qk_norm, kv_heads=kv_heads
        )
        
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = ModernAttention(
            d_model, heads, dropout=dropout, is_cross=True, use_qk_norm=use_qk_norm, kv_heads=kv_heads
        )

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
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self Attention with RoPE and KV cache
        norm_x = self.norm1(x)
        attn = self.self_attn(
            norm_x, norm_x, norm_x, mask=tgt_mask, rope=rope, kv_cache=kv_cache, layer_idx=self.layer_idx, pos=pos
        )
        x = x + self.dropout(attn)

        # Cross Attention (no RoPE: source and target share no relative coordinate frame)
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
    """Modern Encoder-Decoder Transformer with RMSNorm, QK-Norm, GQA, RoPE, SwiGLU, and KV Caching."""

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
        use_qk_norm: bool = True,
        tie_embeddings: bool = False,
        kv_heads: Optional[int] = None,
        encoder_d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_qk_norm = use_qk_norm
        self.tie_embeddings = tie_embeddings
        self.kv_heads = heads if kv_heads is None else int(kv_heads)
        enc_ff = d_ff if encoder_d_ff is None else int(encoder_d_ff)

        # Token Embeddings scaled by sqrt(d_model)
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.embed_scale = math.sqrt(d_model)

        # Shared Rotary Position Embedding
        self.rope = RotaryEmbedding(dim=d_model // heads, max_seq_len=max_seq_len)

        # Encoder stack
        self.encoder_blocks = nn.ModuleList([
            ModernEncoderBlock(
                d_model, heads, enc_ff, dropout, use_qk_norm=use_qk_norm, kv_heads=self.kv_heads
            )
            for _ in range(encoder_layers)
        ])
        self.encoder_final_norm = RMSNorm(d_model)

        # Decoder stack
        self.decoder_blocks = nn.ModuleList([
            ModernDecoderBlock(
                d_model, heads, d_ff, dropout, layer_idx=i, use_qk_norm=use_qk_norm, kv_heads=self.kv_heads
            )
            for i in range(decoder_layers)
        ])
        self.decoder_final_norm = RMSNorm(d_model)

        # Final projection layer to vocabulary
        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Initialize weights with base normal (std=0.02), then depth-scale residual projections
        self.apply(self._init_weights)
        self._apply_depth_scaled_residual_init()

        if tie_embeddings:
            self.proj.weight = self.tgt_embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_depth_scaled_residual_init(self) -> None:
        """Scale residual output projections (w_o, w_down) by 0.02 / sqrt(2 * N)."""
        enc_std = 0.02 / math.sqrt(2 * max(1, self.encoder_layers))
        for block in self.encoder_blocks:
            torch.nn.init.normal_(block.self_attn.w_o.weight, mean=0.0, std=enc_std)
            torch.nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=enc_std)

        dec_std = 0.02 / math.sqrt(2 * max(1, self.decoder_layers))
        for block in self.decoder_blocks:
            torch.nn.init.normal_(block.self_attn.w_o.weight, mean=0.0, std=dec_std)
            torch.nn.init.normal_(block.cross_attn.w_o.weight, mean=0.0, std=dec_std)
            torch.nn.init.normal_(block.ffn.w_down.weight, mean=0.0, std=dec_std)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.src_embed(src) * self.embed_scale
        for block in self.encoder_blocks:
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, src_mask, self.rope, pos, use_reentrant=False)
            else:
                x = block(x, src_mask, self.rope, pos=pos)
        return self.encoder_final_norm(x)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.tgt_embed(tgt) * self.embed_scale
        for block in self.decoder_blocks:
            if self.use_gradient_checkpointing and self.training and kv_cache is None:
                x = checkpoint(block, x, encoder_output, src_mask, tgt_mask, self.rope, None, pos, use_reentrant=False)
            else:
                x = block(x, encoder_output, src_mask, tgt_mask, self.rope, kv_cache=kv_cache, pos=pos)
        return self.decoder_final_norm(x)

    def project_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw unnormalized logits over target vocabulary (avoids double log_softmax with CrossEntropyLoss)."""
        return self.proj(x)

    def log_probs(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Return true fp32 log probabilities over target vocabulary (avoids bf16 precision loss over 40k vocab)."""
        z = self.proj(x).float()
        return F.log_softmax(z if temperature == 1.0 else z / temperature, dim=-1)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Return log probabilities over target vocabulary (compatibility shim for log_probs)."""
        return self.log_probs(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_pos: Optional[torch.Tensor] = None,
        tgt_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enc_out = self.encode(src, src_mask, pos=src_pos)
        dec_out = self.decode(enc_out, src_mask, tgt, tgt_mask, pos=tgt_pos)
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
    use_qk_norm: bool = True,
    tie_embeddings: bool = False,
    kv_heads: Optional[int] = None,
    encoder_d_ff: Optional[int] = None,
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
        use_qk_norm=use_qk_norm,
        tie_embeddings=tie_embeddings,
        kv_heads=kv_heads,
        encoder_d_ff=encoder_d_ff,
    )
