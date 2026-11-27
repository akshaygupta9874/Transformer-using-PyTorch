"""research/chunked_losses.py

Drop-in replacement for ``chunked_cross_entropy`` in research/optimization.py, plus
``chunked_rdrop_loss``.  Nothing here ever materialises a (B*T, V) tensor: logits are
recomputed ``chunk_size`` rows at a time in both forward and backward.

Design
------
* The autograd Functions return a **per-token loss vector** (N_valid,).  Weighting,
  reduction and normalisation are ordinary tensor ops on that vector, so
  ``sample_weight``, ``reduction="sum"`` and the grad-accumulation ``normalizer`` cost
  nothing extra and need no custom backward code.
* Per-token loss (label smoothing eps over V classes, PyTorch convention):
      L = lse - (1-eps)*z_y - eps*mean_j(z_j) + z_loss * lse**2
  d L / d z_j = softmax_j * (1 + 2*z_loss*lse) - (1-eps)*[j==y] - eps/V
* R-Drop per token:
      L = ce_weight * (CE1 + CE2) + alpha * 0.5*(KL(p1||p2) + KL(p2||p1))
  where CE_k includes label smoothing and z-loss.  With ce_weight=0.5 this is
  ``(CE1+CE2)/2 + alpha*symKL``; ce_weight=1.0 gives the original R-Drop paper form.
  d symKL / d z1_k = 0.5 * [ p1_k*(d_k - KL12) + (p1_k - p2_k) ],  d = logp1 - logp2
  d symKL / d z2_k = 0.5 * [ p2_k*(-d_k - KL21) + (p2_k - p1_k) ]
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

Number = Union[float, torch.Tensor]


def _resolve_compute_dtype(hidden: torch.Tensor, compute_dtype: Optional[torch.dtype]) -> torch.dtype:
    if compute_dtype is not None:
        return compute_dtype
    dev = hidden.device.type
    if dev in ("cuda", "cpu") and torch.is_autocast_enabled(dev):
        return torch.get_autocast_dtype(dev)
    return hidden.dtype


def _acc_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    # bf16/fp16 -> fp32 ; fp32 -> fp32 ; fp64 -> fp64 (lets gradcheck run in double)
    return torch.promote_types(compute_dtype, torch.float32)


# --------------------------------------------------------------------------------------
# Cross-entropy (+ label smoothing + z-loss)
# --------------------------------------------------------------------------------------
class _ChunkedLinearCE(torch.autograd.Function):
    """Per-token CE for logits = x @ W.T, recomputed chunk-wise in backward.

    forward(x (N,d), W (V,d), y (N,) long, eps, z_loss, chunk_size, compute_dtype)
        -> (loss (N,), lse (N,))      # lse is non-differentiable (for logging / z-loss stats)
    """

    @staticmethod
    def forward(ctx, x, W, y, label_smoothing, z_loss, chunk_size, compute_dtype):
        N, _ = x.shape
        acc = _acc_dtype(compute_dtype)
        Wc = W.to(compute_dtype)
        xc = x.to(compute_dtype)
        loss = torch.empty(N, dtype=acc, device=x.device)
        lse_all = torch.empty(N, dtype=acc, device=x.device)
        eps = float(label_smoothing)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            z = (xc[s:e] @ Wc.t()).to(acc)                       # (n, V) fp32
            lse = torch.logsumexp(z, dim=-1)
            zy = z.gather(1, y[s:e, None]).squeeze(1)
            l = lse - (1.0 - eps) * zy - eps * z.mean(dim=-1)
            if z_loss:
                l = l + z_loss * lse.square()
            loss[s:e] = l
            lse_all[s:e] = lse
            del z
        ctx.save_for_backward(x, W, y, lse_all)
        ctx.hp = (eps, float(z_loss), int(chunk_size), compute_dtype)
        ctx.mark_non_differentiable(lse_all)
        return loss, lse_all

    @staticmethod
    def backward(ctx, g_loss, _g_lse):
        x, W, y, lse = ctx.saved_tensors
        eps, zl, cs, cd = ctx.hp
        N, d = x.shape
        V = W.shape[0]
        acc = lse.dtype
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        Wc = W.to(cd)
        xc = x.to(cd)
        gx = torch.empty_like(xc) if need_x else None
        gW = torch.zeros(V, d, dtype=acc, device=W.device) if need_w else None
        g = g_loss.to(acc)
        for s in range(0, N, cs):
            e = min(s + cs, N)
            z = (xc[s:e] @ Wc.t()).to(acc)
            ge, lse_c = g[s:e], lse[s:e]
            p = z.sub_(lse_c[:, None]).exp_()                    # softmax, in place
            gz = p.mul_((ge * (1.0 + 2.0 * zl * lse_c))[:, None])
            gz.sub_((ge * (eps / V))[:, None])
            gz.scatter_add_(1, y[s:e, None], (-(1.0 - eps) * ge)[:, None])
            gzc = gz.to(cd)
            if need_x:
                gx[s:e] = gzc @ Wc
            if need_w:
                gW += (gzc.t() @ xc[s:e]).to(acc)
            del z, p, gz, gzc
        return (gx.to(x.dtype) if need_x else None,
                gW.to(W.dtype) if need_w else None,
                None, None, None, None, None)


# --------------------------------------------------------------------------------------
# R-Drop: 0.5*(CE1+CE2)-style CE + alpha * symmetric KL, chunked
# --------------------------------------------------------------------------------------
class _ChunkedRDrop(torch.autograd.Function):
    """forward(x1, x2 (N,d), W (V,d), y, eps, z_loss, alpha, ce_weight, chunk, cd)
        -> (loss (N,), symkl (N,), lse1 (N,), lse2 (N,))   # last three non-differentiable"""

    @staticmethod
    def forward(ctx, x1, x2, W, y, label_smoothing, z_loss, alpha, ce_weight, chunk_size, compute_dtype):
        N, _ = x1.shape
        acc = _acc_dtype(compute_dtype)
        Wc = W.to(compute_dtype)
        x1c, x2c = x1.to(compute_dtype), x2.to(compute_dtype)
        dev = x1.device
        loss, sym, l1, l2, k12, k21 = (torch.empty(N, dtype=acc, device=dev) for _ in range(6))
        eps = float(label_smoothing)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            z1 = (x1c[s:e] @ Wc.t()).to(acc)
            z2 = (x2c[s:e] @ Wc.t()).to(acc)
            lse1, lse2 = torch.logsumexp(z1, -1), torch.logsumexp(z2, -1)
            lp1, lp2 = z1 - lse1[:, None], z2 - lse2[:, None]
            kl12 = (lp1.exp() * (lp1 - lp2)).sum(-1)
            kl21 = (lp2.exp() * (lp2 - lp1)).sum(-1)
            yy = y[s:e, None]
            ce1 = lse1 - (1 - eps) * z1.gather(1, yy).squeeze(1) - eps * z1.mean(-1)
            ce2 = lse2 - (1 - eps) * z2.gather(1, yy).squeeze(1) - eps * z2.mean(-1)
            if z_loss:
                ce1 = ce1 + z_loss * lse1.square()
                ce2 = ce2 + z_loss * lse2.square()
            sk = 0.5 * (kl12 + kl21)
            loss[s:e] = ce_weight * (ce1 + ce2) + alpha * sk
            sym[s:e], l1[s:e], l2[s:e], k12[s:e], k21[s:e] = sk, lse1, lse2, kl12, kl21
            del z1, z2, lp1, lp2
        ctx.save_for_backward(x1, x2, W, y, l1, l2, k12, k21)
        ctx.hp = (eps, float(z_loss), float(alpha), float(ce_weight), int(chunk_size), compute_dtype)
        ctx.mark_non_differentiable(sym, l1, l2)
        return loss, sym, l1, l2

    @staticmethod
    def backward(ctx, g_loss, *_):
        x1, x2, W, y, lse1, lse2, kl12, kl21 = ctx.saved_tensors
        eps, zl, alpha, cew, cs, cd = ctx.hp
        N, d = x1.shape
        V = W.shape[0]
        acc = lse1.dtype
        n1, n2, nw = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        Wc = W.to(cd)
        x1c, x2c = x1.to(cd), x2.to(cd)
        gx1 = torch.empty_like(x1c) if n1 else None
        gx2 = torch.empty_like(x2c) if n2 else None
        gW = torch.zeros(V, d, dtype=acc, device=W.device) if nw else None
        g = g_loss.to(acc)
        h = 0.5 * alpha
        for s in range(0, N, cs):
            e = min(s + cs, N)
            ge = g[s:e]
            z1 = (x1c[s:e] @ Wc.t()).to(acc)
            z2 = (x2c[s:e] @ Wc.t()).to(acc)
            a1, a2 = lse1[s:e], lse2[s:e]
            lp1 = z1.sub_(a1[:, None])
            lp2 = z2.sub_(a2[:, None])
            dlt = lp1 - lp2                      # computed in log space: no 0*inf when p underflows
            p1 = lp1.exp_()
            p2 = lp2.exp_()
            # --- branch 1
            r1 = ge * (cew * (1 + 2 * zl * a1) + h * (1 - kl12[s:e]))
            gz1 = p1 * r1[:, None]
            gz1.add_(h * ge[:, None] * (p1 * dlt - p2))
            gz1.sub_((ge * cew * eps / V)[:, None])
            gz1.scatter_add_(1, y[s:e, None], (-(1 - eps) * cew * ge)[:, None])
            # --- branch 2
            r2 = ge * (cew * (1 + 2 * zl * a2) + h * (1 - kl21[s:e]))
            gz2 = p2 * r2[:, None]
            gz2.add_(h * ge[:, None] * (-p2 * dlt - p1))
            gz2.sub_((ge * cew * eps / V)[:, None])
            gz2.scatter_add_(1, y[s:e, None], (-(1 - eps) * cew * ge)[:, None])
            g1c, g2c = gz1.to(cd), gz2.to(cd)
            if n1:
                gx1[s:e] = g1c @ Wc
            if n2:
                gx2[s:e] = g2c @ Wc
            if nw:
                gW += (g1c.t() @ x1c[s:e]).to(acc)
                gW += (g2c.t() @ x2c[s:e]).to(acc)
            del z1, z2, p1, p2, dlt, gz1, gz2, g1c, g2c
        return (gx1.to(x1.dtype) if n1 else None, gx2.to(x2.dtype) if n2 else None,
                gW.to(W.dtype) if nw else None, None, None, None, None, None, None, None)


# --------------------------------------------------------------------------------------
# Public wrappers
# --------------------------------------------------------------------------------------
def _prep(hidden_list, target, ignore_index, sample_weight):
    d = hidden_list[0].shape[-1]
    t = target.reshape(-1)
    idx = (t != ignore_index).nonzero(as_tuple=True)[0]
    xs = [h.reshape(-1, d).index_select(0, idx) for h in hidden_list]
    w = None
    if sample_weight is not None:
        sw = sample_weight
        if sw.dim() < target.dim():                       # (B,) -> (B,1) -> broadcast over T
            sw = sw.reshape(tuple(sw.shape) + (1,) * (target.dim() - sw.dim()))
        w = sw.expand_as(target).reshape(-1).index_select(0, idx)
    return idx, xs, t.index_select(0, idx), w


def _reduce(loss_vec, w, reduction, normalizer):
    acc = loss_vec.dtype
    w = None if w is None else w.to(acc)
    total = loss_vec.sum() if w is None else (loss_vec * w).sum()
    if normalizer is not None:
        denom = normalizer
    elif reduction == "sum":
        denom = 1.0
    elif reduction == "mean":
        denom = float(loss_vec.numel()) if w is None else w.sum().clamp_min(1e-12)
    else:
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    return total / denom, w, denom


def chunked_cross_entropy(
    hidden: torch.Tensor,                 # (..., d)   post decoder_final_norm
    weight: torch.Tensor,                 # (V, d)     model.proj.weight (tied)
    target: torch.Tensor,                 # (...)      long, PAD rows ignored
    label_smoothing: float = 0.1,
    ignore_index: int = 0,
    reduction: str = "mean",
    sample_weight: Optional[torch.Tensor] = None,   # (B,) or same shape as target
    chunk_size: int = 1024,
    *,
    z_loss: float = 0.0,                  # alpha * logsumexp(z)**2 ; 1e-4 is the PaLM value
    normalizer: Optional[Number] = None,  # overrides reduction: loss = sum(w*L) / normalizer
    compute_dtype: Optional[torch.dtype] = None,
    return_info: bool = False,
):
    """Same signature as before (+ keyword-only z_loss / normalizer / return_info).

    reduction="mean": sum(w*L)/sum(w)   (w=1 -> divide by #non-pad tokens)
    reduction="sum" : sum(w*L)
    normalizer=N    : sum(w*L)/N   <- use this for gradient accumulation (see research/accum.py)
    """
    cd = _resolve_compute_dtype(hidden, compute_dtype)
    idx, (x,), y, w = _prep([hidden], target, ignore_index, sample_weight)
    if idx.numel() == 0:
        zero = hidden.sum() * 0.0 + weight.sum() * 0.0
        return (zero, {}) if return_info else zero
    loss_vec, lse = _ChunkedLinearCE.apply(x, weight, y, label_smoothing, z_loss, chunk_size, cd)
    out, w, denom = _reduce(loss_vec, w, reduction, normalizer)
    if not return_info:
        return out
    with torch.no_grad():
        zl_vec = z_loss * lse.square()
        ww = torch.ones_like(loss_vec) if w is None else w
        info = {"ce": ((loss_vec - zl_vec) * ww).sum() / denom,
                "z": (zl_vec * ww).sum() / denom,
                "lse_mean": lse.mean(), "lse_max": lse.max(),
                "n_tokens": torch.tensor(float(idx.numel()))}
    return out, info


def chunked_rdrop_loss(
    h1: torch.Tensor, h2: torch.Tensor,   # (..., d) two dropout passes of decoder_final_norm output
    weight: torch.Tensor,                 # (V, d)
    target: torch.Tensor,
    label_smoothing: float = 0.1,
    rdrop_alpha: float = 0.4,
    ignore_index: int = 0,
    reduction: str = "mean",
    sample_weight: Optional[torch.Tensor] = None,
    chunk_size: int = 256,                # two logit tensors + temporaries per chunk -> keep smaller
    *,
    z_loss: float = 0.0,
    ce_weight: float = 0.5,               # 0.5: (CE1+CE2)/2 ; 1.0: CE1+CE2 (paper form)
    normalizer: Optional[Number] = None,
    compute_dtype: Optional[torch.dtype] = None,
    return_info: bool = False,
):
    """ce_weight*(CE1+CE2) + rdrop_alpha*0.5*(KL(p1||p2)+KL(p2||p1)); never builds (N, V)."""
    cd = _resolve_compute_dtype(h1, compute_dtype)
    idx, (x1, x2), y, w = _prep([h1, h2], target, ignore_index, sample_weight)
    if idx.numel() == 0:
        zero = h1.sum() * 0.0 + h2.sum() * 0.0 + weight.sum() * 0.0
        return (zero, {}) if return_info else zero
    loss_vec, sym, l1, l2 = _ChunkedRDrop.apply(
        x1, x2, weight, y, label_smoothing, z_loss, rdrop_alpha, ce_weight, chunk_size, cd)
    out, w, denom = _reduce(loss_vec, w, reduction, normalizer)
    if not return_info:
        return out
    with torch.no_grad():
        ww = torch.ones_like(loss_vec) if w is None else w
        info = {"symkl": (sym * ww).sum() / denom,
                "loss_total": (loss_vec * ww).sum() / denom,
                "lse_mean": 0.5 * (l1.mean() + l2.mean()),
                "n_tokens": torch.tensor(float(idx.numel()))}
    return out, info


@torch.no_grad()
def chunked_nll_sum(hidden, weight, target, ignore_index: int = 0, chunk_size: int = 1024):
    """Validation NLL (no smoothing, no z-loss) as (sum_nll, n_tokens) without (B*T,V)."""
    s = chunked_cross_entropy(hidden, weight, target, label_smoothing=0.0, ignore_index=ignore_index,
                              reduction="sum", chunk_size=chunk_size, z_loss=0.0)
    return s, (target != ignore_index).sum()


def chunked_token_logprobs(hidden, weight, target, chunk_size: int = 1024, compute_dtype=None):
    """Differentiable per-token log p(y|.) of shape target.shape (no PAD handling: all rows used).
    For mrt_loss / MBR scoring: log p = -CE with eps=0, z_loss=0."""
    cd = _resolve_compute_dtype(hidden, compute_dtype)
    d = hidden.shape[-1]
    loss_vec, _ = _ChunkedLinearCE.apply(hidden.reshape(-1, d), weight, target.reshape(-1), 0.0, 0.0,
                                         chunk_size, cd)
    return (-loss_vec).reshape(target.shape)
