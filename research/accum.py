"""research/accum.py -- token-normalised gradient accumulation (VRAM-safe).

Why: with variable-size micro-batches, ``loss_mb_mean / ga`` weights every micro-batch equally, so a
1,500-token micro-batch gets the same total weight as a 4,000-token one -> its tokens count ~2.7x as much.
Correct objective for the window:   sum over all valid tokens (of all micro-batches) / total valid tokens.
Implementation: pull ``ga`` micro-batches on CPU, count valid tokens (or sum sample weights) *without
touching the GPU*, then run each micro-batch with reduction="sum", normalizer=window_total.

Only ONE micro-batch (and its activations) is on the GPU at a time; the others stay on the CPU.
"""
from __future__ import annotations

import itertools
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from research.chunked_losses import chunked_cross_entropy, chunked_rdrop_loss

# forward_hidden(model, batch) -> (hidden, target, sample_weight|None)       [normal path]
# For R-Drop it is called twice (dropout makes the two passes differ).
ForwardHidden = Callable[[torch.nn.Module, Dict[str, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]


def default_forward_hidden(model, batch):
    """ModernTransformer forward returning (hidden, target, sample_weight).
    Seamlessly supports both {"src", "src_mask", "tgt_in", ...} and
    {"encoder_input", "encoder_mask", "decoder_input", "decoder_mask", "label", ...} collations.
    """
    if "src" in batch:
        src = batch["src"]
        src_mask = batch.get("src_mask")
        tgt_in = batch["tgt_in"]
        tgt_mask = batch.get("tgt_mask")
        tgt_out = batch.get("tgt_out", batch.get("label"))
        weight = batch.get("weight", batch.get("sample_weight"))
        src_pos = batch.get("src_pos")
        tgt_pos = batch.get("tgt_pos")
        cross_mask = batch.get("cross_mask", src_mask)
    else:
        src = batch["encoder_input"]
        src_mask = batch.get("encoder_mask")
        tgt_in = batch["decoder_input"]
        tgt_mask = batch.get("decoder_mask")
        tgt_out = batch.get("label", batch.get("tgt_out"))
        weight = batch.get("sample_weight", batch.get("weight"))
        src_pos = batch.get("src_pos")
        tgt_pos = batch.get("tgt_pos")
        cross_mask = batch.get("cross_mask", src_mask)

    if src_pos is not None:
        enc = model.encode(src, src_mask, pos=src_pos)
    else:
        enc = model.encode(src, src_mask)

    if tgt_pos is not None:
        h = model.decode(enc, cross_mask, tgt_in, tgt_mask, pos=tgt_pos)
    else:
        h = model.decode(enc, cross_mask, tgt_in, tgt_mask)
    return h, tgt_out, weight


def windows(loader: Iterable, ga: int) -> Iterator[List[Any]]:
    """Group consecutive batches into accumulation windows (last window may be shorter)."""
    it = iter(loader)
    while True:
        w = list(itertools.islice(it, ga))
        if not w:
            return
        yield w


def window_normalizer(
    window: List[Dict[str, torch.Tensor]],
    ignore_index: int = 0,
    target_key: str = "tgt_out",
    weight_key: str = "weight",
) -> float:
    """Total valid tokens (or total sample weight over valid tokens) in the window.  CPU tensors only."""
    total = 0.0
    for b in window:
        t = b.get(target_key)
        if t is None:
            t = b.get("label", b.get("tgt_out"))
        if t is None:
            continue
        valid = t != ignore_index
        w = b.get(weight_key)
        if w is None:
            w = b.get("sample_weight")
        if w is None:
            total += float(valid.sum())
        else:
            if w.dim() < t.dim():
                w = w.reshape(tuple(w.shape) + (1,) * (t.dim() - w.dim()))
            total += float((valid * w.expand_as(t).to(torch.float64)).sum())
    return max(total, 1e-12)


def to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_window(
    model: torch.nn.Module,
    window: List[Dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device,
    forward_hidden: ForwardHidden = default_forward_hidden,
    label_smoothing: float = 0.1,
    z_loss: float = 1e-4,
    rdrop_alpha: float = 0.0,
    ignore_index: int = 0,
    max_grad_norm: Optional[float] = 1.0,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    scheduler=None,
    ema=None,
    chunk_size: int = 1024,
    rdrop_chunk_size: int = 256,
    target_key: str = "tgt_out",
    skip_nonfinite: bool = True,
) -> Dict[str, float]:
    """One optimizer step over ``window`` (a list of CPU micro-batches).  Returns python floats."""
    model.train()
    N = window_normalizer(window, ignore_index, target_key)
    optimizer.zero_grad(set_to_none=True)
    acc_loss = acc_ce = acc_z = acc_kl = None
    ac = (torch.autocast(device.type if hasattr(device, "type") else str(device).split(":")[0], dtype=autocast_dtype)
          if autocast_dtype is not None else nullcontext())
    for mb in window:
        mb = to_device(mb, device)
        with ac:
            h1, tgt, sw = forward_hidden(model, mb)
            if rdrop_alpha > 0:
                h2, _, _ = forward_hidden(model, mb)           # second dropout realisation
                loss, info = chunked_rdrop_loss(
                    h1, h2, model.proj.weight, tgt, label_smoothing, rdrop_alpha, ignore_index,
                    "sum", sw, rdrop_chunk_size, z_loss=z_loss, normalizer=N, return_info=True)
            else:
                loss, info = chunked_cross_entropy(
                    h1, model.proj.weight, tgt, label_smoothing, ignore_index, "sum", sw, chunk_size,
                    z_loss=z_loss, normalizer=N, return_info=True)
        loss.backward()                                         # grads already normalised by window total
        with torch.no_grad():                                   # accumulate metrics on-device, no sync
            acc_loss = loss.detach() if acc_loss is None else acc_loss + loss.detach()
            if "ce" in info:
                acc_ce = info["ce"] if acc_ce is None else acc_ce + info["ce"]
                acc_z = info["z"] if acc_z is None else acc_z + info["z"]
            if "symkl" in info:
                acc_kl = info["symkl"] if acc_kl is None else acc_kl + info["symkl"]
        del h1, loss
    gnorm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm if max_grad_norm is not None else float("inf"))
    if skip_nonfinite and not bool(torch.isfinite(gnorm)):
        optimizer.zero_grad(set_to_none=True)
        return {"loss": float(acc_loss), "skipped": 1.0, "grad_norm": float(gnorm), "tokens": N}
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    if ema is not None:
        ema.update(model)
    out = {"loss": float(acc_loss)}                             # one sync per optimizer step
    if acc_ce is not None:
        out["ce"], out["z"] = float(acc_ce), float(acc_z)
    if acc_kl is not None:
        out["symkl"] = float(acc_kl)
    out["grad_norm"] = float(gnorm)
    out["tokens"] = N
    return out


# ------------------------------------------------------------------ usage in train.py
# for epoch in range(start_epoch, epochs):
#     batch_sampler.set_epoch(epoch)                    # loader built with batch_sampler=batch_sampler
#     for window in windows(loader, gradient_accumulation_steps):
#         stats = train_window(model, window, optimizer, device=device, forward_hidden=my_forward,
#                              rdrop_alpha=rdrop_alpha, scheduler=scheduler, ema=ema)
# Total optimizer steps for the cosine schedule:
#     math.ceil(batch_sampler.expected_len / ga) * epochs
