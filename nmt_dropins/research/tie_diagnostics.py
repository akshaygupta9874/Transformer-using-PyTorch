"""research/tie_diagnostics.py -- measure how the tied matrix W is trained by each of its two roles.

Gradient w.r.t. W has two additive parts:
  * embedding path: tgt_embed(tgt) * embed_scale feeds the decoder           (sparse rows)
  * projection path: logits = decoder_final_norm(h) @ W.T                    (dense rows)
We split them by (a) making the embedding lookup use W.detach() via a forward hook, or (b) giving
the loss W.detach().  Run once at init and again after ~1-2k steps; log it.

    stats = tied_grad_split(model, batch, forward_hidden, device)
    # stats["ratio_emb_over_proj"] ~ 1   -> balanced.   >>10 or <<0.1 -> one role dominates Adam's step.
"""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn.functional as F

from research.chunked_losses import chunked_cross_entropy


def tied_grad_split(model, batch, forward_hidden: Callable, device, label_smoothing: float = 0.1,
                    chunk_size: int = 1024) -> Dict[str, float]:
    W = model.tgt_embed.weight
    assert W is model.proj.weight, "model is not weight-tied"
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model.train()

    def grad_of(emb_path: bool, proj_path: bool):
        model.zero_grad(set_to_none=True)
        handle = None
        if not emb_path:
            handle = model.tgt_embed.register_forward_hook(
                lambda m, inp, out: F.embedding(inp[0], m.weight.detach()))
        try:
            h, tgt, sw = forward_hidden(model, batch)
            w = W if proj_path else W.detach()
            loss = chunked_cross_entropy(h, w, tgt, label_smoothing, 0, "mean", sw, chunk_size)
            (g,) = torch.autograd.grad(loss, W, allow_unused=True)
        finally:
            if handle is not None:
                handle.remove()
        return torch.zeros_like(W) if g is None else g

    g_emb = grad_of(True, False)
    g_proj = grad_of(False, True)
    n_e, n_p = g_emb.norm().item(), g_proj.norm().item()
    cos = F.cosine_similarity(g_emb.flatten(), g_proj.flatten(), dim=0).item()
    return {"norm_emb_path": n_e, "norm_proj_path": n_p,
            "ratio_emb_over_proj": n_e / max(n_p, 1e-30), "cosine": cos}
