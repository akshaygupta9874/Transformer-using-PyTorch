"""research/optim_groups.py -- AdamW parameter groups that understand tied embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_param_groups(model: nn.Module, weight_decay: float = 0.01, decay_embeddings: bool = False):
    """decay: >=2-D weights that are not embeddings (attention/FFN matrices).
    no decay: 1-D params (RMSNorm gains, QK-norm, biases) and -- unless decay_embeddings -- every
    nn.Embedding weight, which includes the tied output projection (same Parameter object)."""
    emb_ids = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():          # named_parameters() de-duplicates tied params
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim < 2 or (id(p) in emb_ids and not decay_embeddings):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(model, lr=3e-4, weight_decay=0.01, betas=None, eps=1e-9, decay_embeddings=False):
    groups = build_param_groups(model, weight_decay, decay_embeddings)
    kw = {} if betas is None else {"betas": betas}          # betas: keep your current setting
    return torch.optim.AdamW(groups, lr=lr, eps=eps, fused=torch.cuda.is_available(), **kw)
