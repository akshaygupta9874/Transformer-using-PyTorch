"""research/logits_utils.py -- one place where logits become (log-)probabilities.

Rule: the model returns RAW logits (``project_logits``).  Every consumer converts exactly once.

    greedy      -> argmax(raw logits)                      (no softmax at all; mask on logits)
    beam / MBR  -> step_log_probs(...)                     (one fp32 log_softmax)
    sampling    -> sample_from_logits(...)                 (temperature, mask, top-p/top-k on logits)
    kNN-MT      -> step_log_probs(...) then log-space interpolation with the kNN distribution

Patch for models/modern.py (rename instead of silently changing project()'s meaning):

    def project_logits(self, x):            # primitive, unchanged
        return self.proj(x)

    def log_probs(self, x, temperature: float = 1.0):
        z = self.proj(x).float()            # fp32 BEFORE log_softmax (bf16 over a 40k vocab is lossy)
        return F.log_softmax(z if temperature == 1.0 else z / temperature, dim=-1)

    def project(self, x):                   # temporary shim; delete after migrating callers
        import warnings; warnings.warn("project() is deprecated: use log_probs()/project_logits()", stacklevel=2)
        return self.log_probs(x)

Note: log_softmax is idempotent (log_softmax(log_softmax(z)) == log_softmax(z)), so the existing double
call in beam_search_decode is numerically harmless -- it only costs an extra pass over (beam*B, 40000).
Likewise softmax(log_softmax(z)/T) == softmax(z/T) exactly, so fast_nucleus_sample's current temperature
handling is correct *as long as* it is followed by softmax/multinomial and not by a bare .exp().
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def step_log_probs(
    model,
    hidden: torch.Tensor,                    # (B, d) or (B, T, d) decoder_final_norm output
    temperature: float = 1.0,
    grammar_mask=None,                       # DevanagariGrammarMask
    prev_ids: Optional[torch.Tensor] = None,  # (B,) previous token ids, required with grammar_mask
    renormalize_mask: bool = False,
) -> torch.Tensor:
    """fp32 log-probs over the vocab, with the grammar mask applied.

    renormalize_mask=False (default, matches 'log_softmax then mask'): scores stay the model's true
        log-probs; disallowed tokens are -inf.  Right for beam search / MBR scoring.
    renormalize_mask=True: mask the logits, then log_softmax -> distribution over allowed tokens only.
        Right for sampling-style use where the output must sum to 1.
    Assumes ``grammar_mask.apply`` writes -inf (masked_fill) into disallowed entries; it works on
    logits and on log-probs alike.
    """
    z = model.project_logits(hidden).float()
    if temperature != 1.0:
        z = z / temperature
    if grammar_mask is not None and renormalize_mask:
        z = grammar_mask.apply(z, prev_ids)
    lp = F.log_softmax(z, dim=-1)
    if grammar_mask is not None and not renormalize_mask:
        lp = grammar_mask.apply(lp, prev_ids)
    return lp


def greedy_next(model, hidden, grammar_mask=None, prev_ids=None) -> torch.Tensor:
    """argmax on raw logits (log_softmax is monotone, so it is skipped entirely)."""
    z = model.project_logits(hidden).float()
    if grammar_mask is not None:
        z = grammar_mask.apply(z, prev_ids)
    return z.argmax(dim=-1)


def top_p_top_k_filter(logits: torch.Tensor, top_p: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """Filter on LOGITS (rows (B, V)); keeps >=1 token per row.  Returns logits with -inf elsewhere."""
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sl, si = torch.sort(logits, dim=-1, descending=True)
        cum = torch.softmax(sl, dim=-1).cumsum(dim=-1)
        drop = (cum - torch.softmax(sl, dim=-1)) >= top_p        # drop tokens whose *preceding* mass >= p
        sl = sl.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)
    return logits


def sample_from_logits(
    model, hidden, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0,
    grammar_mask=None, prev_ids=None, generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Temperature + grammar mask + top-p/top-k on raw logits, then ONE softmax inside multinomial."""
    z = model.project_logits(hidden).float() / max(temperature, 1e-6)
    if grammar_mask is not None:
        z = grammar_mask.apply(z, prev_ids)
    z = top_p_top_k_filter(z, top_p, top_k)
    return torch.multinomial(torch.softmax(z, dim=-1), 1, generator=generator).squeeze(-1)


def interpolate_knn_log_probs(lp_mt: torch.Tensor, lp_knn: torch.Tensor, lam: float) -> torch.Tensor:
    """kNN-MT: log((1-lam)*p_mt + lam*p_knn) in log space.  Needs LOG-PROBS (not raw logits) for both."""
    return torch.logaddexp(lp_mt + torch.log1p(torch.tensor(-lam, dtype=lp_mt.dtype)),
                           lp_knn + torch.log(torch.tensor(lam, dtype=lp_mt.dtype)))
