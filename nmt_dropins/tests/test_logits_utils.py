import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn, torch.nn.functional as F
from research.logits_utils import (step_log_probs, greedy_next, top_p_top_k_filter, sample_from_logits,
                                   interpolate_knn_log_probs)


class M(nn.Module):
    def __init__(s, d=8, V=30):
        super().__init__(); s.proj = nn.Linear(d, V, bias=False)
    def project_logits(s, x): return s.proj(x)


class Mask:
    def __init__(s, ban): s.ban = ban
    def apply(s, x, prev): return x.masked_fill(s.ban[None, :], float("-inf"))


def test_idempotent_and_temperature_equivalence():
    z = torch.randn(5, 30) * 3
    lp = F.log_softmax(z, -1)
    assert torch.allclose(F.log_softmax(lp, -1), lp, atol=1e-6)                   # double log_softmax is a no-op
    for T in (0.5, 0.8, 2.0):
        assert torch.allclose(torch.softmax(lp / T, -1), torch.softmax(z / T, -1), atol=1e-6)


def test_step_log_probs_and_mask_modes():
    m = M(); h = torch.randn(4, 8); ban = torch.zeros(30, dtype=torch.bool); ban[[3, 7, 9]] = True
    lp = step_log_probs(m, h, 1.0, Mask(ban), None)
    full = F.log_softmax(m.project_logits(h), -1)
    assert torch.isinf(lp[:, ban]).all() and torch.allclose(lp[:, ~ban], full[:, ~ban])       # true log-probs
    lpr = step_log_probs(m, h, 1.0, Mask(ban), None, renormalize_mask=True)
    assert torch.allclose(lpr.exp().sum(-1), torch.ones(4), atol=1e-5)
    assert (greedy_next(m, h, Mask(ban), None) == lpr.argmax(-1)).all()


def test_top_p_matches_naive():
    z = torch.randn(6, 50) * 2
    for p in (0.5, 0.9, 0.99):
        out = top_p_top_k_filter(z, top_p=p)
        for r in range(6):
            pr = torch.softmax(z[r], -1); order = pr.argsort(descending=True)
            keep, c = [], 0.0
            for i in order.tolist():
                if c >= p: break
                keep.append(i); c += pr[i].item()
            assert set(torch.isfinite(out[r]).nonzero().flatten().tolist()) == set(keep)
    k = top_k_out = top_p_top_k_filter(z, top_k=5)
    assert (torch.isfinite(k).sum(-1) == 5).all()


def test_sampling_respects_mask_and_distribution():
    m = M(); h = torch.randn(1, 8).expand(20000, 8); ban = torch.zeros(30, dtype=torch.bool); ban[:10] = True
    g = torch.Generator().manual_seed(0)
    s = sample_from_logits(m, h, 1.0, grammar_mask=Mask(ban), generator=g)
    assert (s >= 10).all()
    emp = torch.bincount(s, minlength=30).float() / len(s)
    ref = torch.softmax(m.project_logits(h[:1]).masked_fill(ban, float("-inf")), -1)[0]
    assert (emp - ref).abs().max() < 0.02


def test_knn_interp():
    a = torch.log_softmax(torch.randn(3, 20), -1); b = torch.log_softmax(torch.randn(3, 20), -1)
    out = interpolate_knn_log_probs(a, b, 0.25)
    assert torch.allclose(out.exp(), 0.75 * a.exp() + 0.25 * b.exp(), atol=1e-6)
