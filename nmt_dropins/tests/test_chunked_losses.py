import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import torch.nn.functional as F

from research.chunked_losses import (chunked_cross_entropy, chunked_rdrop_loss, chunked_nll_sum,
                                     chunked_token_logprobs, _ChunkedLinearCE, _ChunkedRDrop)

torch.manual_seed(0)


def make(B=3, T=7, d=16, V=50, dtype=torch.float64, pad_frac=0.3):
    h1 = torch.randn(B, T, d, dtype=dtype, requires_grad=True)
    h2 = (h1.detach() + 0.3 * torch.randn(B, T, d, dtype=dtype)).requires_grad_(True)
    W = (0.3 * torch.randn(V, d, dtype=dtype)).requires_grad_(True)
    y = torch.randint(1, V, (B, T))
    y[torch.rand(B, T) < pad_frac] = 0
    y[0, 0] = 5                     # guarantee at least one valid token
    return h1, h2, W, y


def naive_ce(h, W, y, eps, zl, w=None):
    z = h.reshape(-1, h.shape[-1]) @ W.t()
    yy = y.reshape(-1)
    m = yy != 0
    z, yy = z[m], yy[m]
    lse = torch.logsumexp(z, -1)
    l = F.cross_entropy(z, yy, label_smoothing=eps, reduction="none") + zl * lse ** 2
    if w is not None:
        l = l * w.expand_as(y).reshape(-1)[m]
    return l


def grads(loss, *ts):
    return torch.autograd.grad(loss, ts, allow_unused=False)


@pytest.mark.parametrize("eps,zl,cs", [(0.0, 0.0, 4), (0.1, 0.0, 5), (0.1, 1e-3, 1024), (0.2, 5e-2, 3)])
def test_ce_matches_naive(eps, zl, cs):
    h, _, W, y = make()
    ref = naive_ce(h, W, y, eps, zl).sum()
    got = chunked_cross_entropy(h, W, y, eps, 0, "sum", chunk_size=cs, z_loss=zl)
    assert torch.allclose(ref, got, atol=1e-10)
    gr = grads(ref, h, W)
    gg = grads(got, h, W)
    for a, b in zip(gr, gg):
        assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_ce_mean_and_pytorch_equivalence():
    h, _, W, y = make()
    z = (h.reshape(-1, 16) @ W.t())
    ref = F.cross_entropy(z, y.reshape(-1), label_smoothing=0.1, ignore_index=0)   # torch mean over non-ignored
    got = chunked_cross_entropy(h, W, y, 0.1, 0, "mean", chunk_size=4)
    assert torch.allclose(ref, got, atol=1e-10)


def test_sample_weight_and_normalizer():
    h, _, W, y = make()
    B, T = y.shape
    sw = torch.rand(B, dtype=torch.float64) + 0.5
    l = naive_ce(h, W, y, 0.1, 1e-3, sw[:, None])
    wtok = sw[:, None].expand_as(y).reshape(-1)[(y.reshape(-1) != 0)]
    ref_mean = l.sum() / wtok.sum()
    got = chunked_cross_entropy(h, W, y, 0.1, 0, "mean", sample_weight=sw, chunk_size=6, z_loss=1e-3)
    assert torch.allclose(ref_mean, got, atol=1e-10)
    got = chunked_cross_entropy(h, W, y, 0.1, 0, "sum", sample_weight=sw, chunk_size=6, z_loss=1e-3, normalizer=17.0)
    assert torch.allclose(l.sum() / 17.0, got, atol=1e-10)
    g1 = grads(l.sum() / 17.0, h, W)
    g2 = grads(got, h, W)
    for a, b in zip(g1, g2):
        assert torch.allclose(a, b, atol=1e-10)


def test_gradcheck_float64():
    h, _, W, y = make(B=2, T=3, d=5, V=11)
    yv = y.reshape(-1)
    yv = yv[yv != 0]
    x = h.detach().reshape(-1, 5)[: len(yv)].clone().requires_grad_(True)
    f = lambda x_, W_: _ChunkedLinearCE.apply(x_, W_, yv, 0.1, 1e-2, 2, torch.float64)[0]
    assert torch.autograd.gradcheck(f, (x, W), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("alpha,cew,eps,zl,cs", [(0.4, 0.5, 0.1, 0.0, 4), (1.0, 1.0, 0.1, 1e-3, 1000), (0.3, 0.5, 0.0, 0.0, 3)])
def test_rdrop_matches_naive(alpha, cew, eps, zl, cs):
    h1, h2, W, y = make()
    d = 16
    z1, z2 = h1.reshape(-1, d) @ W.t(), h2.reshape(-1, d) @ W.t()
    m = y.reshape(-1) != 0
    z1, z2, yy = z1[m], z2[m], y.reshape(-1)[m]
    lp1, lp2 = F.log_softmax(z1, -1), F.log_softmax(z2, -1)
    kl12 = F.kl_div(lp2, lp1, log_target=True, reduction="none").sum(-1)      # KL(p1||p2)
    kl21 = F.kl_div(lp1, lp2, log_target=True, reduction="none").sum(-1)
    ce = lambda z: F.cross_entropy(z, yy, label_smoothing=eps, reduction="none") + zl * torch.logsumexp(z, -1) ** 2
    ref = (cew * (ce(z1) + ce(z2)) + alpha * 0.5 * (kl12 + kl21)).sum()
    got = chunked_rdrop_loss(h1, h2, W, y, eps, alpha, 0, "sum", chunk_size=cs, z_loss=zl, ce_weight=cew)
    assert torch.allclose(ref, got, atol=1e-9), (ref, got)
    gr = grads(ref, h1, h2, W)
    gg = grads(got, h1, h2, W)
    for a, b in zip(gr, gg):
        assert torch.allclose(a, b, atol=1e-9), (a - b).abs().max()


def test_rdrop_gradcheck():
    h1, h2, W, y = make(B=2, T=3, d=4, V=9)
    yv = y.reshape(-1); yv = yv[yv != 0]
    n = len(yv)
    x1 = h1.detach().reshape(-1, 4)[:n].clone().requires_grad_(True)
    x2 = h2.detach().reshape(-1, 4)[:n].clone().requires_grad_(True)
    f = lambda a, b, w: _ChunkedRDrop.apply(a, b, w, yv, 0.1, 1e-2, 0.5, 0.5, 2, torch.float64)[0]
    assert torch.autograd.gradcheck(f, (x1, x2, W), eps=1e-6, atol=1e-6)


def test_rdrop_zero_when_identical_and_info():
    h1, _, W, y = make()
    out, info = chunked_rdrop_loss(h1, h1.detach().clone().requires_grad_(True), W, y, 0.1, 5.0, return_info=True)
    assert info["symkl"].abs() < 1e-12


def test_bf16_close_to_fp32_reference():
    B, T, d, V = 4, 16, 64, 500
    h = torch.randn(B, T, d, requires_grad=True)
    W = (0.1 * torch.randn(V, d)).requires_grad_(True)
    y = torch.randint(1, V, (B, T)); y[:, -3:] = 0
    ref = chunked_cross_entropy(h, W, y, 0.1, 0, "mean", chunk_size=7, compute_dtype=torch.float32)
    gh_r, gW_r = grads(ref, h, W)
    got = chunked_cross_entropy(h.bfloat16(), W, y, 0.1, 0, "mean", chunk_size=7, compute_dtype=torch.bfloat16)
    assert abs(ref.item() - got.item()) / ref.item() < 2e-2
    hb = h.detach().bfloat16().requires_grad_(True)
    got = chunked_cross_entropy(hb, W, y, 0.1, 0, "mean", chunk_size=7, compute_dtype=torch.bfloat16)
    gh, gW = grads(got, hb, W)
    assert gW.dtype == torch.float32 and gh.dtype == torch.bfloat16
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    assert rel(gh, gh_r) < 5e-2 and rel(gW, gW_r) < 5e-2


def test_all_pad_returns_graph_connected_zero():
    h, _, W, y = make()
    y.zero_()
    out = chunked_cross_entropy(h, W, y)
    out.backward()
    assert out.item() == 0.0 and h.grad is not None


def test_nll_and_token_logprobs():
    h, _, W, y = make()
    s, n = chunked_nll_sum(h, W, y, chunk_size=5)
    z = h.reshape(-1, 16) @ W.t()
    ref = F.cross_entropy(z, y.reshape(-1), ignore_index=0, reduction="sum")
    assert torch.allclose(s, ref) and n == (y != 0).sum()
    yy = y.clone(); yy[yy == 0] = 1
    lp = chunked_token_logprobs(h, W, yy, chunk_size=4)
    ref = F.log_softmax(z, -1).gather(1, yy.reshape(-1, 1)).reshape(yy.shape)
    assert torch.allclose(lp, ref, atol=1e-10)
    g_ref = grads(ref.sum(), h, W); g = grads(lp.sum(), h, W)
    for a, b in zip(g_ref, g):
        assert torch.allclose(a, b, atol=1e-10)
