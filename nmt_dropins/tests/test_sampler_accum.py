import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pytest, torch, torch.nn as nn, torch.nn.functional as F
from research.dynamic_batching import TokenBucketBatchSampler
from research.accum import train_window, window_normalizer, windows
from research.chunked_losses import chunked_cross_entropy
from research.optim_groups import build_param_groups


def lens(n, seed=0):
    r = np.random.default_rng(seed)
    return np.clip(r.lognormal(3.2, 0.6, n).astype(np.int32), 2, 350)


def test_partition_constraints_and_len_cache():
    L = lens(50_000)
    s = TokenBucketBatchSampler(L, max_tokens=4096, max_items=128, megabatch_size=10_000)
    s.set_epoch(0)
    n = len(s)
    seen = np.zeros(len(L), bool)
    cnt = 0
    for b in s:
        cnt += 1
        assert 1 <= len(b) <= 128
        assert len(b) * L[b].max() <= 4096 or len(b) == 1
        assert not seen[b].any(); seen[b] = True
    assert seen.all() and cnt == n == len(s)
    assert s._plan_epoch == 0                           # plan reused by __len__/__iter__


def test_maximal_greedy():
    # each batch (except megabatch tails) cannot take one more item
    L = lens(20_000, 1)
    s = TokenBucketBatchSampler(L, 2048, max_items=64, megabatch_size=20_000, shuffle=True)
    s.set_epoch(0)
    bs = list(s)
    full = 0
    for b in bs:
        if len(b) == 64: continue
        full += 1
    assert full <= len(bs)


def test_epoch_diversity_and_determinism():
    L = lens(30_000, 2)
    a = TokenBucketBatchSampler(L, megabatch_size=5_000); b = TokenBucketBatchSampler(L, megabatch_size=5_000)
    a.set_epoch(0); b.set_epoch(0)
    assert list(a) == list(b)                           # deterministic
    e0 = {tuple(sorted(x)) for x in a}
    a.set_epoch(1)
    e1 = {tuple(sorted(x)) for x in a}
    assert len(e0 & e1) / len(e0) < 0.02                # batch membership changes across epochs
    # contrast: old global-sort behaviour would have repeated (almost) every batch
    g = TokenBucketBatchSampler(L, shuffle=False)
    assert list(g) == list(g)


def test_efficiency_vs_global_sort_and_random():
    L = lens(400_000, 3)
    mega = TokenBucketBatchSampler(L, 4096, megabatch_size=100_000); mega.set_epoch(0)
    glob = TokenBucketBatchSampler(L, 4096, shuffle=False)
    e_m, e_g = mega.padding_efficiency(), glob.padding_efficiency()
    print(f"\nefficiency megabatch={e_m:.4f} global_sort={e_g:.4f}")
    assert e_g - e_m < 0.01 and e_m > 0.9
    small = TokenBucketBatchSampler(L, 4096, megabatch_size=2_000); small.set_epoch(0)
    print(f"efficiency megabatch=2k  ={small.padding_efficiency():.4f}")


def test_resume_start_batch():
    L = lens(10_000, 4)
    s = TokenBucketBatchSampler(L, megabatch_size=2_000); s.set_epoch(3)
    full = list(s)
    s.set_epoch(3, start_batch=7)
    assert list(s) == full[7:]
    assert list(s) == full                               # skip applies once


def test_expected_len_stable():
    L = lens(30_000, 5)
    s = TokenBucketBatchSampler(L, megabatch_size=5_000)
    s.set_epoch(5)
    e = s.expected_len
    s.set_epoch(0)
    assert len(s) == e


# ---------------- accumulation equivalence ----------------
class Stub(nn.Module):
    def __init__(s, V=60, d=16):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.proj = nn.Linear(d, V, bias=False); s.proj.weight = s.emb.weight
        s.body = nn.Linear(d, d)
    def encode(s, src, m=None): return s.emb(src)
    def decode(s, enc, m, tgt, tm=None): return torch.tanh(s.body(s.emb(tgt) + enc.mean(1, keepdim=True)))


def mk_batches(sizes, V=60):
    g = torch.Generator().manual_seed(0)
    out = []
    for T in sizes:
        t = torch.randint(1, V, (4, T), generator=g)
        t[:, T // 2:] = 0 if T > 4 else t[:, T // 2:]
        out.append({"src": torch.randint(1, V, (4, 5), generator=g), "tgt_in": torch.randint(1, V, (4, T), generator=g), "tgt_out": t})
    return out


def test_accum_equals_big_batch_and_old_scheme_is_biased():
    torch.manual_seed(0)
    m1 = Stub(); m2 = copy.deepcopy(m1); m3 = copy.deepcopy(m1)
    batches = mk_batches([40, 6, 22, 9])                # very different token counts
    opt = torch.optim.SGD(m1.parameters(), lr=0.0)
    # new scheme (lr=0 so params stay put; we compare grads)
    train_window(m1, batches, opt, device=torch.device("cpu"), autocast_dtype=None, z_loss=0.0, max_grad_norm=None)
    # reference: ONE big batch with token-mean loss
    def hid(m, b): return m.decode(m.encode(b["src"]), None, b["tgt_in"])
    h = [hid(m2, b) for b in batches]
    tgt = [b["tgt_out"] for b in batches]
    # pad-concat along time impossible (different T) -> sum of per-token losses / total tokens
    tot = sum(float((t != 0).sum()) for t in tgt)
    L = sum(chunked_cross_entropy(hh, m2.proj.weight, t, 0.1, 0, "sum") for hh, t in zip(h, tgt)) / tot
    L.backward()
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6), (a.grad - b.grad).abs().max()
    # old scheme: mean per micro-batch / ga  -> differs
    ga = len(batches)
    Lold = sum(chunked_cross_entropy(hid(m3, b), m3.proj.weight, b["tgt_out"], 0.1, 0, "mean") for b in batches) / ga
    Lold.backward()
    diff = max((a.grad - c.grad).abs().max().item() for a, c in zip(m2.parameters(), m3.parameters()))
    print("\nold-vs-new grad max diff:", diff)
    assert diff > 1e-3


def test_window_helpers_and_param_groups():
    bs = mk_batches([10, 10, 10])
    assert list(map(len, windows(bs, 2))) == [2, 1]
    n = window_normalizer(bs)
    assert n == sum(float((b["tgt_out"] != 0).sum()) for b in bs)
    m = Stub()
    g = build_param_groups(m, 0.01)
    ids = [id(p) for grp in g for p in grp["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert id(m.emb.weight) in [id(p) for p in g[1]["params"]] and g[1]["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in g[0]["params"]) and any(p is m.body.weight for p in g[0]["params"])


def test_nonfinite_grad_step_is_skipped():
    torch.manual_seed(0)
    m = Stub()
    with torch.no_grad(): m.body.weight[0, 0] = float("nan")          # poison -> NaN loss/grad
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    out = train_window(m, mk_batches([8, 8]), opt, device=torch.device("cpu"), autocast_dtype=None)
    assert out.get("skipped") == 1.0
    for p, b in zip(m.parameters(), before):
        assert torch.allclose(p, b, equal_nan=True)                     # no update applied
    assert len(opt.state) == 0                                          # AdamW moments never touched
