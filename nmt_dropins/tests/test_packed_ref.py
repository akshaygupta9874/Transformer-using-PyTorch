import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest, torch, torch.nn.functional as F
from research.packed_ref import (doc_ids_from_lengths, positions_from_doc_ids, packed_self_mask,
                                 packed_cross_mask, build_packed_batch, expand_doc_weights,
                                 check_packed_equivalence)

D, H, V = 16, 2, 40
torch.manual_seed(0)


def rope(x, pos):                                         # x (B,H,T,dk), pos (B,T)
    dk = x.shape[-1]; half = dk // 2
    fr = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32) / half))
    ang = pos[:, None, :, None].float() * fr
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class ToyAttn(torch.nn.Module):
    """enc self-attn (RoPE) -> dec self-attn (RoPE, causal) -> cross-attn (no RoPE); one layer each."""
    def __init__(s):
        super().__init__()
        s.emb = torch.nn.Embedding(V, D)
        s.p = torch.nn.ModuleList([torch.nn.Linear(D, 3 * D, bias=False) for _ in range(3)])
    def attn(s, i, xq, xkv, mask, pq=None, pk=None):
        B, Tq, _ = xq.shape; Tk = xkv.shape[1]
        W = s.p[i].weight
        q = (xq @ W[:D].t()).view(B, Tq, H, -1).transpose(1, 2)
        k = (xkv @ W[D:2 * D].t()).view(B, Tk, H, -1).transpose(1, 2)
        v = (xkv @ W[2 * D:].t()).view(B, Tk, H, -1).transpose(1, 2)
        if pq is not None: q, k = rope(q, pq), rope(k, pk)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return o.transpose(1, 2).reshape(B, Tq, D)
    def forward(s, src, tgt, src_pos, tgt_pos, enc_mask, dec_mask, cross_mask):
        xs, xt = s.emb(src), s.emb(tgt)
        e = xs + s.attn(0, xs, xs, enc_mask, src_pos, src_pos)
        d = xt + s.attn(1, xt, xt, dec_mask, tgt_pos, tgt_pos)
        return d + s.attn(2, d, e, cross_mask)


def adapters(m):
    def single(src, tgt):
        b = build_packed_batch([[src]], [[tgt]], len(src), len(tgt))          # 1 doc, no padding
        return m(b["src"], b["tgt"], b["src_pos"], b["tgt_pos"], b["enc_mask"], b["dec_mask"], b["cross_mask"])[0]
    def packed(b):
        return m(b["src"], b["tgt"], b["src_pos"], b["tgt_pos"], b["enc_mask"], b["dec_mask"], b["cross_mask"])
    return single, packed


def make_docs(n=5):
    g = torch.Generator().manual_seed(1)
    return [(torch.randint(1, V, (int(torch.randint(3, 9, (1,), generator=g)),), generator=g),
             torch.randint(1, V, (int(torch.randint(2, 8, (1,), generator=g)),), generator=g)) for _ in range(n)]


def test_doc_ids_and_positions():
    doc = doc_ids_from_lengths([[3, 2], [4]], 7)
    assert doc.tolist() == [[1, 1, 1, 2, 2, 0, 0], [1, 1, 1, 1, 0, 0, 0]]
    assert positions_from_doc_ids(doc).tolist() == [[0, 1, 2, 0, 1, 0, 0], [0, 1, 2, 3, 0, 0, 0]]


def test_masks_properties():
    sd = doc_ids_from_lengths([[3, 2], [5]], 7); td = doc_ids_from_lengths([[2, 2], [1]], 5)
    for causal in (False, True):
        m = packed_self_mask(sd, causal)
        assert m.any(-1).all()                               # no fully masked row, ever
        assert not m[0, 0, 0, 3]                             # doc1 cannot see doc2
        assert not m[0, 0, 0, 5] and not m[0, 0, 5, 0]       # real<->pad never
    assert not packed_self_mask(sd, True)[0, 0, 0, 1]        # causal
    c = packed_cross_mask(td, sd)
    assert c.any(-1).all()
    assert c[0, 0, 0, :3].all() and not c[0, 0, 0, 3:].any()  # tgt doc1 -> src doc1 only
    assert c[1, 0, 0, :5].all()
    full = packed_cross_mask(doc_ids_from_lengths([[2]], 4), doc_ids_from_lengths([[4]], 4))   # pad query, full src
    assert full.any(-1).all()


def test_packed_equals_unpacked_with_padding_and_multi_doc():
    m = ToyAttn(); single, packed = adapters(m); docs = make_docs(5)
    with torch.no_grad():
        worst = check_packed_equivalence(single, packed, docs, [[0, 1, 2], [3], [4]], atol=1e-5)
    print("\nmax abs diff packed vs unpacked:", worst)


def test_negative_control_detects_leaky_mask_and_global_positions():
    m = ToyAttn(); single, _ = adapters(m); docs = make_docs(4)
    def leaky(b):          # plain causal / full masks across docs + global positions = the classic packing bug
        T, S = b["tgt"].shape[1], b["src"].shape[1]
        enc = torch.ones(1, 1, S, S, dtype=torch.bool).expand(b["src"].shape[0], 1, S, S)
        dec = torch.ones(T, T, dtype=torch.bool).tril()[None, None].expand(b["tgt"].shape[0], 1, T, T)
        crs = torch.ones(b["tgt"].shape[0], 1, T, S, dtype=torch.bool)
        gp = lambda x: torch.arange(x.shape[1]).expand_as(x)
        return m(b["src"], b["tgt"], gp(b["src"]), gp(b["tgt"]), enc, dec, crs)
    with torch.no_grad(), pytest.raises(AssertionError):
        check_packed_equivalence(single, leaky, docs, [[0, 1, 2]], atol=1e-5)
    # dropping src_pos/tgt_pos only (the R-Drop second-pass bug) must also be caught
    def no_pos(b):
        z = lambda x: torch.zeros_like(x)
        return m(b["src"], b["tgt"], z(b["src_pos"]), z(b["tgt_pos"]), b["enc_mask"], b["dec_mask"], b["cross_mask"])
    with torch.no_grad(), pytest.raises(AssertionError):
        check_packed_equivalence(single, no_pos, docs, [[0, 1, 2]], atol=1e-5)


def test_padding_rows_give_finite_grads():
    m = ToyAttn(); docs = make_docs(3)
    b = build_packed_batch([[docs[0][0]], [docs[1][0], docs[2][0]]], [[docs[0][1]], [docs[1][1], docs[2][1]]], 40, 40)
    out = m(b["src"], b["tgt"], b["src_pos"], b["tgt_pos"], b["enc_mask"], b["dec_mask"], b["cross_mask"])
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_expand_doc_weights():
    doc = doc_ids_from_lengths([[2, 1], [3]], 4)
    w = torch.tensor([[0., 0.5, 2.0], [0., 1.5, 0.]])
    assert expand_doc_weights(doc, w).tolist() == [[0.5, 0.5, 2.0, 0.0], [1.5, 1.5, 1.5, 0.0]]
