import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, pytest, torch
import fake_tok
from scripts.build_memmap_stream import build_memmap, iter_tsv
from dataset_cache import CachedBilingualDataset

KW = dict(tok_factory="fake_tok:load", subst="fake_tok:subst", seq_len=60)

def make_pairs(n, seed=0):
    r = random.Random(seed); out = []
    for i in range(n):
        ls, lt = r.randint(0, 150), r.randint(1, 80)
        s = "".join(r.choice("abcdef 1हिन्दी") for _ in range(ls)); t = "".join(r.choice("xyz1 अ") for _ in range(lt))
        out.append((s, t, r.random()))
    return out

@pytest.fixture(scope="module")
def tok_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("tok"); a, b = d / "src.json", d / "tgt.json"
    a.write_text("{}"); b.write_text('{"x":1}'); return str(a), str(b)

def reference(pairs, cfg_L=60, mode="base", p=0.0):
    st, tt = fake_tok.load("a"), fake_tok.load("b"); S, T, Q = [], [], []
    for s, t, q in pairs:
        s, t = fake_tok.subst(s, t)
        si = st.encode(s, False, True).ids; ti = tt.encode(t, True, True).ids
        if 1 <= len(si) <= cfg_L and 1 <= len(ti) <= cfg_L: S.append(si); T.append(ti); Q.append(q)
    return S, T, Q

def load_out(d, name="src", off="src_offsets"):
    o = np.load(os.path.join(d, off + ".npy")); b = np.fromfile(os.path.join(d, name + ".bin"), dtype=np.uint16)
    return [b[o[i]:o[i + 1]].tolist() for i in range(len(o) - 1)]

def test_matches_reference_and_worker_count_invariance(tmp_path, tok_files):
    pairs = make_pairs(4000)
    S, T, Q = reference(pairs)
    outs = []
    for w in (1, 3):
        d = str(tmp_path / f"w{w}")
        meta = build_memmap(iter(pairs), d, src_tok=tok_files[0], tgt_tok=tok_files[1], workers=w,
                            chunk_pairs=333, write_quality=True, **KW)
        assert meta["kept"] == len(S) and meta["n_in"] == 4000 and meta["dropped"] == 4000 - len(S)
        assert load_out(d) == S and load_out(d, "tgt", "tgt_offsets") == T
        assert np.allclose(np.load(os.path.join(d, "quality.npy")), np.array(Q, np.float32))
        outs.append([open(os.path.join(d, f), "rb").read() for f in ("src.bin", "tgt.bin")])
        assert not [f for f in os.listdir(d) if f.endswith(".partial")]
    assert outs[0] == outs[1]                           # byte-identical regardless of worker count

def test_srcdrop_shard_aligned_deterministic(tmp_path, tok_files):
    pairs = make_pairs(2500, 1); base = str(tmp_path / "b")
    build_memmap(iter(pairs), base, src_tok=tok_files[0], tgt_tok=tok_files[1], workers=2, chunk_pairs=200, **KW)
    res = []
    for w in (1, 2):
        build_memmap(iter(pairs), base, src_tok=tok_files[0], tgt_tok=tok_files[1], workers=w, chunk_pairs=200,
                     mode="srcdrop", srcdrop_p=0.3, srcdrop_tag=f"v{w}", **KW)
        res.append(load_out(base, "src.v%d" % w, "src_offsets.v%d" % w))
    assert res[0] == res[1]
    b = load_out(base)
    assert len(res[0]) == len(b)                        # same rows as base shard
    assert all(len(x) <= 60 for x in res[0]) and sum(map(len, res[0])) > sum(map(len, b))   # dropout -> longer
    assert any(x != y for x, y in zip(res[0], b))

def test_backpressure_bounded_inflight(tmp_path, tok_files):
    consumed = {"n": 0}
    def gen():
        for i in range(60_000):
            consumed["n"] = i; yield ("ab" * 5, "xy" * 5, None)
    meta = build_memmap(gen(), str(tmp_path / "bp"), src_tok=tok_files[0], tgt_tok=tok_files[1], workers=2,
                        chunk_pairs=1000, max_inflight=4, **KW)
    assert meta["max_inflight"] <= 4 and meta["kept"] == 60_000

def test_cached_dataset(tmp_path, tok_files):
    pairs = make_pairs(1500, 2); tsv = tmp_path / "d.tsv"
    tsv.write_text("".join(f"{s.replace(chr(9),' ')}\t{t.replace(chr(9),' ')}\n" for s, t, _ in pairs), encoding="utf-8")
    kw = dict(src_tok=tok_files[0], tgt_tok=tok_files[1], workers=2, chunk_pairs=300, **KW)
    ds = CachedBilingualDataset(str(tsv), str(tmp_path / "c"), **kw)
    assert ds.built
    S, T, _ = reference([(s.replace("\t", " "), t.replace("\t", " "), 0) for s, t, _ in pairs])
    assert len(ds) == len(S)
    for i in (0, 7, len(S) - 1):
        a, b = ds[i]; assert a.tolist() == S[i] and b.tolist() == T[i] and a.dtype == torch.int64
    assert (ds.lengths == np.maximum([len(x) for x in S], [len(x) for x in T])).all()
    ds2 = CachedBilingualDataset(str(tsv), str(tmp_path / "c"), **kw)
    assert not ds2.built                                 # cache hit: nothing re-tokenised
    ds3 = CachedBilingualDataset(str(tsv), str(tmp_path / "c"), **{**kw, "seq_len": 30})
    assert ds3.built and len(ds3) <= len(ds)             # key changed -> rebuilt
    from research.dynamic_batching import TokenBucketBatchSampler
    sm = TokenBucketBatchSampler(ds.lengths, 512, megabatch_size=500); sm.set_epoch(0)
    assert sum(len(b) for b in sm) == len(ds)
