import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest, torch
from research.eval_tools import (canon_digits, copy_preservation, corpus_copy_preservation, hash_split,
                                 evaluate_benchmark, write_samples_tsv)
from research.profile_sweep import find_max_tokens, sweep, synthetic_batch, run_once
from test_sampler_accum import Stub


def test_copy_preservation_scripts_and_grouping():
    assert canon_digits("१२३४५") == "12345"
    assert copy_preservation("Pay 1,00,000 rupees", "१००००० रुपये दें") == (1, 1)         # lakh grouping + script
    assert copy_preservation("It costs 3.5 dollars", "इसकी कीमत ३.५ डॉलर है") == (1, 1)
    assert copy_preservation("Call 112 or 100", "112 पर कॉल करें") == (2, 1)
    assert copy_preservation("112 and 112", "112") == (2, 1)                                # multiset
    assert copy_preservation("See https://Example.com/a.", "देखें https://example.com/a") == (1, 1)
    assert copy_preservation("visit www.x.org/p123", "वेबसाइट www.x.org/p123 देखें") == (1, 1)  # digits inside URL not a number
    assert copy_preservation("no items here", "कुछ नहीं") == (0, 0)
    r = corpus_copy_preservation(["a 5", "b 6 7"], ["५", "६"])
    assert r["items"] == 3 and abs(r["copy_rate"] - 2 / 3) < 1e-9 and r["sent_exact"] == 0.5


def test_hash_split_is_stable_when_dataset_grows():
    big = [f"sentence number {i}" for i in range(200_000)]
    small = big[:50_000]
    mb = {s for s in big if hash_split(s, 2)}; ms = {s for s in small if hash_split(s, 2)}
    assert ms == {s for s in mb if s in set(small)}                     # membership independent of dataset size
    assert 0.0012 < len(mb) / len(big) < 0.0028                          # ~2 permille
    assert [hash_split(s) for s in small[:100]] == [hash_split(s) for s in reversed(small[:100][::-1])]


def test_evaluate_benchmark_and_smoke_detection(tmp_path):
    p = tmp_path / "b.tsv"
    p.write_text("a 1\tx\nb 2\ty\nc 3\tz\n", encoding="utf-8")
    calls = []
    def tr(batch): calls.append(len(batch)); return [s.replace("a", "") for s in batch]
    with pytest.warns(UserWarning, match="smoke"):
        out = evaluate_benchmark(tr, str(p), {"len": lambda h, r: float(sum(map(len, h)))}, batch_size=2)
    assert out["is_smoke"] and out["n"] == 3 and calls == [2, 1] and "len" in out and out["items"] == 3
    big = tmp_path / "big.tsv"
    big.write_text("".join(f"s{i} {i}\tr\n" for i in range(150)), encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = evaluate_benchmark(lambda b: list(b), str(big))
    assert not out["is_smoke"] and out["copy_rate"] == 1.0
    write_samples_tsv(str(tmp_path / "s.tsv"), ["a 1"], ["१"], ["r"])
    assert (tmp_path / "s.tsv").read_text(encoding="utf-8").splitlines()[1].endswith("\t1")


def test_find_max_tokens_bisection():
    limit = 9000
    seen = []
    def fits(t): seen.append(t); return t <= limit
    got = find_max_tokens(fits, lo=1024, hi=32768, granularity=256)
    assert got <= limit and limit - got < 256 and got % 256 == 0
    assert find_max_tokens(lambda t: False) == 0


def test_sweep_cpu_and_oom_handling():
    def build(**kw):
        if kw.get("boom"): raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return Stub(V=50)
    batch_fn = lambda mt: {"src": torch.randint(1, 50, (4, 5)), "tgt_in": torch.randint(1, 50, (4, 6)),
                           "tgt_out": torch.randint(1, 50, (4, 6))}
    rows = sweep(build, [{}, {"boom": True}], 64, device=torch.device("cpu"), batch_fn=batch_fn,
                 steps=3, warmup=1, autocast_dtype=None)
    assert rows[0]["step_ms"] > 0 and rows[0]["peak_alloc_gb"] is None and rows[1]["oom"] is True
    assert synthetic_batch(4096, 40, 40)["src"].shape == (102, 40)
