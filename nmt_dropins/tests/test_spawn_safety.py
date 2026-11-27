"""Windows-spawn safety checks.  Copy `assert_spawn_safe` into research/tests and call it with YOUR
MemmapPairs / PackedMemmapCollate / TokenBucketBatchSampler."""
import os, sys, pickle, random
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, torch
from torch.utils.data import DataLoader
from dataset_cache import CachedBilingualDataset
from research.dynamic_batching import TokenBucketBatchSampler


def pad_collate(items):                       # module-level => picklable
    s = torch.nn.utils.rnn.pad_sequence([a for a, _ in items], batch_first=True)
    t = torch.nn.utils.rnn.pad_sequence([b for _, b in items], batch_first=True)
    return s, t


def assert_spawn_safe(dataset, collate_fn, batch_sampler, max_pickle_mb=5.0, epochs=2, num_workers=2):
    """1) every object that crosses the process boundary pickles small (np.memmap pickles BY VALUE!)
       2) a real spawn DataLoader with persistent workers yields identical batches to in-process loading,
          including after batch_sampler.set_epoch(1)."""
    for name, obj in (("dataset", dataset), ("collate_fn", collate_fn)):
        mb = len(pickle.dumps(obj)) / 1e6
        assert mb < max_pickle_mb, f"{name} pickles to {mb:.1f} MB (memmap copied by value?)"
        pickle.loads(pickle.dumps(obj))
    ref = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0)
    dl = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=num_workers,
                    multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    for e in range(epochs):
        batch_sampler.set_epoch(e)
        a = [tuple(x.clone() for x in b) for b in ref]
        b = [tuple(x.clone() for x in bb) for bb in dl]
        assert len(a) == len(b) == len(batch_sampler)
        assert all(torch.equal(x, y) for p, q in zip(a, b) for x, y in zip(p, q))
    del dl


def test_cached_dataset_spawn_safe(tmp_path):
    r = random.Random(0); tsv = tmp_path / "d.tsv"
    tsv.write_text("".join("".join(r.choice("abcdef 1") for _ in range(r.randint(5, 60))) + "\t" +
                           "".join(r.choice("xyz1 ") for _ in range(r.randint(5, 50))) + "\n" for _ in range(3000)), encoding="utf-8")
    (tmp_path / "a.json").write_text("{}"); (tmp_path / "b.json").write_text("{}")
    ds = CachedBilingualDataset(str(tsv), str(tmp_path / "c"), src_tok=str(tmp_path / "a.json"), tgt_tok=str(tmp_path / "b.json"),
                                tok_factory="fake_tok:load", subst="fake_tok:subst", seq_len=80, workers=1)
    bs = TokenBucketBatchSampler(ds.lengths, 512, megabatch_size=1000)
    assert_spawn_safe(ds, pad_collate, bs)
    ds.close()
    assert ds.src is None
