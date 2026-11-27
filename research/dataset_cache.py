"""dataset_cache.py -- cached BilingualDataset for runs without a pre-built ``memmap_dir``.

substitute_parallel + tokenizer.encode run ONCE (multi-process, via scripts/build_memmap_stream.py) and the
token streams are kept on disk as uint16 memmaps.  Later runs reuse them when the cache key matches
(tokenizer files, flags, seq_len, input fingerprint).  ``__getitem__`` is two slices; ``lengths`` is a
ready int32 array for TokenBucketBatchSampler (one vectorised diff, no per-item Python).

If you already have MemmapPairs reading this on-disk format, simply point it at ``cache_dir`` instead.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.build_memmap_stream import (META_JSON, SRC_BIN, SRC_OFF, TGT_BIN, TGT_OFF, build_memmap,
                                         iter_tsv, _sha)


def _fingerprint_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        st = os.stat(p)
        h.update(f"{os.path.abspath(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


class CachedBilingualDataset(Dataset):
    _ACTIVE_INSTANCES: list[CachedBilingualDataset] = []

    def __init__(
        self,
        pairs: Iterable | str | list,          # iterable of (src, tgt[, q]) OR path(s) to a TSV
        cache_dir: str,
        *,
        src_tok: str, tgt_tok: str, tok_factory: str,
        subst: str = "research.placeholders:substitute_parallel",
        use_placeholders: bool = True,
        seq_len: int = 350,
        src_bos=False, src_eos=True, tgt_bos=True, tgt_eos=True,   # ADAPT to old BilingualDataset.__getitem__
        data_id: Optional[str] = None,          # REQUIRED when `pairs` is an iterator (what data is this?)
        length_mode: str = "max",               # "max" | "sum"  (what your sampler used for lengths[i])
        workers: int = 2, chunk_pairs: int = 20_000, rebuild: bool = False, limit: Optional[int] = None,
    ):
        self.cache_dir = os.path.abspath(cache_dir)
        self.length_mode = length_mode
        paths = [pairs] if isinstance(pairs, str) else (pairs if isinstance(pairs, list) and pairs
                                                       and isinstance(pairs[0], str) else None)
        if paths is not None:
            data_id = data_id or _fingerprint_files(paths)
        elif data_id is None:
            raise ValueError("pass data_id=... when `pairs` is an iterator so the cache can be keyed")
        key_obj = dict(data=data_id, src_sha=_sha(src_tok), tgt_sha=_sha(tgt_tok), seq_len=seq_len,
                       ph=use_placeholders, flags=[src_bos, src_eos, tgt_bos, tgt_eos], limit=limit,
                       subst=subst, factory=tok_factory)
        key = hashlib.sha256(json.dumps(key_obj, sort_keys=True).encode()).hexdigest()[:16]
        meta_path = os.path.join(cache_dir, META_JSON)
        fresh = False
        if not rebuild and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                fresh = json.load(f).get("cache_key") == key
        self.built = not fresh
        if not fresh:
            import gc
            for inst in list(CachedBilingualDataset._ACTIVE_INSTANCES):
                if getattr(inst, "cache_dir", None) == self.cache_dir:
                    inst.close()
            gc.collect()
            it = iter_tsv(paths) if paths is not None else iter(pairs)
            meta = build_memmap(it, cache_dir, src_tok=src_tok, tgt_tok=tgt_tok, tok_factory=tok_factory,
                                subst=subst, use_placeholders=use_placeholders, seq_len=seq_len,
                                src_bos=src_bos, src_eos=src_eos, tgt_bos=tgt_bos, tgt_eos=tgt_eos,
                                workers=workers, chunk_pairs=chunk_pairs, limit=limit)
            meta["cache_key"] = key
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        self._open_arrays()
        sl = np.diff(self.src_off).astype(np.int32)
        tl = np.diff(self.tgt_off).astype(np.int32)
        self.src_lengths, self.tgt_lengths = sl, tl
        self.lengths = np.maximum(sl, tl) if length_mode == "max" else sl + tl
        CachedBilingualDataset._ACTIVE_INSTANCES.append(self)

    # ---- spawn / Windows safety -------------------------------------------------------------
    # np.memmap pickles BY VALUE (the whole file is copied into the pickle), so a default pickle of this
    # object sent to every spawn DataLoader worker would copy gigabytes.  Ship only paths + flags and
    # re-open the memmaps inside the worker.  Lengths stay in the main process (the sampler lives there).
    _ARRAY_ATTRS = ("src", "tgt", "src_off", "tgt_off", "src_lengths", "tgt_lengths", "lengths")

    def _open_arrays(self) -> None:
        cd = self.cache_dir
        self.src_off = np.load(os.path.join(cd, SRC_OFF))
        self.tgt_off = np.load(os.path.join(cd, TGT_OFF))
        self.src = np.memmap(os.path.join(cd, SRC_BIN), dtype=np.uint16, mode="r")
        self.tgt = np.memmap(os.path.join(cd, TGT_BIN), dtype=np.uint16, mode="r")

    def __getstate__(self):
        st = self.__dict__.copy()
        for k in self._ARRAY_ATTRS:
            st.pop(k, None)
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._open_arrays()

    def close(self) -> None:
        """Release file handles (Windows cannot delete/replace a file that is still memory-mapped)."""
        for k in ("src", "tgt"):
            a = getattr(self, k, None)
            m = getattr(a, "_mmap", None)
            if m is not None:
                try:
                    m.close()
                except Exception:
                    pass
            setattr(self, k, None)
        if self in CachedBilingualDataset._ACTIVE_INSTANCES:
            try:
                CachedBilingualDataset._ACTIVE_INSTANCES.remove(self)
            except ValueError:
                pass

    def __len__(self) -> int:
        return len(self.src_off) - 1

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.src_off[i], self.src_off[i + 1]
        c, d = self.tgt_off[i], self.tgt_off[i + 1]
        return (torch.from_numpy(self.src[a:b].astype(np.int64)),
                torch.from_numpy(self.tgt[c:d].astype(np.int64)))

# Usage (replaces on-the-fly BilingualDataset when memmap_dir is unset):
#   ds = CachedBilingualDataset("data/train.tsv", "cache/train_v1", src_tok=..., tgt_tok=...,
#                               tok_factory="tokenizer:load_tokenizer", workers=6)
#   sampler = TokenBucketBatchSampler(ds.lengths, max_tokens=4096)
#   loader  = DataLoader(ds, batch_sampler=sampler, collate_fn=your_collate, num_workers=2, persistent_workers=True)
