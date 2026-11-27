#!/usr/bin/env python
"""scripts/build_memmap_stream.py -- bounded-memory, order-preserving memmap builder (Windows-safe).

Host RAM is O(chunk_pairs * max_inflight), independent of corpus size:
    pairs iterator -> chunks of `chunk_pairs` -> spawn Pool (apply_async, at most `max_inflight` chunks
    outstanding, results consumed in submission order) -> appended to .bin files on disk.
Unlike Pool.imap, whose feeder thread drains the whole generator into its task queue, the explicit deque
below gives real back-pressure.  Only per-pair *lengths* (int32) and quality (float32) are held in RAM
(~120 MB at 10.1M pairs) to build the offsets at the end.

Output (names are constants below -- ADAPT them to what MemmapPairs expects):
    src.bin / tgt.bin               uint16 token streams (vocab < 65536)
    src_offsets.npy / tgt_offsets.npy   int64, length N+1
    quality.npy                     float32 (only with --quality-col)
    meta.json                       counts, config, tokenizer sha256

Determinism: output is byte-identical for any --workers value (chunks are processed independently and
written in order; BPE-dropout RNG is seeded by (seed, chunk_idx)).

Offline source BPE-dropout shard (same rows as the base build):
    --mode srcdrop --srcdrop-p 0.1 --srcdrop-tag v1   ->  src.v1.bin + src_offsets.v1.npy

Adapters (module:callable, imported inside every worker):
    --tok-factory   callable(path) -> tokenizer with .encode(text, add_bos, add_eos, dropout_p, rng).ids
    --subst         callable(src, tgt) -> (src', tgt', ...)      default research.placeholders:substitute_parallel
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SRC_BIN, TGT_BIN = "src.bin", "tgt.bin"
SRC_OFF, TGT_OFF = "src_offsets.npy", "tgt_offsets.npy"
QUALITY_NPY, META_JSON = "quality.npy", "meta.json"
EOS_ID = 3

Pair = Tuple[str, str, Optional[float]]


# ----------------------------------------------------------------------------- sources
def iter_tsv(paths: List[str], quality_col: bool = False) -> Iterator[Pair]:
    for p in paths:
        with open(p, "r", encoding="utf-8", newline="") as f:
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 2:
                    continue
                yield parts[0], parts[1], (float(parts[2]) if quality_col and len(parts) > 2 else None)


def iter_jsonl(paths: List[str], src_key="src", tgt_key="tgt", q_key: Optional[str] = None) -> Iterator[Pair]:
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    yield o[src_key], o[tgt_key], (float(o[q_key]) if q_key and q_key in o else None)


def iter_hf(name="ai4bharat/samanantar", config="hi", split="train") -> Iterator[Pair]:
    from datasets import load_dataset          # streaming: rows are fetched lazily, never fully in RAM
    for r in load_dataset(name, config, split=split, streaming=True):
        yield r["src"], r["tgt"], None


# ----------------------------------------------------------------------------- worker side
_W: Dict[str, object] = {}


def _resolve(spec: str) -> Callable:
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def _init_worker(cfg: dict) -> None:
    load = _resolve(cfg["tok_factory"])
    _W["cfg"] = cfg
    _W["src_tok"] = load(cfg["src_tok"])
    _W["tgt_tok"] = load(cfg["tgt_tok"])
    _W["subst"] = _resolve(cfg["subst"]) if cfg["use_placeholders"] else None


def _enc(tok, text, bos, eos, p=0.0, rng=None) -> List[int]:
    if p > 0.0:
        return tok.encode(text, add_bos=bos, add_eos=eos, dropout_p=p, rng=rng).ids
    return tok.encode(text, add_bos=bos, add_eos=eos).ids


def _encode_chunk(job):
    idx, pairs = job
    cfg = _W["cfg"]
    st, tt, subst = _W["src_tok"], _W["tgt_tok"], _W["subst"]
    L = cfg["seq_len"]
    p = cfg["srcdrop_p"] if cfg["mode"] == "srcdrop" else 0.0
    rng = random.Random(cfg["seed"] * 1_000_003 + idx) if p > 0 else None
    s_ids: List[List[int]] = []
    t_ids: List[List[int]] = []
    qs: List[float] = []
    dropped = truncated = 0
    for s, t, q in pairs:
        if subst is not None:
            r = subst(s, t)
            s, t = r[0], r[1]
        si = _enc(st, s, cfg["src_bos"], cfg["src_eos"])           # undropped ids decide keep/drop,
        ti = _enc(tt, t, cfg["tgt_bos"], cfg["tgt_eos"])           # so shards stay row-aligned with base
        if not (1 <= len(si) <= L and 1 <= len(ti) <= L):
            dropped += 1
            continue
        if cfg["mode"] == "srcdrop":
            si = _enc(st, s, cfg["src_bos"], cfg["src_eos"], p, rng)
            if len(si) > L:                                        # dropout lengthens sequences
                si = si[:L - 1] + [EOS_ID] if cfg["src_eos"] else si[:L]
                truncated += 1
        s_ids.append(si); t_ids.append(ti)
        qs.append(0.0 if q is None else q)
    def flat(lists):
        n = sum(map(len, lists))
        a = np.fromiter(itertools.chain.from_iterable(lists), dtype=np.int64, count=n)
        if n and a.max() > 65535:
            raise ValueError("token id > 65535 does not fit uint16")
        return a.astype(np.uint16)
    return {"idx": idx, "n_in": len(pairs), "dropped": dropped, "truncated": truncated,
            "src": flat(s_ids), "tgt": flat(t_ids),
            "src_len": np.fromiter(map(len, s_ids), dtype=np.int32, count=len(s_ids)),
            "tgt_len": np.fromiter(map(len, t_ids), dtype=np.int32, count=len(t_ids)),
            "q": np.asarray(qs, dtype=np.float32)}


# ----------------------------------------------------------------------------- main process
def _chunks(it: Iterable, n: int) -> Iterator[list]:
    it = iter(it)
    while True:
        b = list(itertools.islice(it, n))
        if not b:
            return
        yield b


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def build_memmap(
    pairs: Iterable[Pair],
    out_dir: str,
    *,
    src_tok: str, tgt_tok: str,
    tok_factory: str,
    subst: str = "research.placeholders:substitute_parallel",
    use_placeholders: bool = True,
    seq_len: int = 350,
    src_bos=False, src_eos=True, tgt_bos=True, tgt_eos=True,     # ADAPT to your old __getitem__ convention
    mode: str = "base", srcdrop_p: float = 0.1, srcdrop_tag: str = "v1", seed: int = 13,
    workers: int = max(1, (os.cpu_count() or 2) - 1),
    chunk_pairs: int = 20_000, max_inflight: Optional[int] = None,
    write_quality: bool = False, limit: Optional[int] = None, log_every: int = 50,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(src_tok=src_tok, tgt_tok=tgt_tok, tok_factory=tok_factory, subst=subst,
               use_placeholders=use_placeholders, seq_len=seq_len, src_bos=src_bos, src_eos=src_eos,
               tgt_bos=tgt_bos, tgt_eos=tgt_eos, mode=mode, srcdrop_p=srcdrop_p, seed=seed)
    tag = "" if mode == "base" else f".{srcdrop_tag}"
    if mode == "base":
        names = dict(src=SRC_BIN, src_off=SRC_OFF, tgt=TGT_BIN, tgt_off=TGT_OFF, q=QUALITY_NPY)
    else:
        names = dict(src=f"src{tag}.bin", src_off=f"src_offsets{tag}.npy")
    max_inflight = max_inflight or 2 * workers
    if limit is not None:
        pairs = itertools.islice(pairs, limit)

    partial = {k: os.path.join(out_dir, v + ".partial") for k, v in names.items() if k in ("src", "tgt")}
    files = {k: open(p, "wb") for k, p in partial.items()}
    src_lens: List[np.ndarray] = []
    tgt_lens: List[np.ndarray] = []
    qual: List[np.ndarray] = []
    stats = dict(n_in=0, kept=0, dropped=0, truncated=0, max_inflight=0)
    t0 = time.time()
    pending: collections.deque = collections.deque()

    def consume(r):
        files["src"].write(r["src"].tobytes())
        if "tgt" in files:
            files["tgt"].write(r["tgt"].tobytes())
        src_lens.append(r["src_len"]); tgt_lens.append(r["tgt_len"]); qual.append(r["q"])
        stats["n_in"] += r["n_in"]; stats["kept"] += len(r["src_len"])
        stats["dropped"] += r["dropped"]; stats["truncated"] += r["truncated"]
        if (r["idx"] + 1) % log_every == 0:
            rate = stats["n_in"] / max(time.time() - t0, 1e-9)
            print(f"[build] {stats['n_in']:,} pairs in, {stats['kept']:,} kept, {rate:,.0f} pairs/s", flush=True)

    def drain_one():
        consume(pending.popleft().get())

    try:
        if workers <= 1:                                     # in-process path (debugging / tiny runs)
            _init_worker(cfg)
            for idx, block in enumerate(_chunks(pairs, chunk_pairs)):
                consume(_encode_chunk((idx, block)))
        else:
            ctx = mp.get_context("spawn")                    # Windows semantics on every OS
            with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                for idx, block in enumerate(_chunks(pairs, chunk_pairs)):
                    pending.append(pool.apply_async(_encode_chunk, ((idx, block),)))
                    stats["max_inflight"] = max(stats["max_inflight"], len(pending))
                    while len(pending) >= max_inflight:      # back-pressure
                        drain_one()
                while pending:
                    drain_one()
    finally:
        for f in files.values():
            f.close()

    def offsets(lens: List[np.ndarray]) -> np.ndarray:
        o = np.zeros(1 + sum(len(a) for a in lens), dtype=np.int64)
        if lens:
            np.cumsum(np.concatenate(lens), out=o[1:])
        return o

    np.save(os.path.join(out_dir, names["src_off"]), offsets(src_lens))
    if mode == "base":
        np.save(os.path.join(out_dir, names["tgt_off"]), offsets(tgt_lens))
        if write_quality:
            np.save(os.path.join(out_dir, names["q"]), np.concatenate(qual) if qual else np.zeros(0, np.float32))
    for k, p in partial.items():
        os.replace(p, os.path.join(out_dir, names[k]))       # atomic: no half-written .bin under final name
    meta_path = os.path.join(out_dir, META_JSON if mode == "base" else f"meta{tag}.json")
    meta = dict(stats, config=cfg, src_tok_sha=_sha(src_tok), tgt_tok_sha=_sha(tgt_tok),
                seconds=round(time.time() - t0, 1))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["tsv", "jsonl", "hf"], required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--quality-col", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--src-tok", required=True); ap.add_argument("--tgt-tok", required=True)
    ap.add_argument("--tok-factory", required=True, help="module:callable(path)->tokenizer")
    ap.add_argument("--subst", default="research.placeholders:substitute_parallel")
    ap.add_argument("--no-placeholders", action="store_true")
    ap.add_argument("--seq-len", type=int, default=350)
    ap.add_argument("--mode", choices=["base", "srcdrop"], default="base")
    ap.add_argument("--srcdrop-p", type=float, default=0.1); ap.add_argument("--srcdrop-tag", default="v1")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk-pairs", type=int, default=20_000)
    ap.add_argument("--max-inflight", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.source == "tsv":
        pairs = iter_tsv(a.inputs, a.quality_col)
    elif a.source == "jsonl":
        pairs = iter_jsonl(a.inputs, q_key="quality" if a.quality_col else None)
    else:
        pairs = iter_hf()
    meta = build_memmap(pairs, a.out_dir, src_tok=a.src_tok, tgt_tok=a.tgt_tok, tok_factory=a.tok_factory,
                        subst=a.subst, use_placeholders=not a.no_placeholders, seq_len=a.seq_len,
                        mode=a.mode, srcdrop_p=a.srcdrop_p, srcdrop_tag=a.srcdrop_tag, seed=a.seed,
                        workers=a.workers, chunk_pairs=a.chunk_pairs, max_inflight=a.max_inflight,
                        write_quality=a.quality_col, limit=a.limit)
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    mp.freeze_support()          # required for frozen/Windows entry points
    main()
