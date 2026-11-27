"""Memory-mapped pre-tokenized dataset, offline source-dropout variants, and sequence packing.

Eliminates per-epoch CPU Python tokenization overhead while keeping Windows DataLoader
spawn workers safe via lazy file-handle initialization.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import random
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch

from dataset import causal_mask

PAD, UNK, BOS, EOS = 0, 1, 2, 3

# Worker globals for Windows-safe spawn multiprocessing
_WORKER_TOK_SRC = None
_WORKER_TOK_TGT = None
_WORKER_PLACEHOLDERS = True
_WORKER_MAX_SRC = 348
_WORKER_MAX_TGT = 349
_WORKER_VARIANT = 0
_WORKER_DROPOUT_P = 0.0
_WORKER_SEED = 13


def _encode_batch(args):
    fn, batch = args
    return [fn(s, t, v, idx) for (s, t, v, idx) in batch]


def _init_tok_worker(
    tok_src_path: str,
    tok_tgt_path: str,
    use_placeholders: bool,
    seq_len: int,
    variant: int,
    dropout_p: float,
    seed: int,
) -> None:
    global _WORKER_TOK_SRC, _WORKER_TOK_TGT, _WORKER_PLACEHOLDERS
    global _WORKER_MAX_SRC, _WORKER_MAX_TGT, _WORKER_VARIANT, _WORKER_DROPOUT_P, _WORKER_SEED
    from tokenizer import Tokenizer
    _WORKER_TOK_SRC = Tokenizer.load(Path(tok_src_path))
    _WORKER_TOK_TGT = _WORKER_TOK_SRC if tok_tgt_path == tok_src_path else Tokenizer.load(Path(tok_tgt_path))
    _WORKER_PLACEHOLDERS = bool(use_placeholders)
    _WORKER_MAX_SRC = max(1, int(seq_len) - 2)
    _WORKER_MAX_TGT = max(1, int(seq_len) - 1)
    _WORKER_VARIANT = int(variant)
    _WORKER_DROPOUT_P = float(dropout_p)
    _WORKER_SEED = int(seed)


def _encode_tok_item(item: Tuple[int, str, str]) -> Tuple[List[int], List[int]]:
    idx, src_text, tgt_text = item
    if _WORKER_PLACEHOLDERS:
        from research.placeholders import substitute_parallel
        src_text, tgt_text, _ = substitute_parallel(src_text, tgt_text)
    if _WORKER_VARIANT > 0 and _WORKER_DROPOUT_P > 0.0:
        rng = random.Random(_WORKER_SEED * 1_000_003 + _WORKER_VARIANT * 7_919 + idx)
        src_ids = _WORKER_TOK_SRC.encode(src_text, dropout_p=_WORKER_DROPOUT_P, rng=rng).ids
    else:
        src_ids = _WORKER_TOK_SRC.encode(src_text).ids
    tgt_ids = _WORKER_TOK_TGT.encode(tgt_text).ids
    return src_ids[:_WORKER_MAX_SRC], tgt_ids[:_WORKER_MAX_TGT]


def _encode_tok_batch(batch: list[Tuple[int, str, str]]) -> list[Tuple[list[int], list[int]]]:
    return [_encode_tok_item(item) for item in batch]


def build_memmap(
    pairs_iter: Iterable[Tuple[str, str]],
    encode_pair: Callable[[str, str, int, int], Tuple[Sequence[int], Sequence[int]]],
    out_dir: str | Path,
    split: str = "train",
    variant: int = 0,
    n_proc: int = 1,
    chunk: int = 20_000,
    quality_scores: Optional[Sequence[float]] = None,
) -> Path:
    """One-off pre-tokenization to uint16 binary streams + int64 offsets.

    encode_pair(src_str, tgt_str, variant, idx) -> (src_ids, tgt_ids)
    Must be top-level picklable when n_proc > 1.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sfx = "" if variant == 0 else f".v{variant}"
    so, to = [0], [0]

    src_bin = out_path / f"{split}.src{sfx}.bin"
    tgt_bin = out_path / f"{split}.tgt{sfx}.bin"

    def _batches():
        buf = []
        for i, (s, t) in enumerate(pairs_iter):
            buf.append((s, t, variant, i))
            if len(buf) == chunk:
                yield buf
                buf = []
        if buf:
            yield buf

    with src_bin.open("wb") as fs, tgt_bin.open("wb") as ft:
        if n_proc > 1:
            with mp.Pool(n_proc) as pool:
                for res in pool.imap(_encode_batch, ((encode_pair, b) for b in _batches())):
                    for s_ids, t_ids in res:
                        fs.write(np.asarray(s_ids, np.uint16).tobytes())
                        ft.write(np.asarray(t_ids, np.uint16).tobytes())
                        so.append(so[-1] + len(s_ids))
                        to.append(to[-1] + len(t_ids))
        else:
            for b in _batches():
                for s, t, v, idx in b:
                    s_ids, t_ids = encode_pair(s, t, v, idx)
                    fs.write(np.asarray(s_ids, np.uint16).tobytes())
                    ft.write(np.asarray(t_ids, np.uint16).tobytes())
                    so.append(so[-1] + len(s_ids))
                    to.append(to[-1] + len(t_ids))

    np.save(out_path / f"{split}.src{sfx}.off.npy", np.asarray(so, np.int64))
    np.save(out_path / f"{split}.tgt{sfx}.off.npy", np.asarray(to, np.int64))
    if quality_scores is not None and variant == 0:
        np.save(out_path / f"{split}.quality.npy", np.asarray(quality_scores, np.float32))
    return out_path


def build_memmap_from_tokenizers(
    pairs: Sequence[Tuple[str, str]],
    tok_src_path: str | Path,
    tok_tgt_path: str | Path,
    out_dir: str | Path,
    split: str = "train",
    seq_len: int = 350,
    use_placeholders: bool = True,
    n_src_variants: int = 0,
    src_dropout_p: float = 0.05,
    seed: int = 13,
    n_proc: int = 1,
    chunk: int = 20_000,
    quality_scores: Optional[Sequence[float]] = None,
) -> Path:
    """Pre-tokenize pairs with project Akshara tokenizers into memmap files (+ optional dropout variants)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    n_pairs = len(pairs)

    for variant in range(n_src_variants + 1):
        sfx = "" if variant == 0 else f".v{variant}"
        so = [0]
        to = [0] if variant == 0 else None
        src_bin = out_path / f"{split}.src{sfx}.bin"
        tgt_bin = (out_path / f"{split}.tgt.bin") if variant == 0 else None

        def _indexed_batches():
            buf = []
            for idx, (s, t) in enumerate(pairs):
                buf.append((idx, s, t))
                if len(buf) == chunk:
                    yield buf
                    buf = []
            if buf:
                yield buf

        fs = src_bin.open("wb")
        ft = tgt_bin.open("wb") if tgt_bin is not None else None
        try:
            if n_proc > 1:
                with mp.Pool(
                    processes=n_proc,
                    initializer=_init_tok_worker,
                    initargs=(
                        str(tok_src_path),
                        str(tok_tgt_path),
                        use_placeholders,
                        seq_len,
                        variant,
                        src_dropout_p,
                        seed,
                    ),
                ) as pool:
                    for res in pool.imap(_encode_tok_batch, _indexed_batches()):
                        for s_ids, t_ids in res:
                            fs.write(np.asarray(s_ids, np.uint16).tobytes())
                            so.append(so[-1] + len(s_ids))
                            if ft is not None and to is not None:
                                ft.write(np.asarray(t_ids, np.uint16).tobytes())
                                to.append(to[-1] + len(t_ids))
            else:
                _init_tok_worker(
                    str(tok_src_path),
                    str(tok_tgt_path),
                    use_placeholders,
                    seq_len,
                    variant,
                    src_dropout_p,
                    seed,
                )
                for b in _indexed_batches():
                    for s_ids, t_ids in _encode_tok_batch(b):
                        fs.write(np.asarray(s_ids, np.uint16).tobytes())
                        so.append(so[-1] + len(s_ids))
                        if ft is not None and to is not None:
                            ft.write(np.asarray(t_ids, np.uint16).tobytes())
                            to.append(to[-1] + len(t_ids))
        finally:
            fs.close()
            if ft is not None:
                ft.close()

        np.save(out_path / f"{split}.src{sfx}.off.npy", np.asarray(so, np.int64))
        if to is not None:
            np.save(out_path / f"{split}.tgt.off.npy", np.asarray(to, np.int64))

    if quality_scores is not None:
        if len(quality_scores) != n_pairs:
            raise ValueError(f"Expected {n_pairs} quality_scores, got {len(quality_scores)}")
        np.save(out_path / f"{split}.quality.npy", np.asarray(quality_scores, np.float32))
    return out_path


class MemmapPairs(torch.utils.data.Dataset):
    """Zero-tokenization memmap Dataset safe for Windows spawn DataLoader workers."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        n_src_variants: int = 0,
        dropout_frac: float = 0.3,
        seed: int = 13,
        shared_epoch=None,
    ):
        self.root = str(root)
        self.split = split
        self.nv = int(n_src_variants)
        self.frac = float(dropout_frac)
        self.seed = int(seed)
        self.epoch = 0
        self.shared_epoch = shared_epoch
        self._m = None
        self._off = None
        self._q = None
        self.so = np.load(f"{self.root}/{split}.src.off.npy")
        self.to = np.load(f"{self.root}/{split}.tgt.off.npy")
        self.q_path = f"{self.root}/{split}.quality.npy"
        self.has_quality = os.path.exists(self.q_path)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.shared_epoch is not None:
            self.shared_epoch.set(epoch)

    def __len__(self) -> int:
        return len(self.so) - 1

    def pair_lengths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (src_token_lengths, tgt_token_lengths) arrays."""
        return np.diff(self.so), np.diff(self.to)

    def lengths(self) -> list[int]:
        """Return max(src + 2, tgt + 1) per pair for TokenBucketBatchSampler."""
        src_l = np.diff(self.so) + 2
        tgt_l = np.diff(self.to) + 1
        return np.maximum(src_l, tgt_l).tolist()

    def _open(self) -> None:
        if self._m is not None:
            return
        r, s = self.root, self.split
        self._m = {"tgt": np.memmap(f"{r}/{s}.tgt.bin", np.uint16, "r")}
        self._off = {0: self.so}
        for v in range(self.nv + 1):
            sfx = "" if v == 0 else f".v{v}"
            self._m[v] = np.memmap(f"{r}/{s}.src{sfx}.bin", np.uint16, "r")
            if v > 0:
                self._off[v] = np.load(f"{r}/{s}.src{sfx}.off.npy", mmap_mode="r")
        if self.has_quality:
            self._q = np.load(self.q_path, mmap_mode="r")

    def __getstate__(self):
        st = self.__dict__.copy()
        st["_m"] = None
        st["_off"] = None
        st["_q"] = None
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self._m = None
        self._off = None
        self._q = None

    def close(self) -> None:
        """Close underlying memory-mapped file handles cleanly (important on Windows)."""
        for container in (self._m, self._off):
            if isinstance(container, dict):
                for arr in container.values():
                    mm = getattr(arr, "_mmap", None)
                    if mm is not None:
                        try:
                            mm.close()
                        except Exception:
                            pass
        if self._q is not None:
            mm = getattr(self._q, "_mmap", None)
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
        self._m = None
        self._off = None
        self._q = None

    def __del__(self) -> None:
        self.close()

    def _variant(self, idx: int) -> int:
        if self.nv <= 0:
            return 0
        cur_epoch = self.shared_epoch.get() if self.shared_epoch is not None else self.epoch
        h = random.Random(self.seed * 1_000_003 + cur_epoch * 7_919 + idx)
        return 0 if h.random() > self.frac else 1 + h.randrange(self.nv)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        if self._m is None:
            self._open()
        v = self._variant(idx)
        so = self._off[v]
        src = np.asarray(self._m[v][so[idx] : so[idx + 1]], dtype=np.int64)
        tgt = np.asarray(self._m["tgt"][self.to[idx] : self.to[idx + 1]], dtype=np.int64)
        w = 1.0 if self._q is None else float(self._q[idx])
        return src, tgt, w


class MemmapCollate:
    """Dynamic padding collator for MemmapPairs producing standard training batch dicts."""

    def __init__(self, pad_id: int = PAD, bos_id: int = BOS, eos_id: int = EOS):
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

    def __call__(self, batch: Sequence[Tuple[np.ndarray, np.ndarray, float]]) -> dict:
        max_enc = max(len(s) + 2 for s, _, _ in batch)
        max_dec = max(len(t) + 1 for _, t, _ in batch)
        bs = len(batch)

        encoder_input = torch.full((bs, max_enc), self.pad_id, dtype=torch.long)
        decoder_input = torch.full((bs, max_dec), self.pad_id, dtype=torch.long)
        label = torch.full((bs, max_dec), self.pad_id, dtype=torch.long)
        weights = torch.empty((bs,), dtype=torch.float32)

        for i, (s_arr, t_arr, w) in enumerate(batch):
            s_t = torch.from_numpy(s_arr)
            t_t = torch.from_numpy(t_arr)
            ls, lt = s_t.numel(), t_t.numel()
            encoder_input[i, 0] = self.bos_id
            encoder_input[i, 1 : 1 + ls] = s_t
            encoder_input[i, 1 + ls] = self.eos_id

            decoder_input[i, 0] = self.bos_id
            decoder_input[i, 1 : 1 + lt] = t_t
            label[i, :lt] = t_t
            label[i, lt] = self.eos_id
            weights[i] = w

        encoder_mask = (encoder_input != self.pad_id).unsqueeze(1).unsqueeze(1)
        decoder_mask = (decoder_input != self.pad_id).unsqueeze(1).unsqueeze(1) & causal_mask(max_dec)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "sample_weight": weights,
        }


def pack_pairs(
    pairs: Sequence[Tuple[Sequence[int], Sequence[int], float]],
    src_cap: int = 512,
    tgt_cap: int = 512,
) -> dict:
    """First-fit-decreasing bin-packing of parallel token ID pairs with per-document positions."""
    rows = []
    for k in sorted(range(len(pairs)), key=lambda i: -len(pairs[i][1])):
        ls, lt = len(pairs[k][0]) + 1, len(pairs[k][1]) + 1
        if ls > src_cap or lt > tgt_cap:
            continue
        for r in rows:
            if r["s"] + ls <= src_cap and r["t"] + lt <= tgt_cap:
                break
        else:
            r = {"s": 0, "t": 0, "items": []}
            rows.append(r)
        r["items"].append(k)
        r["s"] += ls
        r["t"] += lt

    if not rows:
        empty_l = torch.zeros((0, 0), dtype=torch.long)
        empty_f = torch.zeros((0, 0), dtype=torch.float32)
        return dict(
            src=empty_l, src_doc=empty_l, src_pos=empty_l,
            tgt_in=empty_l, tgt_out=empty_l, tgt_doc=empty_l, tgt_pos=empty_l, tgt_w=empty_f,
        )

    n_rows = len(rows)
    s_max = max(r["s"] for r in rows)
    t_max = max(r["t"] for r in rows)
    s_max = -(-s_max // 8) * 8
    t_max = -(-t_max // 8) * 8

    z = lambda n: np.zeros((n_rows, n), np.int64)
    src, sdoc, spos, tin, tout, tdoc, tpos = z(s_max), z(s_max), z(s_max), z(t_max), z(t_max), z(t_max), z(t_max)
    tw = np.zeros((n_rows, t_max), np.float32)

    for ri, r in enumerate(rows):
        so = to = 0
        for d, k in enumerate(r["items"], start=1):
            s, t, w = pairs[k]
            ls, lt = len(s) + 1, len(t) + 1
            src[ri, so : so + ls - 1] = s
            src[ri, so + ls - 1] = EOS
            sdoc[ri, so : so + ls] = d
            spos[ri, so : so + ls] = np.arange(ls)

            tin[ri, to] = BOS
            tin[ri, to + 1 : to + lt] = t
            tout[ri, to : to + lt - 1] = t
            tout[ri, to + lt - 1] = EOS
            tdoc[ri, to : to + lt] = d
            tpos[ri, to : to + lt] = np.arange(lt)
            tw[ri, to : to + lt] = w
            so += ls
            to += lt

    return {
        "src": torch.from_numpy(src),
        "src_doc": torch.from_numpy(sdoc),
        "src_pos": torch.from_numpy(spos),
        "tgt_in": torch.from_numpy(tin),
        "tgt_out": torch.from_numpy(tout),
        "tgt_doc": torch.from_numpy(tdoc),
        "tgt_pos": torch.from_numpy(tpos),
        "tgt_w": torch.from_numpy(tw),
    }


def packed_masks(src_doc: torch.Tensor, tgt_doc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build NaN-free boolean block-diagonal SDPA masks (True = may attend) for packed batches."""
    t_len = tgt_doc.size(1)
    enc = src_doc[:, None, :, None] == src_doc[:, None, None, :]
    causal = torch.ones((t_len, t_len), dtype=torch.bool, device=tgt_doc.device).tril()
    dec = (tgt_doc[:, None, :, None] == tgt_doc[:, None, None, :]) & causal
    cross = (tgt_doc[:, None, :, None] == src_doc[:, None, None, :]) | (tgt_doc == 0)[:, None, :, None]
    return enc, dec, cross


class PackedMemmapCollate:
    """Packs a batch of MemmapPairs into packed rows with per-document RoPE positions and SDPA masks."""

    def __init__(self, src_cap: int = 512, tgt_cap: int = 512):
        self.src_cap = int(src_cap)
        self.tgt_cap = int(tgt_cap)

    def __call__(self, batch: Sequence[Tuple[np.ndarray, np.ndarray, float]]) -> dict:
        packed = pack_pairs(batch, src_cap=self.src_cap, tgt_cap=self.tgt_cap)
        enc_m, dec_m, cross_m = packed_masks(packed["src_doc"], packed["tgt_doc"])
        return {
            "encoder_input": packed["src"],
            "decoder_input": packed["tgt_in"],
            "encoder_mask": enc_m,
            "decoder_mask": dec_m,
            "cross_mask": cross_m,
            "src_pos": packed["src_pos"],
            "tgt_pos": packed["tgt_pos"],
            "label": packed["tgt_out"],
            "sample_weight": packed["tgt_w"],
        }
