"""research/profile_sweep.py -- measure, don't guess: peak VRAM and step time per setting, and the largest
micro-batch (in tokens) that fits a VRAM budget.

    from research.profile_sweep import sweep, find_max_tokens, synthetic_batch
    rows = sweep(build_model, settings=[{"kv_heads": None}, {"kv_heads": 4}], max_tokens=4096, ...)

``build_model(**overrides) -> nn.Module`` must return your ModernTransformer.  Every setting is run on a
synthetic batch of exactly ``max_tokens`` padded tokens per side, forward+backward+fused AdamW step through
``research.accum.train_window`` (so the chunked loss is included).  On CPU it still runs (memory = None).
Run it from a ``if __name__ == "__main__":`` block on Windows.
"""
from __future__ import annotations

import gc
import time
from typing import Callable, Dict, List, Optional

import torch

from research.accum import train_window, default_forward_hidden


def synthetic_batch(max_tokens: int, src_len: int = 40, tgt_len: int = 40, src_vocab: int = 16000,
                    tgt_vocab: int = 40000, seed: int = 0) -> Dict[str, torch.Tensor]:
    """B = max_tokens // max(src_len, tgt_len) rows, no padding (worst case for memory)."""
    g = torch.Generator().manual_seed(seed)
    B = max(1, max_tokens // max(src_len, tgt_len))
    tgt = torch.randint(4, tgt_vocab, (B, tgt_len + 1), generator=g)
    return {"src": torch.randint(4, src_vocab, (B, src_len), generator=g),
            "tgt_in": tgt[:, :-1].contiguous(), "tgt_out": tgt[:, 1:].contiguous()}


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_once(model, batch, device, steps: int = 6, warmup: int = 2, forward_hidden=default_forward_hidden,
             lr: float = 1e-4, **tw_kwargs) -> Dict[str, Optional[float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == "cuda"))
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    times: List[float] = []
    for i in range(steps):
        if cuda:
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        train_window(model, [batch], opt, device=device, forward_hidden=forward_hidden, **tw_kwargs)
        if cuda:
            torch.cuda.synchronize(device)
        if i >= warmup:
            times.append(time.perf_counter() - t)
    toks = float((batch["tgt_out"] != 0).sum())
    ms = 1000 * sum(times) / len(times)
    return {"step_ms": ms, "tok_per_s": toks / (ms / 1000),
            "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if cuda else None}


def sweep(build_model: Callable[..., torch.nn.Module], settings: List[dict], max_tokens: int,
          device: Optional[torch.device] = None, batch_fn: Callable[..., dict] = synthetic_batch,
          **run_kwargs) -> List[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for st in settings:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {"setting": st, "max_tokens": max_tokens}
        try:
            model = build_model(**st).to(device)
            row["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
            row.update(run_once(model, batch_fn(max_tokens), device, **run_kwargs))
        except BaseException as e:                              # noqa: BLE001
            if not _is_oom(e):
                raise
            row["oom"] = True
        finally:
            model = None
        rows.append(row)
        print(row, flush=True)
    return rows


def find_max_tokens(fits: Callable[[int], bool], lo: int = 1024, hi: int = 32768, granularity: int = 256) -> int:
    """Largest token budget (multiple of ``granularity``) for which fits(tokens) is True.
    ``fits`` should run one train step and return False on OOM.  Exponential search, then bisection."""
    if not fits(lo):
        return 0
    good = lo
    while good * 2 <= hi and fits(good * 2):
        good *= 2
    bad = min(good * 2, hi + granularity)
    while bad - good > granularity:
        mid = (good + bad) // 2 // granularity * granularity
        if mid <= good:
            break
        if fits(mid):
            good = mid
        else:
            bad = mid
    return good
