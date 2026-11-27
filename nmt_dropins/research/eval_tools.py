"""research/eval_tools.py -- pure-Python evaluation helpers (no sacrebleu / torchmetrics needed).

* copy_preservation      : are the numbers / URLs of the source reproduced in the hypothesis, whatever the
                           numeral script (Latin 123 vs Devanagari १२३) and grouping (1,00,000 vs 100000)?
* hash_split             : content-hash train/val split -> membership does NOT change when you grow
                           max_train_samples (1M -> 2M -> 10.2M) or reshuffle the file.
* evaluate_benchmark     : run any ``translate_fn(list[str]) -> list[str]`` on a src<TAB>ref TSV, apply your
                           metric callables, and refuse to call a 3-line smoke placeholder a benchmark.
"""
from __future__ import annotations

import collections
import hashlib
import re
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_DEV = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"'⟨⟩]+", re.IGNORECASE)
_TRAIL = ".,;:!?)]}»”’।"


def canon_digits(text: str) -> str:
    """Devanagari digits -> ASCII digits."""
    return text.translate(_DEV)


def _numbers(text: str) -> List[str]:
    text = _URL.sub(" ", canon_digits(text))                    # digits inside URLs are not 'numbers'
    return [m.group(0).replace(",", "").rstrip(".") for m in _NUM.finditer(text)]


def _urls(text: str) -> List[str]:
    return [u.rstrip(_TRAIL).lower() for u in _URL.findall(text)]


def copy_items(src: str) -> Tuple[List[str], List[str]]:
    return _numbers(src), _urls(src)


def copy_preservation(src: str, hyp: str) -> Tuple[int, int]:
    """(n_items_in_source, n_preserved).  Multiset semantics: a number appearing twice in the source must
    appear twice in the hypothesis."""
    n_s, u_s = copy_items(src)
    n_h, u_h = collections.Counter(_numbers(hyp)), collections.Counter(_urls(hyp))
    kept = 0
    for items, pool in ((n_s, n_h), (u_s, u_h)):
        for it in items:
            if pool[it] > 0:
                pool[it] -= 1
                kept += 1
    return len(n_s) + len(u_s), kept


def corpus_copy_preservation(srcs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    tot = kept = sent_all = sent_with = 0
    for s, h in zip(srcs, hyps):
        n, k = copy_preservation(s, h)
        tot += n; kept += k
        if n:
            sent_with += 1; sent_all += int(n == k)
    return {"copy_rate": kept / tot if tot else float("nan"), "items": tot,
            "sent_exact": sent_all / sent_with if sent_with else float("nan"), "sents_with_items": sent_with}


def hash_split(src: str, val_permille: int = 2, salt: str = "v1") -> bool:
    """True -> validation.  Keyed on the SOURCE sentence only, so every translation of the same source
    lands on the same side (no leakage through duplicate sources)."""
    h = hashlib.blake2b(f"{salt}\x1f{src}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % 1000 < val_permille


def read_tsv(path: str, max_n: Optional[int] = None) -> Tuple[List[str], List[str]]:
    srcs, refs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if len(p) >= 2 and p[0].strip():
                srcs.append(p[0]); refs.append(p[1])
                if max_n and len(srcs) >= max_n:
                    break
    return srcs, refs


def evaluate_benchmark(
    translate_fn: Callable[[List[str]], List[str]],
    tsv_path: str,
    metrics: Optional[Dict[str, Callable[[List[str], List[str]], float]]] = None,
    max_n: Optional[int] = None,
    min_lines: int = 100,
    batch_size: int = 64,
) -> Dict[str, object]:
    """metrics: name -> callable(hyps, refs) -> float  (plug in your pure-Python chrF++/BLEU here).
    Adds copy-preservation automatically.  Flags tiny files as smoke placeholders."""
    srcs, refs = read_tsv(tsv_path, max_n)
    out: Dict[str, object] = {"n": len(srcs), "path": tsv_path, "is_smoke": len(srcs) < min_lines}
    if out["is_smoke"]:
        warnings.warn(f"{tsv_path} has only {len(srcs)} lines: smoke placeholder, scores are meaningless "
                      f"(run scripts/fetch_benchmarks.py).", stacklevel=2)
    hyps: List[str] = []
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))            # length-sorted -> less padding
    res: Dict[int, str] = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        for i, h in zip(idx, translate_fn([srcs[i] for i in idx])):
            res[i] = h
    hyps = [res[i] for i in range(len(srcs))]
    for name, fn in (metrics or {}).items():
        out[name] = fn(hyps, refs)
    out.update(corpus_copy_preservation(srcs, hyps))
    out["_hyps"] = hyps
    return out


def write_samples_tsv(path: str, srcs: Iterable[str], hyps: Iterable[str], refs: Iterable[str], k: int = 200) -> None:
    """Dump the first k (src, hyp, ref, copy_ok) rows -- paste these when asking for error-taxonomy help."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("src\thyp\tref\tcopy_ok\n")
        for i, (s, h, r) in enumerate(zip(srcs, hyps, refs)):
            if i >= k:
                break
            n, kp = copy_preservation(s, h)
            f.write("\t".join(x.replace("\t", " ").replace("\n", " ") for x in (s, h, r)) + f"\t{int(n == kp)}\n")
