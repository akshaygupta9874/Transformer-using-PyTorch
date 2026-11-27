"""research/dynamic_batching.py  --  megabatch-shuffled TokenBucketBatchSampler.

Per epoch:
  1. random permutation of all N indices (seeded by (seed, epoch))
  2. cut into megabatches of ``megabatch_size`` samples
  3. sort each megabatch by length, greedily cut into batches with
        count * max_len_in_batch <= max_tokens   and   count <= max_items
  4. shuffle the batch order globally

A megabatch of ~100k random samples sees essentially the whole length distribution, so padding
efficiency is within a fraction of a percent of a global sort, but batch *membership* differs every
epoch.  The plan is built once per epoch and cached, so ``__len__`` + ``__iter__`` cost one pass.
Plan storage is two flat numpy arrays (no per-batch Python lists): fine for 10M rows.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterator, List, Optional, Sequence

import numpy as np


class TokenBucketBatchSampler:
    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        max_tokens: int = 4096,
        num_buckets: int = 20,            # accepted for backward compatibility; unused
        shuffle: bool = True,
        seed: int = 13,
        max_items: int = 128,
        megabatch_size: int = 100_000,
        drop_last: bool = False,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        if self.lengths.ndim != 1 or len(self.lengths) == 0:
            raise ValueError("lengths must be a non-empty 1-D array")
        self.max_tokens = int(max_tokens)
        self.max_items = int(max_items)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.megabatch_size = int(megabatch_size)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._skip = 0
        self._plan_epoch: Optional[int] = None
        self._order: Optional[np.ndarray] = None      # flat int32 indices, batch after batch
        self._bounds: Optional[np.ndarray] = None     # int64, len = n_batches + 1
        self._expected_len: Optional[int] = None

    # ------------------------------------------------------------------ public API
    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        """Call before iterating each epoch.  ``start_batch`` skips batches (mid-epoch resume)."""
        self.epoch = int(epoch)
        self._skip = int(start_batch)

    def __len__(self) -> int:
        self._ensure_plan()
        return len(self._bounds) - 1

    @property
    def expected_len(self) -> int:
        """Batches in epoch 0 (deterministic).  Use this for LR-schedule length: other epochs differ
        by a handful of batches because megabatch composition changes."""
        if self._expected_len is None:
            if self._plan_epoch == 0:
                self._expected_len = len(self)
            else:
                self._expected_len = len(self._build(0)[1]) - 1
        return self._expected_len

    def __iter__(self) -> Iterator[List[int]]:
        self._ensure_plan()
        order, bounds = self._order, self._bounds
        start, self._skip = self._skip, 0
        for b in range(start, len(bounds) - 1):
            yield order[bounds[b]:bounds[b + 1]].tolist()

    def padding_efficiency(self) -> float:
        """real tokens / padded tokens for the current epoch plan (1.0 = no padding)."""
        self._ensure_plan()
        L = self.lengths[self._order].astype(np.int64)
        real = int(L.sum())
        padded = 0
        for b in range(len(self._bounds) - 1):
            seg = L[self._bounds[b]:self._bounds[b + 1]]
            padded += len(seg) * int(seg.max())
        return real / max(padded, 1)

    # ------------------------------------------------------------------ internals
    def _ensure_plan(self) -> None:
        if self._plan_epoch != self.epoch:
            self._order, self._bounds = self._build(self.epoch)
            self._plan_epoch = self.epoch

    def _cut(self, sorted_len: List[int], base: int, out_bounds: List[int]) -> None:
        """Greedy maximal batches over an ascending-sorted length list (positions base..)."""
        n = len(sorted_len)
        T, M = self.max_tokens, self.max_items
        i = 0
        while i < n:
            hi = min(M, n - i)                                   # candidate count upper bound
            # largest c in [1, hi] with c * sorted_len[i+c-1] <= T  (monotone in c -> bisect)
            lo, best = 1, 1
            hi_c = hi
            while lo <= hi_c:
                mid = (lo + hi_c) // 2
                if mid * sorted_len[i + mid - 1] <= T:
                    best, lo = mid, mid + 1
                else:
                    hi_c = mid - 1
            i += best
            out_bounds.append(base + i)

    def _build(self, epoch: int):
        N = len(self.lengths)
        if self.shuffle:
            rng = np.random.default_rng([self.seed, epoch])
            perm = rng.permutation(N).astype(np.int32)
            mb = self.megabatch_size
        else:
            perm = np.arange(N, dtype=np.int32)
            mb = N                                               # global sort, deterministic
        order = np.empty(N, dtype=np.int32)
        bounds: List[int] = [0]
        for s in range(0, N, mb):
            chunk = perm[s:s + mb]
            ln = self.lengths[chunk]
            srt = np.argsort(ln, kind="stable")
            order[s:s + len(chunk)] = chunk[srt]
            self._cut(ln[srt].tolist(), s, bounds)
        bounds_arr = np.asarray(bounds, dtype=np.int64)
        n_b = len(bounds_arr) - 1
        if self.drop_last and n_b > 1:
            # drop the (single) smallest-by-token batch of each epoch tail only if it is tiny
            last = order[bounds_arr[-2]:bounds_arr[-1]]
            if int(self.lengths[last].sum()) < 0.25 * self.max_tokens:
                bounds_arr = bounds_arr[:-1]
                n_b -= 1
        if self.shuffle:
            rng = np.random.default_rng([self.seed, epoch, 1])
            b_perm = rng.permutation(n_b)
            sizes = np.diff(bounds_arr)[b_perm]
            starts = bounds_arr[:-1][b_perm]
            new_order = np.empty(int(sizes.sum()), dtype=np.int32)
            new_bounds = np.zeros(n_b + 1, dtype=np.int64)
            np.cumsum(sizes, out=new_bounds[1:])
            for k in range(n_b):                                 # n_b ~ 1e5: cheap slice copies
                new_order[new_bounds[k]:new_bounds[k + 1]] = order[starts[k]:starts[k] + sizes[k]]
            return new_order, new_bounds
        return order[:bounds_arr[-1]], bounds_arr
