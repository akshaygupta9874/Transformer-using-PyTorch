"""Auditable filtering for back-translated and distilled parallel data."""

from __future__ import annotations

from dataclasses import dataclass

from .data import ParallelCleaner, pair_fingerprint


@dataclass(frozen=True)
class SyntheticPair:
    source: str
    target: str
    origin: str
    teacher_score: float | None = None


def filter_synthetic(pairs, cleaner: ParallelCleaner, minimum_score: float | None = None):
    seen, kept, rejected = set(), [], []
    for pair in pairs:
        reason = cleaner.reason_to_reject(pair.source, pair.target)
        fingerprint = pair_fingerprint(pair.source, pair.target)
        if reason is None and minimum_score is not None and (pair.teacher_score is None or pair.teacher_score < minimum_score):
            reason = "teacher_score"
        if reason is None and fingerprint in seen:
            reason = "duplicate"
        if reason is None:
            seen.add(fingerprint)
            kept.append(pair)
        else:
            rejected.append((pair, reason))
    return kept, rejected
