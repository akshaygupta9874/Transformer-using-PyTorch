"""Data manifests, deterministic cleaning, deduplication, and contamination checks."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetProvenance:
    name: str
    source: str
    license: str
    language_pair: str
    pulled_at: str
    split: str
    notes: str = ""


class ParallelCleaner:
    """Conservative deterministic prefilter; semantic QE belongs in optional adapters."""

    HTML = re.compile(r"<[^>]+>")
    URL = re.compile(r"https?://\S+", re.IGNORECASE)

    def __init__(self, min_chars: int = 1, max_chars: int = 1000, max_length_ratio: float = 3.0):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_length_ratio = max_length_ratio

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFC", text).split())

    def reason_to_reject(self, source: str, target: str) -> str | None:
        source, target = self.normalize(source), self.normalize(target)
        if not source or not target:
            return "empty"
        if min(len(source), len(target)) < self.min_chars:
            return "too_short"
        if max(len(source), len(target)) > self.max_chars:
            return "too_long"
        if max(len(source), len(target)) / min(len(source), len(target)) > self.max_length_ratio:
            return "length_ratio"
        if self.HTML.search(source) or self.HTML.search(target):
            return "html"
        return None


def pair_fingerprint(source: str, target: str) -> str:
    normalized = f"{ParallelCleaner.normalize(source)}\0{ParallelCleaner.normalize(target)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_datasheet(path: str | Path, provenance: DatasetProvenance, statistics: dict) -> None:
    payload = {"provenance": asdict(provenance), "statistics": statistics}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def contamination_fingerprints(records: list[tuple[str, str]], heldout: list[tuple[str, str]]) -> set[str]:
    heldout_hashes = {pair_fingerprint(*pair) for pair in heldout}
    return {pair_fingerprint(*pair) for pair in records} & heldout_hashes
