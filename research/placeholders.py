"""Copy-safe placeholders for numbers and URLs during translation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DEVA_DIGITS = "०१२३४५६७८९"
LATIN_DIGITS = "0123456789"
DEVA_TO_LATIN = str.maketrans(DEVA_DIGITS, LATIN_DIGITS)
LATIN_TO_DEVA = str.maketrans(LATIN_DIGITS, DEVA_DIGITS)

URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w])[-+]?[0-9०-९]+(?:[.,][0-9०-९]+)*(?![\w])")
TOKEN = re.compile(r"⟨(?:URL|NUM)_\d+⟩")


def normalize_digits(s: str) -> str:
    """Normalize Devanagari numerals to Latin numerals."""
    return s.translate(DEVA_TO_LATIN)


def to_devanagari_digits(s: str) -> str:
    """Convert Latin numerals to Devanagari numerals."""
    return s.translate(LATIN_TO_DEVA)


def canonical_numeric_value(s: str) -> str:
    """
    Returns a canonical representation of a numeric string for semantic comparison.
    - Converts Devanagari digits to Latin digits
    - Normalizes thousands separators (removes commas if followed by 3 digits)
    - Normalizes leading plus sign
    """
    latin = normalize_digits(s)
    # Remove standard thousands commas: e.g. 10,000 -> 10000
    if "," in latin and "." not in latin:
        parts = latin.split(",")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            latin = "".join(parts)
    elif "," in latin and "." in latin:
        # e.g. 1,000.50
        latin = latin.replace(",", "")
    if latin.startswith("+"):
        latin = latin[1:]
    return latin


@dataclass(frozen=True)
class PlaceholderMap:
    values: dict[str, str] = field(default_factory=dict)
    target_values: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str, script: str = "preserve") -> str:
        """
        Restore placeholders in text.
        Args:
            text: Text containing ⟨NUM_0⟩, ⟨URL_0⟩ placeholders.
            script: "preserve" (uses captured target or source representation),
                    "devanagari" (formats numbers in Devanagari digits),
                    "latin" (formats numbers in Latin digits).
        """
        def repl(match: re.Match[str]) -> str:
            tok = match.group(0)
            val = self.target_values.get(tok) or self.values.get(tok, tok)
            if tok.startswith("⟨NUM_") and script in ("devanagari", "latin"):
                if script == "devanagari":
                    return to_devanagari_digits(normalize_digits(val))
                elif script == "latin":
                    return normalize_digits(val)
            return val

        return TOKEN.sub(repl, text)

    def restore_source(self, text: str) -> str:
        """Restore using the original source representation."""
        return TOKEN.sub(lambda m: self.values.get(m.group(0), m.group(0)), text)

    def restore_target(self, text: str) -> str:
        """Restore using the captured target representation."""
        return TOKEN.sub(
            lambda m: self.target_values.get(m.group(0), self.values.get(m.group(0), m.group(0))), text
        )


@dataclass
class _SpanMatch:
    start: int
    end: int
    text: str
    kind: str
    canonical: str


def _extract_spans(text: str) -> List[_SpanMatch]:
    spans: List[_SpanMatch] = []
    # 1. URLs
    for m in URL.finditer(text):
        spans.append(_SpanMatch(m.start(), m.end(), m.group(0), "URL", m.group(0).lower().rstrip("/")))

    # 2. Numbers (exclude spans overlapping with extracted URLs)
    for m in NUMBER.finditer(text):
        st, ed = m.start(), m.end()
        if any(u.start <= st and ed <= u.end for u in spans):
            continue
        spans.append(_SpanMatch(st, ed, m.group(0), "NUM", canonical_numeric_value(m.group(0))))

    spans.sort(key=lambda s: s.start)
    return spans


def substitute(text: str) -> tuple[str, PlaceholderMap]:
    """Replace copy-sensitive spans in left-to-right source order."""
    spans = _extract_spans(text)
    values: dict[str, str] = {}
    counters = {"URL": 0, "NUM": 0}

    chunks = []
    last_end = 0
    for span in spans:
        chunks.append(text[last_end : span.start])
        token = f"⟨{span.kind}_{counters[span.kind]}⟩"
        counters[span.kind] += 1
        values[token] = span.text
        chunks.append(token)
        last_end = span.end
    chunks.append(text[last_end:])

    return "".join(chunks), PlaceholderMap(values=values)


def substitute_parallel(source: str, target: str) -> tuple[str, str, PlaceholderMap]:
    """Apply source placeholders to a parallel pair via semantic copy verification.

    Handles:
    - Devanagari vs Latin numeral conventions (e.g. १२३ vs 123)
    - Currency and percentage conversions ($100 -> 100 डॉलर, 100% -> 100 प्रतिशत)
    - Date format variations (24/11/2026 -> २४-११-२०२६)
    - Reordered target spans (modifier/noun and clause reordering)
    - Unrelated repeated values (distinguishes sequential occurrences without duplicate token bugs)
    - Unrelated target numbers are preserved untouched.
    """
    src_spans = _extract_spans(source)
    tgt_spans = _extract_spans(target)

    counters = {"URL": 0, "NUM": 0}
    src_tokens: List[Tuple[_SpanMatch, str]] = []
    values: dict[str, str] = {}
    for span in src_spans:
        token = f"⟨{span.kind}_{counters[span.kind]}⟩"
        counters[span.kind] += 1
        src_tokens.append((span, token))
        values[token] = span.text

    # Build substituted source
    src_chunks = []
    last_end = 0
    for span, token in src_tokens:
        src_chunks.append(source[last_end : span.start])
        src_chunks.append(token)
        last_end = span.end
    src_chunks.append(source[last_end:])
    sub_source = "".join(src_chunks)

    # Pool source tokens by kind and canonical value
    src_pool: Dict[Tuple[str, str], List[Tuple[_SpanMatch, str]]] = {}
    for span, token in src_tokens:
        key = (span.kind, span.canonical)
        src_pool.setdefault(key, []).append((span, token))

    # Match target spans to unconsumed source tokens
    matched_target_spans: List[Tuple[_SpanMatch, str]] = []
    target_values: dict[str, str] = {}
    consumed_tokens = set()

    for t_span in tgt_spans:
        key = (t_span.kind, t_span.canonical)
        candidates = src_pool.get(key, [])
        matched_tok = None
        for s_span, tok in candidates:
            if tok not in consumed_tokens:
                matched_tok = tok
                consumed_tokens.add(tok)
                break
        if matched_tok is not None:
            matched_target_spans.append((t_span, matched_tok))
            target_values[matched_tok] = t_span.text

    # Build substituted target from non-overlapping matched spans
    matched_target_spans.sort(key=lambda item: item[0].start)
    tgt_chunks = []
    last_end = 0
    for span, token in matched_target_spans:
        tgt_chunks.append(target[last_end : span.start])
        tgt_chunks.append(token)
        last_end = span.end
    tgt_chunks.append(target[last_end:])
    sub_target = "".join(tgt_chunks)

    mapping = PlaceholderMap(values=values, target_values=target_values)
    return sub_source, sub_target, mapping
