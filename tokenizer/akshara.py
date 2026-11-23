import unicodedata
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

C_CONS, C_INDV, C_MATRA, C_MOD, C_NUKTA, C_VIRAMA, C_ZW, C_OTHER, C_START = range(9)
CLASS_NAMES = ("C", "INDV", "MATRA", "MOD", "NUKTA", "VIRAMA", "ZW", "OTHER", "START")

_CONS = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960)) | set(range(0x0978, 0x0980))
_INDV = set(range(0x0904, 0x0915)) | {0x0950, 0x0960, 0x0961} | set(range(0x0972, 0x0978))
_MATRA = ({0x093A, 0x093B} | set(range(0x093E, 0x094D))
          | {0x094E, 0x094F, 0x0955, 0x0956, 0x0957, 0x0962, 0x0963})
_MOD = {0x0900, 0x0901, 0x0902, 0x0903} | set(range(0x0951, 0x0955))

_CLASS_OF: Dict[int, int] = {}
for _s, _c in ((_CONS, C_CONS), (_INDV, C_INDV), (_MATRA, C_MATRA), (_MOD, C_MOD)):
    for _cp in _s:
        _CLASS_OF[_cp] = _c
_CLASS_OF[0x093C] = C_NUKTA
_CLASS_OF[0x094D] = C_VIRAMA
_CLASS_OF[0x200C] = C_ZW
_CLASS_OF[0x200D] = C_ZW

ZWJ = "\u200D"
ZWNJ = "\u200C"
VIRAMA = "\u094D"
PLACEHOLDER_RE = re.compile(r"[\u27E8⟨][A-Za-z0-9_]+[\u27E9⟩]")


def char_class(ch: str) -> int:
    return _CLASS_OF.get(ord(ch), C_OTHER)


def is_dev(ch: str) -> bool:
    cp = ord(ch)
    return 0x0900 <= cp <= 0x097F or cp == 0x200C or cp == 0x200D


def script_of(ch: str) -> str:
    if is_dev(ch):
        return "dev"
    if ch.isdecimal():
        return "digit"
    if ch.isalpha() or unicodedata.category(ch).startswith("M"):
        return "alpha"
    return "punct"


class AksharaSegmenter:
    """
    Grammar:
      akshara  := cons_unit (VIRAMA [ZW] cons_unit)* MATRA* MOD*     (conjunct + vowel sign)
                | cons_unit (VIRAMA [ZW] cons_unit)* VIRAMA [ZW]     (word-final / halant-terminated)
                | INDV MOD*
                | PLACEHOLDER (⟨NUM_0⟩)
                | any other single code point (orphans)
      cons_unit := C [NUKTA]
    """

    VERSION = "2.0"
    VIRAMA = "\u094D"
    DEVANAGARI_START = "\u0900"
    DEVANAGARI_END = "\u097F"

    @staticmethod
    def is_devanagari(character: str) -> bool:
        if not character:
            return False
        cp = ord(character)
        return (0x0900 <= cp <= 0x097F) or cp in (0x200C, 0x200D)

    @classmethod
    def segment(cls, text: str) -> List[str]:
        n = len(text)
        out: List[str] = []
        i = 0
        while i < n:
            # Check for atomic placeholder first (e.g. ⟨NUM_0⟩, ⟨URL_1⟩, ⟨ENTITY_TEST_99⟩)
            if text[i] in ("\u27E8", "⟨"):
                m = PLACEHOLDER_RE.match(text, i)
                if m:
                    out.append(m.group())
                    i = m.end()
                    continue

            # Whitespace
            if text[i].isspace():
                out.append(text[i])
                i += 1
                continue

            # Devanagari Akshara scanning
            j = cls._scan(text, i, n)
            out.append(text[i:j])
            i = j
        return out

    @staticmethod
    def _scan(text: str, i: int, n: int) -> int:
        c = char_class(text[i])
        if c == C_CONS:
            j = i
            while True:
                j += 1  # consume consonant
                if j < n and char_class(text[j]) == C_NUKTA:
                    j += 1
                if j < n and char_class(text[j]) == C_VIRAMA:
                    j += 1
                    if j < n and char_class(text[j]) == C_ZW:
                        j += 1
                    if j < n and char_class(text[j]) == C_CONS:
                        continue  # conjunct continues
                    return j  # word-final / halant-terminated (e.g. संसद्)
                break
            while j < n and char_class(text[j]) == C_MATRA:
                j += 1
            while j < n and char_class(text[j]) == C_MOD:
                j += 1
            return j

        if c == C_INDV:
            j = i + 1
            while j < n and char_class(text[j]) == C_MOD:
                j += 1
            return j

        return i + 1


class Pretokenizer:
    """word -> [(is_reserved_placeholder, unit)]; units are aksharas or single chars."""

    def __init__(self, normalizer, segmenter: AksharaSegmenter,
                 segment_aksharas: bool = True, reserved: Iterable[str] = ()):
        self.normalizer = normalizer
        self.segmenter = segmenter
        self.segment_aksharas = segment_aksharas
        self.reserved = frozenset(reserved)

    def parts(self, word: str) -> List[Tuple[bool, str]]:
        out: List[Tuple[bool, str]] = []
        for is_ph, piece in self.normalizer.split_placeholders(word):
            if is_ph and piece in self.reserved:
                out.append((True, piece))
                continue
            for run in self.normalizer.script_split(piece):
                if self.segment_aksharas and is_dev(run[0]):
                    out.extend((False, u) for u in self.segmenter.segment(run))
                else:
                    out.extend((False, ch) for ch in run)
        return out
