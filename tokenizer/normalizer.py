import re
import unicodedata
from typing import List, Optional, Tuple

# B3: Script base codepoints for major Brahmic scripts
BRAHMIC_BASES = {
    "devanagari": 0x0900,
    "hi": 0x0900,
    "bengali": 0x0980,
    "bn": 0x0980,
    "gurmukhi": 0x0A00,
    "pa": 0x0A00,
    "gujarati": 0x0A80,
    "gu": 0x0A80,
    "oriya": 0x0B00,
    "or": 0x0B00,
    "tamil": 0x0B80,
    "ta": 0x0B80,
    "telugu": 0x0C00,
    "te": 0x0C00,
    "kannada": 0x0C80,
    "kn": 0x0C80,
    "malayalam": 0x0D00,
    "ml": 0x0D00,
}

PLACEHOLDER_RE = re.compile(r"[\u27E8⟨][A-Za-z0-9_]+[\u27E9⟩]")  # ⟨NUM_0⟩, ⟨URL_1⟩, ⟨ENTITY_TEST_99⟩
_WS_RE = re.compile(r"\s+")

WORD_MARK = "\u2581"
ZWJ = "\u200D"
ZWNJ = "\u200C"
VIRAMA = "\u094D"
NUKTA = "\u093C"

# NFC decomposes U+0958-095F but recomposes U+0929 / U+0931 / U+0934.
# Normalize all of them to base + nukta.
_NUKTA_DECOMP = {
    0x0929: "\u0928\u093C", 0x0931: "\u0930\u093C", 0x0934: "\u0933\u093C",
    0x0958: "\u0915\u093C", 0x0959: "\u0916\u093C", 0x095A: "\u0917\u093C",
    0x095B: "\u091C\u093C", 0x095C: "\u0921\u093C", 0x095D: "\u0922\u093C",
    0x095E: "\u092B\u093C", 0x095F: "\u092F\u093C",
}

_MATRA_SET = ({0x093A, 0x093B} | set(range(0x093E, 0x094D))
              | {0x094E, 0x094F, 0x0955, 0x0956, 0x0957, 0x0962, 0x0963})
_MOD_SET = {0x0900, 0x0901, 0x0902, 0x0903} | set(range(0x0951, 0x0955))

_MARK_CHARS = "".join(chr(c) for c in sorted(
    _MATRA_SET | _MOD_SET | {0x093C, 0x094D, 0x200C, 0x200D}))
_MARK_RUN_RE = re.compile("[" + re.escape(_MARK_CHARS) + "]{2,}")


def _mark_rank(ch: str) -> int:
    cp = ord(ch)
    if cp == 0x093C:
        return 0  # NUKTA
    if cp in _MOD_SET:
        return 2  # MODIFIERS
    return 1  # MATRA / VIRAMA / ZW


def unify_brahmic_script(text: str, target_script: str = "hi") -> str:
    """Translate Brahmic characters to target script base via consistent Unicode offsets."""
    target_base = BRAHMIC_BASES.get(target_script.lower(), 0x0900)
    output = []
    for char in text:
        cp = ord(char)
        mapped = False
        for script_name, base in BRAHMIC_BASES.items():
            if len(script_name) > 2:
                if base <= cp <= base + 0x7F:
                    offset = cp - base
                    target_cp = target_base + offset
                    try:
                        output.append(chr(target_cp))
                        mapped = True
                    except ValueError:
                        pass
                    break
        if not mapped:
            output.append(char)
    return "".join(output)


class DevanagariNormalizer:
    """Total, idempotent normalization. Placeholders are masked first."""

    VERSION = "3.1"

    def __init__(
        self,
        devanagari_rules: bool = True,
        unify_danda: bool = True,
        decompose_nukta: bool = True,
        preserve_zwj: bool = True,
        preserve_zwnj: bool = True,
        preserve_zwjn: Optional[bool] = None,  # backwards compatibility alias
        normalize_colon_to_visarga: bool = False,
        normalize_danda: Optional[bool] = None,
        joiner_policy: Optional[str] = None,
        **kwargs,
    ):
        self.devanagari_rules = devanagari_rules
        self.unify_danda = unify_danda if normalize_danda is None else normalize_danda
        self.decompose_nukta = decompose_nukta

        # Joiner policy resolution: "preserve_canonical", "remove_all", "preserve_all"
        if joiner_policy is not None:
            self.joiner_policy = joiner_policy
            if joiner_policy == "remove_all":
                self.preserve_zwj = False
                self.preserve_zwnj = False
            elif joiner_policy == "preserve_all":
                self.preserve_zwj = True
                self.preserve_zwnj = True
            else:
                self.joiner_policy = "preserve_canonical"
                self.preserve_zwj = True
                self.preserve_zwnj = True
        else:
            self.preserve_zwj = preserve_zwj
            self.preserve_zwnj = preserve_zwjn if preserve_zwjn is not None else preserve_zwnj
            if not self.preserve_zwj and not self.preserve_zwnj:
                self.joiner_policy = "remove_all"
            else:
                self.joiner_policy = "preserve_canonical"

        self.preserve_zwjn = self.preserve_zwnj  # alias
        self.normalize_colon_to_visarga = normalize_colon_to_visarga
        self.colon_visarga_count = 0

    def get_policy(self) -> dict:
        """Returns the exact normalization policy dictionary for tokenizer metadata."""
        return {
            "normalizer_version": self.VERSION,
            "joiner_policy": self.joiner_policy,
            "preserve_zwj": self.preserve_zwj,
            "preserve_zwnj": self.preserve_zwnj,
            "nukta_decomposition": (
                "decompose_to_base_plus_nukta" if self.decompose_nukta else "preserve_precomposed"
            ),
            "danda_handling": (
                "unify_pipe_to_danda" if self.unify_danda else "preserve_verbatim"
            ),
            "colon_visarga_handling": (
                "normalize_devanagari_colon_to_visarga"
                if self.normalize_colon_to_visarga
                else "preserve_ascii_colon"
            ),
            "nfc_policy": "canonical_nfc",
            "pretokenization_policy": "placeholder_masking_script_run_akshara",
        }

    @classmethod
    def from_policy(cls, policy: dict) -> "DevanagariNormalizer":
        """Instantiates a DevanagariNormalizer from a policy metadata dictionary."""
        joiner_pol = policy.get("joiner_policy", "preserve_canonical")
        preserve_zwj = policy.get("preserve_zwj", True)
        preserve_zwnj = policy.get("preserve_zwnj", True)
        nukta_decomp = policy.get("nukta_decomposition", "decompose_to_base_plus_nukta") == "decompose_to_base_plus_nukta"
        unify_danda = policy.get("danda_handling", "unify_pipe_to_danda") == "unify_pipe_to_danda"
        colon_vis = policy.get("colon_visarga_handling") == "normalize_devanagari_colon_to_visarga"
        return cls(
            devanagari_rules=True,
            joiner_policy=joiner_pol,
            preserve_zwj=preserve_zwj,
            preserve_zwnj=preserve_zwnj,
            decompose_nukta=nukta_decomp,
            unify_danda=unify_danda,
            normalize_colon_to_visarga=colon_vis,
        )

    @staticmethod
    def split_placeholders(text: str) -> List[Tuple[bool, str]]:
        out: List[Tuple[bool, str]] = []
        pos = 0
        for m in PLACEHOLDER_RE.finditer(text):
            if m.start() > pos:
                out.append((False, text[pos:m.start()]))
            out.append((True, m.group()))
            pos = m.end()
        if pos < len(text):
            out.append((False, text[pos:]))
        return out

    @staticmethod
    def _reorder_marks(s: str) -> str:
        return _MARK_RUN_RE.sub(lambda m: "".join(sorted(m.group(), key=_mark_rank)), s)

    def _apply_joiner_policy(self, s: str) -> str:
        if ZWJ not in s and ZWNJ not in s:
            return s
        if self.joiner_policy == "remove_all" or (not self.preserve_zwj and not self.preserve_zwnj):
            return s.replace(ZWJ, "").replace(ZWNJ, "")
        if self.joiner_policy == "preserve_all":
            return s

        # "preserve_canonical": legitimate joiners directly after a virama are kept; stray ones are dropped
        out: List[str] = []
        for ch in s:
            if ch == ZWJ:
                if not self.preserve_zwj or not (out and out[-1] == VIRAMA):
                    continue
            elif ch == ZWNJ:
                if not self.preserve_zwnj or not (out and out[-1] == VIRAMA):
                    continue
            out.append(ch)
        return "".join(out)

    @staticmethod
    def _drop_stray_joiners(s: str) -> str:
        if ZWJ not in s and ZWNJ not in s:
            return s
        out: List[str] = []
        for ch in s:
            if (ch == ZWJ or ch == ZWNJ) and not (out and out[-1] == VIRAMA):
                continue
            out.append(ch)
        return "".join(out)

    def _segment(self, s: str) -> str:
        s = unicodedata.normalize("NFC", s)

        # Optional colon-as-visarga convention (e.g. QWERTY typing of ':')
        if self.normalize_colon_to_visarga and ":" in s:
            def _replace_visarga(match):
                self.colon_visarga_count += 1
                return match.group(1) + "\u0903"
            s = re.sub(r"([\u0900-\u097F]):", _replace_visarga, s)

        if self.devanagari_rules:
            if self.decompose_nukta:
                s = s.translate(_NUKTA_DECOMP)          # after NFC, covers all nukta cases
            s = self._reorder_marks(s)                  # nukta < matra/virama < modifiers
            s = self._apply_joiner_policy(s)            # joiner policy resolution
            if self.unify_danda:
                s = s.replace("||", "\u0965").replace("|", "\u0964")

        return s.replace(WORD_MARK, " ")                # ▁ is reserved as word marker

    def _once(self, text: str) -> str:
        parts = [p if is_ph else self._segment(p)
                 for is_ph, p in self.split_placeholders(text)]
        return _WS_RE.sub(" ", "".join(parts)).strip(" ")

    def normalize(self, text: str) -> str:
        for _ in range(4):                              # fixpoint => idempotent
            new = self._once(text)
            if new == text:
                break
            text = new
        return text

    @staticmethod
    def script_split(word: str) -> List[str]:
        """Split at script transitions: 'WiFiराउटर' -> ['WiFi', 'राउटर']."""
        runs: List[str] = []
        start = 0
        prev = None

        def _script_of(ch: str) -> str:
            cp = ord(ch)
            if 0x0900 <= cp <= 0x097F or cp == 0x200C or cp == 0x200D:
                return "dev"
            if ch.isdecimal():
                return "digit"
            if ch.isalpha() or unicodedata.category(ch).startswith("M"):
                return "alpha"
            return "punct"

        for i, ch in enumerate(word):
            sc = _script_of(ch)
            if prev is not None and sc != prev:
                runs.append(word[start:i])
                start = i
            prev = sc
        if word:
            runs.append(word[start:])
        return runs
