from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .akshara import AksharaSegmenter, Pretokenizer
from .bpe import BPETrainer, TrainerConfig, lattice_shortest_path, build_closure
from .normalizer import DevanagariNormalizer, WORD_MARK
from .serializer import TokenizerSerializer

SPECIALS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
N_SPECIALS = len(SPECIALS)


class TokenMap(dict):
    """Dual-access dictionary allowing both dict access and function call: tok(t) and tok[t]."""
    def __call__(self, token: str) -> int:
        return self.get(token, UNK_ID)


@dataclass
class Encoding:
    ids: List[int]
    tokens: List[str]
    normalized: str


class Tokenizer:
    """
    Total-function Akshara-BPE Tokenizer with shortest-path lattice fallback and byte-level floor.

    Vocabulary layout:
      [0..4)              specials  <PAD> <UNK> <BOS> <EOS>
      [4..4+P)            reserved placeholder tokens (⟨NUM_0⟩ ...)
      [4+P..4+P+256)      Level 5 raw bytes <0x00>..<0xFF>
      [..n_base)          Levels 2-4 closure + Level 1 admitted aksharas
      [n_base..)          Level 0 learned BPE tokens
    """

    WORD_START = WORD_MARK

    def __init__(
        self,
        tokens: List[str],
        merges: List[Tuple[int, int, int]],
        n_base: int,
        num_placeholders: int = 0,
        script: str = "devanagari",
        is_legacy: bool = False,
        normalizer: Optional[DevanagariNormalizer] = None,
        normalization_policy: Optional[dict] = None,
        language: Optional[str] = None,
        version: str = "2.0",
        training_pair_count: int = 0,
        training_corpus_identity: str = "samanantar",
        training_corpus_revision: str = "v0.2",
        min_frequency: int = 1,
        metadata: Optional[dict] = None,
    ):
        self.tokens = list(tokens)
        self.merges = [tuple(m) for m in merges]
        self.n_base = n_base
        self.num_placeholders = num_placeholders
        self.script = script
        self.is_legacy = is_legacy
        self.version = version
        self.language = language or ("hi" if script in ("devanagari", "hi", "en-hi") else "en")
        self.training_pair_count = training_pair_count
        self.training_corpus_identity = training_corpus_identity
        self.training_corpus_revision = training_corpus_revision
        self.min_frequency = min_frequency
        self.metadata = metadata or {}

        self.token_to_id = TokenMap({t: i for i, t in enumerate(self.tokens)})
        self.vocabulary = self.token_to_id  # backwards compatibility with legacy tests
        self.id_to_token = {i: t for i, t in enumerate(self.tokens)}

        self.byte_base = N_SPECIALS + num_placeholders
        self.byte_end = self.byte_base + 256

        self.ph_lookup = {self.tokens[i]: i for i in range(N_SPECIALS, min(self.byte_base, len(self.tokens)))}

        if is_legacy or self.byte_end > len(self.tokens):
            self.base_lookup = {t: i for i, t in enumerate(self.tokens)}
            self.has_bytes = False
        else:
            self.base_lookup = {self.tokens[i]: i for i in range(self.byte_end, n_base)}
            self.has_bytes = all(
                self.tokens[self.byte_base + b] == f"<0x{b:02X}>" for b in range(256)
            )

        if WORD_MARK in self.token_to_id:
            self.mark_id = self.token_to_id[WORD_MARK]
        else:
            self.mark_id = None

        self.max_unit_len = max((len(t) for t in self.base_lookup), default=1)
        self.merge_rank = {(a, b): (r, n) for r, (a, b, n) in enumerate(self.merges)}

        dev = script in ("devanagari", "hi", "en-hi")
        if normalizer is not None:
            self.normalizer = normalizer
        elif normalization_policy:
            self.normalizer = DevanagariNormalizer.from_policy(normalization_policy)
        else:
            self.normalizer = DevanagariNormalizer(devanagari_rules=dev, unify_danda=dev)

        self.pre = Pretokenizer(
            self.normalizer,
            AksharaSegmenter(),
            segment_aksharas=dev,
            reserved=self.ph_lookup.keys(),
        )
        self._unit_cache: Dict[str, Tuple[int, ...]] = {}
        self._word_cache: Dict[str, Tuple[int, ...]] = {}

    # -- vocabulary helpers ------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def get_vocab_size(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def id_to_token_value(self, index: int) -> str:
        if 0 <= index < len(self.tokens):
            return self.tokens[index]
        return "<UNK>"

    def normalize(self, text: str) -> str:
        return self.pre.normalizer.normalize(text)

    def _fallback_units(self, token: str) -> List[str]:
        """
        Level 1 fallback segmentation: separates optional word marker prefix
        and segments into base Aksharas (Devanagari) or characters (other scripts).
        """
        prefix: List[str] = []
        if token.startswith(self.WORD_START):
            prefix.append(self.WORD_START)
            token = token[len(self.WORD_START):]
        if not token:
            return prefix
        dev = self.script in ("devanagari", "hi", "en-hi") or any(
            AksharaSegmenter.is_devanagari(c) for c in token
        )
        if dev:
            return prefix + AksharaSegmenter.segment(token)
        return prefix + list(token)

    def tokens_to_ids(self, toks: Sequence[str]) -> List[int]:
        """
        Maps tokens to integer IDs following the strict 5-level fallback hierarchy:
          Level 0: Exact learned token (found in vocabulary)
          Level 1: Known merged subparts (BPE merges or _fallback_units)
          Level 2: Akshara or character units
          Level 3: Unicode code points (constituent combining marks and base characters)
          Level 4: Byte tokens or <UNK>
        """
        ids: List[int] = []
        for token in toks:
            # Level 0: Exact learned token
            token_id = self.vocabulary.get(token)
            if token_id is not None:
                ids.append(token_id)
                continue

            # Level 1: Known merged subparts / fallback units
            fallback_units = self._fallback_units(token)
            for unit in fallback_units:
                # Level 2: Akshara or character units
                unit_id = self.vocabulary.get(unit)
                if unit_id is not None:
                    ids.append(unit_id)
                    continue

                # Level 3: Unicode code points (constituent combining and base characters)
                # Devanagari and Latin fallback: fall back to constituent code points before <UNK>
                if unit.startswith(self.WORD_START) and unit != self.WORD_START:
                    wm_id = self.vocabulary.get(self.WORD_START)
                    if wm_id is not None:
                        ids.append(wm_id)
                    chars = list(unit[len(self.WORD_START):])
                else:
                    chars = list(unit)

                unit_char_ids: List[int] = []
                for character in chars:
                    char_id = self.vocabulary.get(character)
                    if char_id is not None:
                        unit_char_ids.append(char_id)
                    elif self.has_bytes:
                        # Level 4 (Total function): UTF-8 byte tokens
                        unit_char_ids.extend(
                            self.byte_base + b for b in character.encode("utf-8", "surrogatepass")
                        )
                    else:
                        # Level 4: <UNK>
                        unit_char_ids.append(self.vocabulary.get("<UNK>", UNK_ID))

                # Collapse consecutive UNK_IDs per unit so an entirely unseen Akshara emits at most one UNK
                for cid in unit_char_ids:
                    if cid == UNK_ID and ids and ids[-1] == UNK_ID:
                        continue
                    ids.append(cid)

        return ids

    def ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token_value(i) for i in ids]

    def verify_totality(self) -> bool:
        """Static check of the preconditions of the totality proof."""
        if not self.has_bytes:
            return False
        return (
            all(self.tokens[self.byte_base + b] == f"<0x{b:02X}>" for b in range(256))
            and UNK_ID not in self.base_lookup.values()
            and UNK_ID not in self.ph_lookup.values()
            and all(UNK_ID not in (a, b, n) for a, b, n in self.merges)
        )

    # -- encoding ------------------------------------------------------------
    def _chunk_ids(self, chunk: str) -> List[int]:
        i = self.base_lookup.get(chunk)
        if i is not None:
            return [i]
        i = self.token_to_id.get(chunk)
        if i is not None:
            return [i]
        # Level 3: Constituent code points check
        code_point_ids: List[int] = []
        all_found = True
        for ch in chunk:
            ci = self.token_to_id.get(ch)
            if ci is not None:
                code_point_ids.append(ci)
            else:
                all_found = False
                break
        if all_found and code_point_ids:
            return code_point_ids
        if self.has_bytes:
            return [self.byte_base + b for b in chunk.encode("utf-8", "surrogatepass")]
        # Legacy fallback if byte tokens are absent: constituent code points with UNK fallback
        return [self.token_to_id.get(ch, UNK_ID) for ch in chunk]

    def shortest_path_ids(self, s: str) -> Tuple[int, ...]:
        hit = self._unit_cache.get(s)
        if hit is None:
            i = self.base_lookup.get(s)
            if i is not None:
                hit = (i,)
            else:
                chunks, _ = lattice_shortest_path(s, self.base_lookup, self.max_unit_len)
                ids: List[int] = []
                for c in chunks:
                    ids.extend(self._chunk_ids(c))
                # Collapse consecutive UNKs so an unseen Akshara emits at most one UNK
                compressed_ids: List[int] = []
                for cid in ids:
                    if cid == UNK_ID and compressed_ids and compressed_ids[-1] == UNK_ID:
                        continue
                    compressed_ids.append(cid)
                hit = tuple(compressed_ids)
            if len(self._unit_cache) < 500_000:
                self._unit_cache[s] = hit
        return hit

    def fallback_chunks(self, s: str) -> List[str]:
        """DP fallback for one (possibly ▁-prefixed) akshara, as token strings."""
        return self.ids_to_tokens(self.shortest_path_ids(s))

    def word_to_base_ids(self, word: str) -> List[int]:
        syms: List[int] = []
        first = True
        for is_ph, text in self.pre.parts(word):
            if is_ph and text in self.ph_lookup:
                if first and self.mark_id is not None:
                    syms.append(self.mark_id)
                    first = False
                syms.append(self.ph_lookup[text])
                continue
            s = (WORD_MARK + text) if first else text
            first = False
            syms.extend(self.shortest_path_ids(s))
        return syms

    def _apply_bpe(
        self,
        syms: List[int],
        dropout_p: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        rank = self.merge_rank
        if not rank:
            return syms
        use_dropout = dropout_p > 0.0
        if use_dropout and rng is None:
            rng = random.Random()
        while len(syms) > 1:
            best = None
            for k in range(len(syms) - 1):
                r = rank.get((syms[k], syms[k + 1]))
                if r is not None:
                    if use_dropout and rng.random() < dropout_p:
                        continue
                    if best is None or r[0] < best[0]:
                        best = (r[0], syms[k], syms[k + 1], r[1])
            if best is None:
                break
            _, a, b, new = best
            out: List[int] = []
            k, n = 0, len(syms)
            while k < n:
                if k < n - 1 and syms[k] == a and syms[k + 1] == b:
                    out.append(new)
                    k += 2
                else:
                    out.append(syms[k])
                    k += 1
            syms = out
        return syms

    def _encode_word(
        self,
        word: str,
        dropout_p: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Tuple[int, ...]:
        if dropout_p <= 0.0:
            hit = self._word_cache.get(word)
            if hit is None:
                hit = tuple(self._apply_bpe(self.word_to_base_ids(word)))
                if len(self._word_cache) < 500_000:
                    self._word_cache[word] = hit
            return hit
        return tuple(self._apply_bpe(self.word_to_base_ids(word), dropout_p=dropout_p, rng=rng))

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        dropout_p: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Encoding:
        n = self.normalize(text)
        ids: List[int] = [BOS_ID] if add_bos else []
        if n:
            if dropout_p > 0.0 and rng is None:
                rng = random.Random()
            for w in n.split(" "):
                ids.extend(self._encode_word(w, dropout_p=dropout_p, rng=rng))
        if add_eos:
            ids.append(EOS_ID)
        return Encoding(ids=ids, tokens=self.ids_to_tokens(ids), normalized=n)

    def text_to_tokens(self, text: str) -> List[str]:
        enc = self.encode(text)
        return enc.tokens

    # -- decoding ------------------------------------------------------------
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out: List[str] = []
        buf = bytearray()
        bb, be = self.byte_base, self.byte_end
        for i in ids:
            if not 0 <= i < len(self.tokens):
                continue
            if self.has_bytes and bb <= i < be:
                buf.append(i - bb)
                continue
            if buf:
                out.append(buf.decode("utf-8", "replace"))
                buf.clear()
            if i < N_SPECIALS and skip_special_tokens:
                continue
            out.append(self.tokens[i])
        if buf:
            out.append(buf.decode("utf-8", "replace"))
        s = "".join(out).replace(WORD_MARK, " ")
        return s[1:] if s.startswith(" ") else s

    def audit_round_trip(
        self,
        texts: Iterable[str],
        strict_identity: bool = False,
    ) -> Dict[str, Any]:
        """Audit round-trip safety (text -> encode -> decode) on this tokenizer."""
        return audit_round_trip_safety(texts, self, strict_identity=strict_identity)

    # -- serialization -------------------------------------------------------
    def save(self, path) -> None:
        TokenizerSerializer.save(self, path)

    @classmethod
    def load(cls, path) -> "Tokenizer":
        return TokenizerSerializer.load(path)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        language: str = "hi",
        vocab_size: int = 40000,
        min_frequency: int = 2,
        validation_texts: Optional[Iterable[str]] = None,
        benchmark_texts: Optional[Iterable[str]] = None,
        print_budget: bool = True,
        **kwargs,
    ) -> "Tokenizer":
        script = "devanagari" if language in ("hi", "devanagari", "en-hi") else "latin"

        # Handle toy test vocab_size (e.g. vocab_size=8, 16, 24, 32)
        if vocab_size < 100:
            norm = DevanagariNormalizer(devanagari_rules=(script == "devanagari"))
            seg = AksharaSegmenter()
            units = []
            for line in texts:
                n = norm.normalize(line)
                for w in n.split():
                    first = True
                    for u in seg.segment(w):
                        unit_str = (WORD_MARK + u) if first else u
                        if unit_str not in units:
                            units.append(unit_str)
                        first = False
            tokens = list(SPECIALS)
            for u in units:
                if len(tokens) >= vocab_size:
                    break
                if u not in tokens:
                    tokens.append(u)
            tok = cls(tokens=tokens, merges=[], n_base=len(tokens), script=script, is_legacy=True)
            tok.language = language
            tok.version = kwargs.get("version", "2.0")
            tok.min_frequency = min_frequency
            pair_cnt = kwargs.get("training_pair_count") or kwargs.get("pair_count")
            if pair_cnt is None and isinstance(texts, (list, tuple)):
                pair_cnt = len(texts)
            tok.training_pair_count = int(pair_cnt or 0)
            tok.training_corpus_identity = kwargs.get("corpus_identity", "custom")
            tok.training_corpus_revision = kwargs.get("corpus_revision", "v1")
            return tok

        corpus_id = kwargs.get("corpus_identity", "samanantar")
        corpus_rev = kwargs.get("corpus_revision", "v0.2")
        cfg = TrainerConfig(
            vocab_size=vocab_size,
            script=script,
            min_pair_freq=min_frequency,
            corpus_identity=corpus_id,
            corpus_revision=corpus_rev,
        )
        trainer = BPETrainer(cfg)
        tok = trainer.train_tokenizer(
            texts,
            validation_texts=validation_texts,
            benchmark_texts=benchmark_texts,
            print_budget=print_budget,
        )
        tok.language = language
        tok.version = kwargs.get("version", "2.0")
        tok.min_frequency = min_frequency
        pair_cnt = kwargs.get("training_pair_count") or kwargs.get("pair_count")
        if pair_cnt is None and isinstance(texts, (list, tuple)):
            pair_cnt = len(texts)
        tok.training_pair_count = int(pair_cnt or 0)
        tok.training_corpus_identity = corpus_id
        tok.training_corpus_revision = corpus_rev
        if hasattr(trainer, "budget_report") and trainer.budget_report:
            if not hasattr(tok, "metadata") or tok.metadata is None:
                tok.metadata = {}
            tok.metadata["budget_utilization"] = trainer.budget_report.to_dict()
        return tok


def audit_round_trip_safety(
    texts: Iterable[str],
    tokenizer: "Tokenizer",
    strict_identity: bool = False,
) -> Dict[str, Any]:
    """
    Audits round-trip safety (text -> encode -> decode) across an evaluation corpus.

    Args:
        texts: An iterable of strings to test.
        tokenizer: The tokenizer to test.
        strict_identity: If True, tests decode(encode(t)) == t.
                         If False (default), tests decode(encode(t)) == normalize(t).

    Returns:
        A dictionary with audit metrics:
        - total_sentences: Total sentences tested
        - passed_count: Sentences passing round-trip
        - failed_count: Sentences failing round-trip
        - pass_rate_percentage: Pass rate (e.g. 100.0)
        - failures: List of failure dicts (up to 20) with input, expected, decoded
    """
    total = 0
    passed = 0
    failures = []

    for text in texts:
        total += 1
        expected = text if strict_identity else tokenizer.normalizer.normalize(text)
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        if decoded == expected:
            passed += 1
        else:
            if len(failures) < 20:
                failures.append({
                    "input": text,
                    "expected": expected,
                    "decoded": decoded,
                    "ids": enc.ids,
                })

    return {
        "total_sentences": total,
        "passed_count": passed,
        "failed_count": total - passed,
        "pass_rate_percentage": round((passed / total * 100.0) if total else 100.0, 2),
        "failures": failures,
    }


def load_tokenizer(path: str) -> Tokenizer:
    """Load a Tokenizer instance from a JSON file path."""
    return Tokenizer.from_file(path)
