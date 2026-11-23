from __future__ import annotations

import heapq
from pathlib import Path
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .akshara import (
    C_CONS, C_INDV, C_MATRA, C_MOD, _CONS, _INDV, AksharaSegmenter, Pretokenizer,
)
from .normalizer import DevanagariNormalizer, NUKTA, VIRAMA, WORD_MARK, ZWJ, ZWNJ


@dataclass
class BudgetUtilizationReport:
    """Detailed vocabulary budget utilization and token fertility diagnostics."""

    unique_atomic_units: int
    base_vocab_size: int
    configured_vocab_size: int
    occupied_percentage: float
    remaining_merge_slots: int
    rare_unit_count: int
    corpus_unk_rate: float
    corpus_fallback_rate: float
    corpus_fertility: float
    validation_fertility: Optional[float] = None
    benchmark_fertility: Optional[float] = None
    language: str = "hi"
    rare_frequency_threshold: int = 2
    post_bpe_corpus_fertility: Optional[float] = None
    post_bpe_validation_fertility: Optional[float] = None
    post_bpe_benchmark_fertility: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "unique_atomic_units": self.unique_atomic_units,
            "base_vocab_size": self.base_vocab_size,
            "configured_vocab_size": self.configured_vocab_size,
            "occupied_percentage": round(self.occupied_percentage, 2),
            "remaining_merge_slots": self.remaining_merge_slots,
            "rare_unit_count": self.rare_unit_count,
            "rare_frequency_threshold": self.rare_frequency_threshold,
            "corpus_unk_rate": round(self.corpus_unk_rate, 4),
            "corpus_fallback_rate": round(self.corpus_fallback_rate, 2),
            "corpus_fertility": round(self.corpus_fertility, 3),
            "validation_fertility": round(self.validation_fertility, 3) if self.validation_fertility is not None else None,
            "benchmark_fertility": round(self.benchmark_fertility, 3) if self.benchmark_fertility is not None else None,
            "language": self.language,
        }
        if self.post_bpe_corpus_fertility is not None:
            d["post_bpe_corpus_fertility"] = round(self.post_bpe_corpus_fertility, 3)
        if self.post_bpe_validation_fertility is not None:
            d["post_bpe_validation_fertility"] = round(self.post_bpe_validation_fertility, 3)
        if self.post_bpe_benchmark_fertility is not None:
            d["post_bpe_benchmark_fertility"] = round(self.post_bpe_benchmark_fertility, 3)
        return d

    def print_report(self) -> None:
        val_str = f"{self.validation_fertility:.3f} tokens/word" if self.validation_fertility is not None else "N/A"
        bench_str = f"{self.benchmark_fertility:.3f} tokens/word" if self.benchmark_fertility is not None else "N/A"
        sep = "=" * 80
        print(f"\n{sep}")
        print("                    PRE-TRAINING VOCABULARY BUDGET ANALYSIS")
        print(sep)
        print(f"  Language:                              {self.language}")
        print(f"  Configured Vocabulary Size:            {self.configured_vocab_size:,}")
        print(f"  Number of Unique Atomic Units:         {self.unique_atomic_units:,}")
        print(f"  Base Vocabulary Size:                  {self.base_vocab_size:,}")
        print(f"  Budget Occupied by Base:               {self.occupied_percentage:.2f}%")
        print(f"  Remaining Merge Slots:                 {self.remaining_merge_slots:,}")
        print(f"  Rare-Unit Count (freq < {self.rare_frequency_threshold}):             {self.rare_unit_count:,}")
        print(f"  Corpus UNK Rate:                       {self.corpus_unk_rate:.2f}% (Total Function)")
        print(f"  Corpus Fallback Rate:                  {self.corpus_fallback_rate:.2f}%")
        print(f"  Token Fertility (Corpus Base):         {self.corpus_fertility:.3f} tokens/word")
        print(f"  Token Fertility (Validation Text):     {val_str}")
        print(f"  Token Fertility (Benchmark Text):      {bench_str}")
        print(f"{sep}\n", flush=True)


def load_benchmark_sentences(
    language: str = "hi",
    max_sentences: int = 100,
    benchmark_dir: Optional[str] = None,
) -> List[str]:
    """
    Load benchmark evaluation sentences from data/benchmarks/flores200_devtest.tsv
    or similar benchmark TSV files.
    """
    candidates = []
    if benchmark_dir:
        candidates.append(Path(benchmark_dir) / "flores200_devtest.tsv")
    candidates.extend([
        Path("data/benchmarks/flores200_devtest.tsv"),
        Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "flores200_devtest.tsv",
    ])
    col_idx = 1 if language in ("hi", "devanagari", "en-hi") else 0
    sentences: List[str] = []
    for cand in candidates:
        if cand.exists():
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) > col_idx:
                            s = parts[col_idx].strip()
                            if s:
                                sentences.append(s)
                                if len(sentences) >= max_sentences:
                                    break
                if sentences:
                    return sentences
            except Exception:
                pass
    return sentences


def compute_fertility(texts: Iterable[str], tokenizer) -> float:
    """
    Computes token fertility: tokens_count / words_count.
    Higher values mean more token fragments per word.
    """
    total_tokens = 0
    total_words = 0
    for text in texts:
        words = text.strip().split()
        if not words:
            continue
        total_words += len(words)
        total_tokens += len(tokenizer.encode(text).ids)
    return (total_tokens / total_words) if total_words > 0 else 0.0

# Core combining marks for Closure Set
_CORE_MATRAS = [chr(c) for c in (0x093E, 0x093F, 0x0940, 0x0941, 0x0942, 0x0943,
                                  0x0947, 0x0948, 0x094B, 0x094C, 0x0945, 0x0949)]
_NUKTA_BASES = [chr(c) for c in (0x0915, 0x0916, 0x0917, 0x091C, 0x0921, 0x0922,
                                 0x092B, 0x092F, 0x0928, 0x0930, 0x0933)]


def lattice_shortest_path(s: str, vocab, max_len: int) -> Tuple[List[str], int]:
    """
    Minimum-cost segmentation of s into chunks that are in `vocab`. A character
    that is not in vocab costs len(utf8) (its byte tokens) and appears as a
    one-char chunk absent from vocab. Always feasible. Ties prefer the longest
    last chunk, so output is deterministic.
    """
    n = len(s)
    INF = 1 << 60
    cost = [INF] * (n + 1)
    back = [0] * (n + 1)
    cost[0] = 0
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):  # ascending j => longest first
            if s[j:i] in vocab:
                c = cost[j] + 1
                if c < cost[i]:
                    cost[i], back[i] = c, j
        ch = s[i - 1]
        if ch not in vocab:
            c = cost[i - 1] + len(ch.encode("utf-8", "surrogatepass"))
            if c < cost[i]:
                cost[i], back[i] = c, i - 1
    chunks: List[str] = []
    i = n
    while i > 0:
        j = back[i]
        chunks.append(s[j:i])
        i = j
    chunks.reverse()
    return chunks, cost[n]


@dataclass
class ClosureSet:
    levels: Dict[int, List[str]]  # level -> token strings (bytes excluded)
    byte_tokens: List[str]

    def non_byte_tokens(self) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for lvl in (2, 3, 4):
            for t in self.levels.get(lvl, []):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    def size(self) -> int:
        return len(self.non_byte_tokens()) + len(self.byte_tokens)


def build_closure(script: str = "devanagari", reserve_c_matra: bool = True) -> ClosureSet:
    lv2: List[str] = []
    lv3: List[str] = []
    if script in ("devanagari", "hi", "en-hi"):
        for cp in range(0x0900, 0x0980):
            if unicodedata.category(chr(cp)) != "Cn":
                lv3.append(chr(cp))
        lv3 += [ZWNJ, ZWJ]
        cons = [chr(c) for c in sorted(_CONS) if unicodedata.category(chr(c)) != "Cn"
                and not 0x0958 <= c <= 0x095F]
        std_cons = [chr(c) for c in range(0x0915, 0x093A)]
        indv = [chr(c) for c in sorted(_INDV) if unicodedata.category(chr(c)) != "Cn"]
        for c in cons:
            lv2.append(c + VIRAMA)
            lv2.append(c + VIRAMA + ZWJ)
        for c in _NUKTA_BASES:
            lv2.append(c + NUKTA)
        if reserve_c_matra:
            for c in std_cons:
                for m in _CORE_MATRAS:
                    lv2.append(c + m)
        # word-initial (▁-prefixed) forms of the reserved pieces
        for c in cons + indv:
            lv2.append(WORD_MARK + c)
        for c in cons:
            lv2.append(WORD_MARK + c + VIRAMA)
    lv4 = [chr(c) for c in range(0x21, 0x7F)] + [WORD_MARK]
    return ClosureSet(levels={2: lv2, 3: lv3, 4: lv4},
                      byte_tokens=[f"<0x{b:02X}>" for b in range(256)])


@dataclass
class TrainerConfig:
    vocab_size: int = 40000
    script: str = "devanagari"  # "devanagari" (Hindi) or "latin" (English)
    coverage: float = 0.9995  # cumulative akshara coverage cutoff
    max_admitted_frac: float = 0.4  # cap on admitted aksharas, as a fraction of vocab_size
    dp_prune_cost: int = 2  # aksharas rank > min_rank decomposable in <= this many chunks are left to BPE
    dp_prune_min_rank: int = 1000
    min_pair_freq: int = 2
    reserve_c_matra: bool = True
    placeholder_types: Tuple[str, ...] = ("NUM", "URL", "DATE", "EMAIL", "TIME", "PHONE")
    placeholders_per_type: int = 16
    corpus_identity: str = "samanantar"
    corpus_revision: str = "v0.2"


class BPETrainer:
    """
    Deterministic lazy-heap BPE trainer supporting both:
    1. Reference sequence-level merges for unit tests.
    2. Full 3-stage Akshara-BPE training with Closure Set G and lattice DP.
    """

    WORD_START = WORD_MARK

    def __init__(
        self,
        config: Optional[TrainerConfig] = None,
        vocab_size: int = 40000,
        min_frequency: int = 2,
        progress_every: int = 100,
        **kwargs,
    ):
        if config is not None:
            self.cfg = config
        else:
            self.cfg = TrainerConfig(vocab_size=vocab_size, min_pair_freq=min_frequency)

        self.vocab_size = self.cfg.vocab_size
        self.min_frequency = self.cfg.min_pair_freq
        self.progress_every = progress_every

        self.closure = build_closure(self.cfg.script, self.cfg.reserve_c_matra)
        self.placeholders = [f"\u27E8{t}_{i}\u27E9" for t in self.cfg.placeholder_types
                             for i in range(self.cfg.placeholders_per_type)]
        dev = self.cfg.script in ("devanagari", "hi", "en-hi")
        self.pre = Pretokenizer(DevanagariNormalizer(devanagari_rules=dev, unify_danda=dev),
                                AksharaSegmenter(), segment_aksharas=dev,
                                reserved=self.placeholders)

        self.base_frequencies: Counter = Counter()
        self.base_vocabulary: Set[str] = set()
        self.atomic_vocabulary: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self._learned_merges_full: List[Tuple[int, int, int]] = []
        self.budget_report: Optional[BudgetUtilizationReport] = None

    @staticmethod
    def _boundary(unit: str) -> bool:
        return (
            not unit
            or unit.isspace()
            or (
                len(unit) == 1
                and not unit.isalnum()
                and not any("\u0900" <= char <= "\u097f" for char in unit)
            )
        )

    @classmethod
    def _valid_pair(cls, left: str, right: str) -> bool:
        if cls._boundary(left) or cls._boundary(right):
            return False
        if right.startswith(cls.WORD_START):
            return False
        return True

    @staticmethod
    def _descending_text_key(value: str) -> tuple:
        return tuple(-ord(c) for c in value)

    @classmethod
    def _heap_entry(cls, pair: Tuple[str, str], count: int) -> tuple:
        return (
            -count,
            cls._descending_text_key(pair[0]),
            cls._descending_text_key(pair[1]),
            pair[0],
            pair[1],
        )

    def apply(self, units: Sequence[str]) -> List[str]:
        """Apply learned string merges deterministically to a sequence of units."""
        if not self.merges:
            return list(units)
        seq = list(units)
        for left, right in self.merges:
            pair = left + right
            out = []
            k = 0
            n = len(seq)
            while k < n:
                if k < n - 1 and seq[k] == left and seq[k + 1] == right:
                    out.append(pair)
                    k += 2
                else:
                    out.append(seq[k])
                    k += 1
            seq = out
        return seq

    def vocab_statistics(self, target_vocab_size: int = 40) -> dict:
        base_count = len(self.base_vocabulary)
        if not base_count:
            if self.budget_report:
                base_count = self.budget_report.base_vocab_size
            else:
                base_count = len(self.closure.non_byte_tokens())
        stats = {
            "unique_base_units": base_count,
            "merges_count": len(self.merges),
        }
        if self.budget_report:
            stats.update(self.budget_report.to_dict())
        return stats

    # -- Sequence-level BPE for unit tests (matches bpe_ref) -------------------
    def _train_on_sequences(self, sequences: Sequence[Sequence[str]]) -> "BPETrainer":
        corpus = [list(seq) for seq in sequences]
        self.base_frequencies = Counter(unit for seq in corpus for unit in seq)
        self.base_vocabulary = set(self.base_frequencies)
        self.atomic_vocabulary = {c for seq in corpus for u in seq for c in u}
        vocabulary = set(self.base_vocabulary)

        self.merges = []
        while len(vocabulary) < self.vocab_size:
            counts = Counter()
            for seq in corpus:
                for left, right in zip(seq, seq[1:]):
                    if self._valid_pair(left, right):
                        counts[(left, right)] += 1
            candidates = [
                (count, pair)
                for pair, count in counts.items()
                if count >= self.min_frequency and (pair[0] + pair[1]) not in vocabulary
            ]
            if not candidates:
                break
            pair = max(candidates, key=lambda item: (item[0], item[1][0], item[1][1]))[1]
            merged = pair[0] + pair[1]
            self.merges.append(pair)
            vocabulary.add(merged)

            for idx, seq in enumerate(corpus):
                updated = []
                k = 0
                while k < len(seq):
                    if k + 1 < len(seq) and (seq[k], seq[k + 1]) == pair:
                        updated.append(merged)
                        k += 2
                    else:
                        updated.append(seq[k])
                        k += 1
                corpus[idx] = updated
        return self

    # -- Stage 1: word & unit counting over sentences -------------------------
    def _word_freq(self, corpus: Iterable[str]) -> Counter:
        wf: Counter = Counter()
        for line in corpus:
            n = self.pre.normalizer.normalize(line)
            if n:
                wf.update(n.split(" "))
        return wf

    def _unit_freq(self, word_freq: Counter) -> Counter:
        uf: Counter = Counter()
        for w, f in word_freq.items():
            first = True
            for is_ph, text in self.pre.parts(w):
                if is_ph:
                    first = False
                    continue
                uf[(WORD_MARK + text) if first else text] += f
                first = False
        return uf

    # -- Stage 2: akshara admission with cumulative coverage + DP pruning -----
    def _admit(self, unit_freq: Counter, closure: Set[str], budget: int) -> List[str]:
        total = sum(unit_freq.values())
        if total == 0:
            return []
        vocab: Set[str] = set(closure)
        max_len = max(len(t) for t in vocab)
        admitted: List[str] = []
        cum = 0
        ranked = sorted(unit_freq.items(), key=lambda kv: (-kv[1], kv[0]))
        for rank, (u, f) in enumerate(ranked):
            if cum / total >= self.cfg.coverage or len(admitted) >= budget:
                break
            cum += f
            if u in vocab:
                continue
            if rank >= self.cfg.dp_prune_min_rank:
                _, cost = lattice_shortest_path(u, vocab, max(max_len, len(u)))
                if cost <= self.cfg.dp_prune_cost:
                    continue  # cheap to express: let BPE merge it if useful
            admitted.append(u)
            vocab.add(u)
            max_len = max(max_len, len(u))
        return admitted

    # -- Stage 3: lazy-heap BPE on words --------------------------------------
    def _learn_merges(self, word_counts: Counter, tokens: List[str], mergeable: Set[int],
                      reserved: Set[str]) -> List[Tuple[int, int, int]]:
        seqs = [list(w) for w in word_counts]
        freqs = [word_counts[w] for w in word_counts]
        str_id = {tokens[i]: i for i in mergeable}
        pair_cnt: Dict[Tuple[int, int], int] = defaultdict(int)
        where: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        def add_pairs(wi: int, touched: Optional[Set[Tuple[int, int]]] = None) -> None:
            s, f = seqs[wi], freqs[wi]
            for k in range(len(s) - 1):
                a, b = s[k], s[k + 1]
                if a in mergeable and b in mergeable:
                    p = (a, b)
                    pair_cnt[p] += f
                    where[p].add(wi)
                    if touched is not None:
                        touched.add(p)

        def remove_pairs(wi: int) -> None:
            s, f = seqs[wi], freqs[wi]
            for k in range(len(s) - 1):
                a, b = s[k], s[k + 1]
                if a in mergeable and b in mergeable:
                    p = (a, b)
                    pair_cnt[p] -= f
                    where[p].discard(wi)

        for wi in range(len(seqs)):
            add_pairs(wi)
        heap = [(-c, p) for p, c in pair_cnt.items() if c > 0]
        heapq.heapify(heap)
        merges: List[Tuple[int, int, int]] = []
        banned: Set[Tuple[int, int]] = set()

        while len(tokens) < self.cfg.vocab_size and heap:
            negc, p = heapq.heappop(heap)
            c = pair_cnt.get(p, 0)
            if c <= 0 or p in banned:
                continue
            if -negc != c:  # stale entry -> refresh lazily
                heapq.heappush(heap, (-c, p))
                continue
            if c < self.cfg.min_pair_freq:
                break
            a, b = p
            new_s = tokens[a] + tokens[b]
            if new_s in reserved:  # never collide with <0x41>, <PAD>, ⟨NUM_0⟩ ...
                banned.add(p)
                continue
            nid = str_id.get(new_s)
            if nid is None:
                nid = len(tokens)
                tokens.append(new_s)
                str_id[new_s] = nid
                mergeable.add(nid)
            merges.append((a, b, nid))
            self.merges.append((tokens[a], tokens[b]))

            touched: Set[Tuple[int, int]] = set()
            for wi in list(where[p]):
                remove_pairs(wi)
                s = seqs[wi]
                out: List[int] = []
                k, n = 0, len(s)
                while k < n:
                    if k < n - 1 and s[k] == a and s[k + 1] == b:
                        out.append(nid)
                        k += 2
                    else:
                        out.append(s[k])
                        k += 1
                seqs[wi] = out
                add_pairs(wi, touched)
            pair_cnt.pop(p, None)
            where.pop(p, None)
            for q in touched:
                cq = pair_cnt.get(q, 0)
                if cq > 0:
                    heapq.heappush(heap, (-cq, q))
        return merges

    def _build_budget_report(
        self,
        word_freq: Counter,
        unit_freq: Counter,
        tokens: List[str],
        n_base: int,
        base_tok,
        validation_texts: Optional[Iterable[str]] = None,
        benchmark_texts: Optional[Iterable[str]] = None,
    ) -> BudgetUtilizationReport:
        from .tokenizer import UNK_ID

        cfg = self.cfg
        lang = "hi" if cfg.script in ("devanagari", "hi", "en-hi") else "en"
        unique_atomic_units = len(unit_freq)
        base_vocab_size = n_base
        configured_vocab_size = cfg.vocab_size
        occupied_percentage = (base_vocab_size / configured_vocab_size * 100.0) if configured_vocab_size else 0.0
        remaining_merge_slots = max(0, configured_vocab_size - base_vocab_size)
        rare_unit_count = sum(1 for f in unit_freq.values() if f < cfg.min_pair_freq)

        base_token_set = set(tokens)
        total_unit_freq = sum(unit_freq.values())
        fallback_unit_freq = sum(f for u, f in unit_freq.items() if u not in base_token_set)
        corpus_fallback_rate = (fallback_unit_freq / total_unit_freq * 100.0) if total_unit_freq else 0.0

        total_words = sum(word_freq.values())
        total_tokens = 0
        total_unks = 0
        for w, f in word_freq.items():
            base_ids = base_tok.word_to_base_ids(w)
            total_tokens += len(base_ids) * f
            total_unks += base_ids.count(UNK_ID) * f

        corpus_unk_rate = (total_unks / total_tokens * 100.0) if total_tokens else 0.0
        corpus_fertility = (total_tokens / total_words) if total_words else 0.0

        validation_fertility = None
        if validation_texts is not None:
            val_list = list(validation_texts)
            if val_list:
                validation_fertility = compute_fertility(val_list, base_tok)

        if benchmark_texts is None:
            bench_list = load_benchmark_sentences(language=lang)
        else:
            bench_list = list(benchmark_texts)
        benchmark_fertility = compute_fertility(bench_list, base_tok) if bench_list else None

        return BudgetUtilizationReport(
            unique_atomic_units=unique_atomic_units,
            base_vocab_size=base_vocab_size,
            configured_vocab_size=configured_vocab_size,
            occupied_percentage=occupied_percentage,
            remaining_merge_slots=remaining_merge_slots,
            rare_unit_count=rare_unit_count,
            corpus_unk_rate=corpus_unk_rate,
            corpus_fallback_rate=corpus_fallback_rate,
            corpus_fertility=corpus_fertility,
            validation_fertility=validation_fertility,
            benchmark_fertility=benchmark_fertility,
            language=lang,
            rare_frequency_threshold=cfg.min_pair_freq,
        )

    def measure_budget_utilization(
        self,
        corpus: Iterable[str],
        validation_texts: Optional[Iterable[str]] = None,
        benchmark_texts: Optional[Iterable[str]] = None,
        print_report: bool = True,
    ) -> BudgetUtilizationReport:
        from .tokenizer import Tokenizer, SPECIALS, N_SPECIALS

        cfg = self.cfg
        word_freq = self._word_freq(corpus)
        unit_freq = self._unit_freq(word_freq)

        closure_tokens = self.closure.non_byte_tokens()
        n_fixed = N_SPECIALS + len(self.placeholders) + 256 + len(closure_tokens)
        budget = max(0, min(int(cfg.vocab_size * cfg.max_admitted_frac), cfg.vocab_size - n_fixed - 1))
        admitted = self._admit(unit_freq, set(closure_tokens), budget)

        tokens: List[str] = (list(SPECIALS) + self.placeholders
                             + self.closure.byte_tokens + closure_tokens + admitted)
        n_base = len(tokens)
        base_tok = Tokenizer(tokens, [], n_base, len(self.placeholders), cfg.script)

        report = self._build_budget_report(
            word_freq=word_freq,
            unit_freq=unit_freq,
            tokens=tokens,
            n_base=n_base,
            base_tok=base_tok,
            validation_texts=validation_texts,
            benchmark_texts=benchmark_texts,
        )
        self.budget_report = report
        if print_report:
            report.print_report()
        return report

    # -- General driver --------------------------------------------------------
    def train(
        self,
        corpus,
        validation_texts: Optional[Iterable[str]] = None,
        benchmark_texts: Optional[Iterable[str]] = None,
        print_budget: bool = True,
    ) -> "BPETrainer":
        # Check if caller passed a list of pre-segmented token sequences (unit test mode)
        if corpus and isinstance(corpus, (list, tuple)) and isinstance(corpus[0], (list, tuple)):
            return self._train_on_sequences(corpus)

        from .tokenizer import Tokenizer, SPECIALS, N_SPECIALS

        cfg = self.cfg
        word_freq = self._word_freq(corpus)
        unit_freq = self._unit_freq(word_freq)

        closure_tokens = self.closure.non_byte_tokens()
        n_fixed = N_SPECIALS + len(self.placeholders) + 256 + len(closure_tokens)
        budget = max(0, min(int(cfg.vocab_size * cfg.max_admitted_frac), cfg.vocab_size - n_fixed - 1))
        admitted = self._admit(unit_freq, set(closure_tokens), budget)

        tokens: List[str] = (list(SPECIALS) + self.placeholders
                             + self.closure.byte_tokens + closure_tokens + admitted)
        n_base = len(tokens)

        # encode every distinct word into base ids (lattice DP), then merge
        tmp = Tokenizer(tokens, [], n_base, len(self.placeholders), cfg.script)
        word_counts: Counter = Counter()
        for w, f in word_freq.items():
            word_counts[tuple(tmp.word_to_base_ids(w))] += f

        # Measure & print vocabulary budget utilization before learning merges
        self.budget_report = self._build_budget_report(
            word_freq=word_freq,
            unit_freq=unit_freq,
            tokens=tokens,
            n_base=n_base,
            base_tok=tmp,
            validation_texts=validation_texts,
            benchmark_texts=benchmark_texts,
        )
        if print_budget:
            self.budget_report.print_report()

        byte_end = N_SPECIALS + len(self.placeholders) + 256
        mergeable = set(range(byte_end, n_base))
        reserved = set(tokens[:byte_end])
        merges = self._learn_merges(word_counts, tokens, mergeable, reserved)
        self._learned_merges_full = merges
        tok = Tokenizer(tokens, merges, n_base, len(self.placeholders), cfg.script)
        tok.language = "hi" if cfg.script in ("devanagari", "hi", "en-hi") else "en"
        tok.min_frequency = cfg.min_pair_freq
        tok.training_pair_count = len(corpus) if isinstance(corpus, (list, tuple)) else 0
        tok.training_corpus_identity = cfg.corpus_identity
        tok.training_corpus_revision = cfg.corpus_revision

        if not hasattr(tok, "metadata") or tok.metadata is None:
            tok.metadata = {}

        if self.budget_report:
            if word_freq:
                sample_words = [w for w, _ in word_freq.most_common(1000)]
                self.budget_report.post_bpe_corpus_fertility = compute_fertility(sample_words, tok)
            if validation_texts:
                self.budget_report.post_bpe_validation_fertility = compute_fertility(validation_texts, tok)
            bench = benchmark_texts if benchmark_texts is not None else load_benchmark_sentences(language=tok.language)
            if bench:
                self.budget_report.post_bpe_benchmark_fertility = compute_fertility(bench, tok)

            tok.metadata["budget_utilization"] = self.budget_report.to_dict()

        self._built_tokenizer = tok
        return self

    def train_tokenizer(
        self,
        corpus,
        validation_texts: Optional[Iterable[str]] = None,
        benchmark_texts: Optional[Iterable[str]] = None,
        print_budget: bool = True,
    ) -> "Tokenizer":
        self.train(
            corpus,
            validation_texts=validation_texts,
            benchmark_texts=benchmark_texts,
            print_budget=print_budget,
        )
        if hasattr(self, "_built_tokenizer"):
            return self._built_tokenizer
        from .tokenizer import Tokenizer, SPECIALS
        tokens = list(SPECIALS) + [left + right for left, right in self.merges]
        return Tokenizer(tokens, [(0, 0, i) for i in range(len(tokens))], len(tokens))

    def train_from_file(self, path: str) -> "BPETrainer":
        with open(path, "r", encoding="utf-8") as fh:
            return self.train(fh)
