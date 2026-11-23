"""Metric evaluation, paired-bootstrap significance testing (B11), and linguistic error taxonomy (B18)."""

from __future__ import annotations

from collections import Counter
import math
import random
import re
from typing import Optional, List, Dict, Any, Sequence

# Pre-compiled punctuation regex matching 13a tokenization for Latin and Devanagari (including danda)
_13A_PUNC_RE = re.compile(r"([,.:;!?¿¡()\"«»„”{}<>[\]%#$@^~_/|*+\-\u0964\u0965])")


def tokenize_13a(line: str) -> list[str]:
    """Tokenize a string according to SacreBLEU 13a standard rules, handling Latin and Brahmic scripts."""
    if not line:
        return []
    line = _13A_PUNC_RE.sub(r" \1 ", line)
    return line.strip().split()


def pure_python_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    """
    Pure-Python standard-library calculation of corpus BLEU (Papineni et al., 2002) with brevity penalty.

    Features:
    - 13a international/Devanagari word and punctuation tokenization
    - Clipped n-gram precision for orders 1..max_order
    - Brevity Penalty (BP) matching SacreBLEU
    - Smoothing 1 (Chen & Cherry, 2014) preventing zero BLEU when higher orders have zero matches
    - Dynamic effective order adaptation for very short sentences
    """
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have equal length.")
    if not hypotheses:
        return 0.0

    c = 0
    r = 0
    clipped_matches = [0] * (max_order + 1)
    total_candidates = [0] * (max_order + 1)

    for hyp_str, ref_str in zip(hypotheses, references):
        hyp_toks = tokenize_13a(hyp_str)
        ref_toks = tokenize_13a(ref_str)
        c += len(hyp_toks)
        r += len(ref_toks)

        for n in range(1, max_order + 1):
            hyp_ngrams = Counter(tuple(hyp_toks[i : i + n]) for i in range(len(hyp_toks) - n + 1))
            ref_ngrams = Counter(tuple(ref_toks[i : i + n]) for i in range(len(ref_toks) - n + 1))
            for ng, cnt in hyp_ngrams.items():
                clipped_matches[n] += min(cnt, ref_ngrams.get(ng, 0))
                total_candidates[n] += cnt

    if c == 0:
        return 0.0

    bp = 1.0 if c > r else math.exp(1.0 - r / c)

    eff_order = max_order
    while eff_order > 1 and total_candidates[eff_order] == 0:
        eff_order -= 1

    log_p_sum = 0.0
    for n in range(1, eff_order + 1):
        cand = total_candidates[n]
        match = clipped_matches[n]
        if cand == 0:
            p_n = 1.0 / (c + 1) if smooth else 0.0
        elif match == 0:
            if smooth and clipped_matches[1] > 0:
                p_n = 1.0 / (cand + 1)
            else:
                return 0.0
        else:
            p_n = match / cand

        if p_n <= 0.0:
            return 0.0
        log_p_sum += (1.0 / eff_order) * math.log(p_n)

    return float(bp * math.exp(log_p_sum) * 100.0)


def score_corpus(
    hypotheses: list[str],
    references: list[str],
    compute_fallback_bleu: bool = True,
    omit_unavailable_bleu: bool = False,
) -> dict[str, Any]:
    """Calculate corpus BLEU and chrF using SacreBLEU with pure-Python fallback.

    When SacreBLEU is unavailable:
    - If omit_unavailable_bleu is True (or compute_fallback_bleu is False):
      BLEU is explicitly marked as None and reported as 'unavailable' to prevent misinterpretation.
    - Otherwise (default):
      BLEU is accurately calculated using pure_python_bleu and reported with bleu_engine='pure_python'.
    """
    if len(hypotheses) != len(references):
        raise ValueError("Hypotheses and references must have equal length.")
    try:
        import sacrebleu
        return {
            "bleu": float(sacrebleu.corpus_bleu(hypotheses, [references]).score),
            "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
            "bleu_engine": "sacrebleu",
            "sacrebleu_available": True,
        }
    except ImportError:
        from research.decoding import pure_python_chrf
        chrf_scores = [pure_python_chrf(h, r) * 100.0 for h, r in zip(hypotheses, references)]
        mean_chrf = sum(chrf_scores) / max(1, len(chrf_scores))

        if omit_unavailable_bleu or not compute_fallback_bleu:
            return {
                "bleu": None,
                "chrf": float(mean_chrf),
                "bleu_engine": "unavailable",
                "sacrebleu_available": False,
            }

        bleu_score = pure_python_bleu(hypotheses, references)
        return {
            "bleu": float(bleu_score),
            "chrf": float(mean_chrf),
            "bleu_engine": "pure_python",
            "sacrebleu_available": False,
        }


def paired_bootstrap_sentence(
    baseline: list[float], candidate: list[float], samples: int = 10000, seed: int = 13
) -> dict[str, float]:
    """Paired bootstrap resampling for sentence-level utilities (e.g. COMET)."""
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Both non-empty score lists must have the same length.")
    rng = random.Random(seed)
    count = len(baseline)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        deltas.append(sum(candidate[i] - baseline[i] for i in indices) / count)
    deltas.sort()
    lower = deltas[int(samples * 0.025)]
    upper = deltas[int(samples * 0.975)]
    p_value = 2 * min(sum(d <= 0 for d in deltas) / samples, sum(d >= 0 for d in deltas) / samples)
    return {
        "mean_delta": (sum(candidate) - sum(baseline)) / count,
        "ci_low": lower,
        "ci_high": upper,
        "p_value": min(1.0, p_value),
        "significant_p05": p_value < 0.05,
    }


def paired_bootstrap_corpus(
    baseline_hyps: list[str],
    candidate_hyps: list[str],
    references: list[str],
    metric_name: str = "chrf",
    samples: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    """Paired bootstrap resampling (Koehn 2004) for corpus metrics (BLEU, chrF).
    
    Resamples test set sentences with replacement to determine if candidate is
    statistically significantly superior to baseline.
    """
    n = len(references)
    if not (len(baseline_hyps) == len(candidate_hyps) == n):
        raise ValueError("Baseline, candidate, and reference lists must all have the same length.")

    try:
        import sacrebleu
        use_sacrebleu = True
    except ImportError:
        use_sacrebleu = False
        from research.decoding import pure_python_chrf

    rng = random.Random(seed)
    deltas = []

    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_refs = [references[i] for i in indices]
        sampled_base = [baseline_hyps[i] for i in indices]
        sampled_cand = [candidate_hyps[i] for i in indices]

        if use_sacrebleu:
            if metric_name.lower() == "bleu":
                s_base = sacrebleu.corpus_bleu(sampled_base, [sampled_refs]).score
                s_cand = sacrebleu.corpus_bleu(sampled_cand, [sampled_refs]).score
            else:
                s_base = sacrebleu.corpus_chrf(sampled_base, [sampled_refs]).score
                s_cand = sacrebleu.corpus_chrf(sampled_cand, [sampled_refs]).score
        else:
            if metric_name.lower() == "bleu":
                s_base = pure_python_bleu(sampled_base, sampled_refs)
                s_cand = pure_python_bleu(sampled_cand, sampled_refs)
            else:
                s_base = sum(pure_python_chrf(h, r) * 100.0 for h, r in zip(sampled_base, sampled_refs)) / max(1, len(indices))
                s_cand = sum(pure_python_chrf(h, r) * 100.0 for h, r in zip(sampled_cand, sampled_refs)) / max(1, len(indices))

        delta = s_cand - s_base
        deltas.append(delta)

    deltas.sort()
    lower = deltas[int(samples * 0.025)]
    upper = deltas[int(samples * 0.975)]
    p_value = 2.0 * min(sum(d <= 0 for d in deltas) / samples, sum(d >= 0 for d in deltas) / samples)

    return {
        "metric": metric_name,
        "mean_delta": sum(deltas) / samples,
        "ci_low": lower,
        "ci_high": upper,
        "p_value": min(1.0, p_value),
        "significant_p05": p_value < 0.05,
    }


def paired_bootstrap(
    baseline: list[float], candidate: list[float], samples: int = 10000, seed: int = 13
) -> dict[str, float]:
    """Backwards-compatible sentence-level paired bootstrap."""
    return paired_bootstrap_sentence(baseline, candidate, samples=samples, seed=seed)


def exact_copy_rate(hypotheses: list[str], expected_spans: list[list[str]]) -> float:
    """Fraction of sentences where all expected spans are preserved verbatim in output."""
    if len(hypotheses) != len(expected_spans):
        raise ValueError("Each hypothesis must have an expected span list.")
    return sum(all(span in hypothesis for span in spans) for hypothesis, spans in zip(hypotheses, expected_spans)) / max(1, len(hypotheses))


def evaluate_copy_preservation(
    sources: Sequence[str],
    hypotheses: Sequence[str],
    references: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """
    Evaluates entity copy preservation (numbers and URLs) from source to translation hypothesis.
    
    Robustly handles:
    - Exact URL preservation
    - Latin vs Devanagari numerals (e.g. 123 in source matching १२३ in hypothesis)
    - Currency and date number conversions
    """
    from research.placeholders import canonical_numeric_value

    URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
    NUM_RE = re.compile(r"(?<![\w])[-+]?[0-9०-९]+(?:[.,][0-9०-९]+)*(?![\w])")

    total_sentences = len(sources)
    total_urls = 0
    preserved_urls = 0
    total_nums = 0
    preserved_nums = 0
    perfect_copy_sentences = 0

    for i in range(total_sentences):
        src = sources[i]
        hyp = hypotheses[i]

        src_urls = URL_RE.findall(src)
        src_nums = NUM_RE.findall(src)

        sent_urls_ok = True
        for u in src_urls:
            total_urls += 1
            if u in hyp:
                preserved_urls += 1
            else:
                sent_urls_ok = False

        sent_nums_ok = True
        hyp_nums = NUM_RE.findall(hyp)
        hyp_canon = {canonical_numeric_value(h) for h in hyp_nums}

        for n in src_nums:
            total_nums += 1
            src_canon = canonical_numeric_value(n)
            if src_canon in hyp_canon or n in hyp:
                preserved_nums += 1
            else:
                sent_nums_ok = False

        if (not src_urls and not src_nums) or (sent_urls_ok and sent_nums_ok):
            perfect_copy_sentences += 1

    return {
        "url_copy_rate": (preserved_urls / max(1, total_urls)) if total_urls > 0 else 1.0,
        "number_copy_rate": (preserved_nums / max(1, total_nums)) if total_nums > 0 else 1.0,
        "perfect_copy_sentence_rate": perfect_copy_sentences / max(1, total_sentences),
        "total_urls": total_urls,
        "preserved_urls": preserved_urls,
        "total_numbers": total_nums,
        "preserved_numbers": preserved_nums,
        "total_sentences": total_sentences,
    }


def comprehensive_error_taxonomy(
    hypotheses: list[str],
    references: list[str],
    sources: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Automated linguistic error taxonomy (B18).
    
    Analyzes:
    - Number omissions & numeral script mismatches
    - URL corruptions
    - Length ratio anomalies (over/under translation)
    - Latin token leakage in Hindi output
    - Devanagari script purity
    """
    categories = Counter()
    total = len(hypotheses)
    if total == 0:
        return {}

    LATIN_WORD = re.compile(r"[a-zA-Z]{3,}")
    NUM_REGEX = re.compile(r"[0-9०-९]+")
    URL_REGEX = re.compile(r"https?://\S+")

    for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
        src = sources[i] if sources and i < len(sources) else ""

        # 1. Number omission
        ref_nums = set(NUM_REGEX.findall(ref))
        hyp_nums = set(NUM_REGEX.findall(hyp))
        if ref_nums and not hyp_nums:
            categories["number_omission_total"] += 1
        elif ref_nums and not ref_nums.intersection(hyp_nums):
            categories["number_mismatch"] += 1

        # 2. URL preservation
        if src:
            src_urls = set(URL_REGEX.findall(src))
            if src_urls and not all(url in hyp for url in src_urls):
                categories["url_corruption"] += 1

        # 3. Length distortion
        ref_len = max(1, len(ref.split()))
        hyp_len = len(hyp.split())
        if hyp_len > 1.8 * ref_len:
            categories["over_translation"] += 1
        elif hyp_len < 0.55 * ref_len:
            categories["under_translation"] += 1

        # 4. English leakage in Hindi output
        latin_words = LATIN_WORD.findall(hyp)
        if latin_words:
            categories["untranslated_english_words"] += 1

        # 5. Empty output
        if not hyp.strip():
            categories["empty_output"] += 1

    return {
        "counts": dict(categories),
        "rates": {k: v / total for k, v in categories.items()},
        "total_evaluated": total,
    }


def lightweight_error_taxonomy(hypotheses: list[str], references: list[str]) -> dict[str, int]:
    """Backwards-compatible lightweight error flags."""
    return comprehensive_error_taxonomy(hypotheses, references)["counts"]
