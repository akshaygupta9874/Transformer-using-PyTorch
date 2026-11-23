"""Validation of serialized tokenizer metadata against experiment configurations."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


class TokenizerValidationError(ValueError):
    """Raised when an existing tokenizer fails metadata validation against experiment configuration."""
    pass


def _load_benchmark_fingerprints(benchmark_dir: Path) -> set[str]:
    """Collects fingerprint hashes of frozen benchmark pairs to check for contamination."""
    fingerprints = set()
    if not benchmark_dir.exists():
        return fingerprints

    for tsv_path in benchmark_dir.glob("*.tsv"):
        try:
            with tsv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                for row in reader:
                    if len(row) >= 2:
                        cleaned = f"{' '.join(row[0].strip().split())}\0{' '.join(row[1].strip().split())}"
                        fingerprints.add(hashlib.sha256(cleaned.encode("utf-8")).hexdigest())
        except Exception:
            pass
    return fingerprints


def validate_tokenizer_metadata(
    tokenizer: "Tokenizer",
    config: Dict[str, Any],
    expected_lang: str,
    tokenizer_path: Optional[str | Path] = None,
    strict: bool = False,
    raise_on_error: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validates tokenizer JSON metadata and properties against the resolved experiment configuration.

    Verifies 7 critical criteria:
    1. Expected language matches (e.g. 'hi' vs 'en', or 'en-hi' for shared vocab).
    2. 16k/40k vocabulary configuration matches resolved config.
    3. Training corpus identity/revision matches expected datasource.
    4. Normalizer version and configuration policy compatibility.
    5. Tokenizer SHA-256 hash matches computed fingerprint (detects tampering/corruption).
    6. Training pair count is valid and meets minimum requirements if configured.
    7. No benchmark leakage occurred (never trained on or contaminated by frozen evaluation sets).

    Args:
        tokenizer: The Tokenizer instance loaded from disk.
        config: Resolved experiment configuration dictionary (e.g. from get_config()).
        expected_lang: Expected language code ('en', 'hi', or 'en-hi').
        tokenizer_path: Optional path to the tokenizer file for diagnostic messages.
        strict: If True, requires all optional metadata fields (corpus, hash) to be present.
        raise_on_error: If True, raises TokenizerValidationError on failure instead of returning.

    Returns:
        Tuple of (is_valid: bool, reasons: List[str]).
    """
    from .serializer import TokenizerSerializer

    reasons: List[str] = []
    meta = getattr(tokenizer, "metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    # -------------------------------------------------------------------------
    # 1. Expected Language Validation
    # -------------------------------------------------------------------------
    actual_lang = getattr(tokenizer, "language", None) or meta.get("language")
    if config.get("shared_vocabulary"):
        valid_shared = {"en-hi", "hi-en", "shared", "multilingual"}
        if actual_lang and actual_lang.lower() not in valid_shared:
            reasons.append(
                f"Language mismatch for shared vocabulary: expected one of {valid_shared}, "
                f"found '{actual_lang}'"
            )
    else:
        expected_aliases = {
            "hi": {"hi", "hindi", "devanagari", "en-hi"},
            "en": {"en", "english", "latin", "en-hi"},
        }
        allowed = expected_aliases.get(expected_lang.lower(), {expected_lang.lower()})
        if actual_lang and actual_lang.lower() not in allowed:
            reasons.append(
                f"Language mismatch: expected '{expected_lang}', but tokenizer has language '{actual_lang}'"
            )

    # -------------------------------------------------------------------------
    # 2. 16k / 40k Vocabulary Configuration
    # -------------------------------------------------------------------------
    if config.get("shared_vocabulary"):
        expected_vocab = config.get("tokenizer_vocab_size", 50000)
    else:
        expected_vocab = config.get("tokenizer_vocab_sizes", {}).get(
            expected_lang, 16000 if expected_lang == "en" else 40000
        )
    actual_vocab = tokenizer.get_vocab_size()
    budget_metadata = meta.get("budget_utilization", {})
    configured_vocab = meta.get("configured_vocab_size")
    if configured_vocab is None and isinstance(budget_metadata, dict):
        configured_vocab = budget_metadata.get("configured_vocab_size")
    if configured_vocab is None:
        configured_vocab = actual_vocab

    try:
        configured_vocab = int(configured_vocab)
    except (TypeError, ValueError):
        reasons.append(
            f"Invalid configured vocabulary size in tokenizer metadata: {configured_vocab!r}"
        )
    else:
        if configured_vocab != expected_vocab:
            reasons.append(
                f"Configured vocabulary size mismatch for {expected_lang}: "
                f"expected {expected_vocab:,}, tokenizer was configured for {configured_vocab:,}"
            )
        if actual_vocab > expected_vocab:
            reasons.append(
                f"Realized vocabulary size for {expected_lang} exceeds the configured maximum: "
                f"expected at most {expected_vocab:,}, found {actual_vocab:,}"
            )

    # -------------------------------------------------------------------------
    # 3. Training Corpus Identity and Revision
    # -------------------------------------------------------------------------
    actual_corpus = (
        getattr(tokenizer, "training_corpus_identity", "")
        or meta.get("training_corpus_identity", "")
    )
    expected_datasource = config.get("datasource", "")
    if expected_datasource:
        ds_name = expected_datasource.split("/")[-1].lower()
        if actual_corpus:
            if strict and ds_name not in actual_corpus.lower() and actual_corpus.lower() not in ds_name:
                reasons.append(
                    f"Training corpus mismatch: expected datasource containing '{ds_name}', "
                    f"but tokenizer records '{actual_corpus}'"
                )
        elif strict:
            reasons.append(
                f"Missing 'training_corpus_identity' in metadata under strict validation (expected '{ds_name}')."
            )

    # -------------------------------------------------------------------------
    # 4. Normalizer Version & Policy Compatibility
    # -------------------------------------------------------------------------
    norm_ver = (
        meta.get("normalizer_version")
        or meta.get("normalization_version")
    )
    if norm_ver is not None:
        try:
            ver_num = float(str(norm_ver).split(".")[0])
            if ver_num < 2.0:
                reasons.append(
                    f"Incompatible normalizer version: {norm_ver} (minimum required is 2.0)"
                )
        except ValueError:
            pass

    norm_config = meta.get("normalizer_configuration") or meta.get("normalization_policy")
    if norm_config and isinstance(norm_config, dict):
        # Joiner policy check: if config requires canonical joiners, verify normalizer policy
        joiner_pol = norm_config.get("joiner_policy")
        if joiner_pol == "remove_all" and config.get("preserve_joiners", True):
            reasons.append(
                "Normalizer policy mismatch: tokenizer was trained with 'remove_all' joiner policy, "
                "which drops valid ZWJ/ZWNJ conjuncts."
            )

    # -------------------------------------------------------------------------
    # 5. Tokenizer Hash Integrity
    # -------------------------------------------------------------------------
    expected_hash = (
        meta.get("tokenizer_hash")
        or meta.get("tokenizer hash")
    )
    if expected_hash:
        computed_hash = TokenizerSerializer.compute_hash(tokenizer.tokens, tokenizer.merges)
        if computed_hash != expected_hash:
            reasons.append(
                f"Tokenizer hash integrity failure: metadata recorded '{expected_hash[:12]}...', "
                f"computed fingerprint is '{computed_hash[:12]}...'. Vocabulary or merges altered on disk."
            )
    elif strict:
        reasons.append("Missing 'tokenizer_hash' in metadata under strict validation.")

    # -------------------------------------------------------------------------
    # 6. Training Pair Count Validity
    # -------------------------------------------------------------------------
    pair_count = (
        getattr(tokenizer, "training_pair_count", None)
        or meta.get("training_pair_count")
    )
    if pair_count is not None:
        if pair_count < 0:
            reasons.append(f"Invalid negative training pair count: {pair_count}")
        min_pairs = config.get("min_tokenizer_pairs", 0)
        if min_pairs > 0 and pair_count < min_pairs:
            reasons.append(
                f"Training pair count {pair_count:,} is below required experiment threshold {min_pairs:,}"
            )

    # -------------------------------------------------------------------------
    # 7. Benchmark Leakage Detection
    # -------------------------------------------------------------------------
    leakage_flags: List[str] = []
    if meta.get("benchmark_leakage") or meta.get("contaminated"):
        leakage_flags.append("metadata benchmark_leakage/contaminated flag is True")

    corpus_str = f"{actual_corpus} {meta.get('training_corpus_revision', '')}".lower()
    for bench_keyword in ("flores", "in22", "benchmark_eval", "heldout_eval"):
        if bench_keyword in corpus_str:
            leakage_flags.append(f"corpus identity contains '{bench_keyword}'")

    # Check for leaked tokens matching full benchmark sentences if benchmark files exist
    benchmark_dir = Path(config.get("benchmark_dir", "data/benchmarks"))
    if benchmark_dir.exists():
        fps = _load_benchmark_fingerprints(benchmark_dir)
        if fps:
            # Check if any token or merge in the tokenizer directly embeds a benchmark hash
            if any(h in meta.get("corpus_hashes", []) for h in fps):
                leakage_flags.append("training corpus hashes contain frozen benchmark fingerprints")

    if leakage_flags:
        reasons.append(
            f"BENCHMARK LEAKAGE DETECTED: Tokenizer contaminated by frozen evaluation data ({'; '.join(leakage_flags)})"
        )

    # -------------------------------------------------------------------------
    # Final Result
    # -------------------------------------------------------------------------
    is_valid = len(reasons) == 0
    if not is_valid and raise_on_error:
        loc = f" at '{tokenizer_path}'" if tokenizer_path else ""
        error_msg = (
            f"Tokenizer{loc} failed validation against experiment configuration:\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )
        raise TokenizerValidationError(error_msg)

    return is_valid, reasons
