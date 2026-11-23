import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


class TokenizerSerializer:
    VERSION = 1

    @staticmethod
    def compute_hash(tokens: Sequence[str], merges: Sequence) -> str:
        """Deterministic SHA-256 fingerprint over vocabulary tokens and merges."""
        h = hashlib.sha256()
        for t in tokens:
            h.update(t.encode("utf-8", "surrogatepass"))
            h.update(b"\x00")
        for m in merges:
            h.update(f"{m[0]},{m[1]},{m[2] if len(m) > 2 else ''}\n".encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def to_dict(tok: "Tokenizer") -> dict:
        from .akshara import AksharaSegmenter
        from .normalizer import DevanagariNormalizer

        norm_config = tok.normalizer.get_policy() if hasattr(tok, "normalizer") and tok.normalizer else {}
        if not norm_config and hasattr(tok, "normalization_policy") and tok.normalization_policy:
            norm_config = tok.normalization_policy
        if not norm_config:
            dev = tok.script in ("devanagari", "hi", "en-hi")
            norm_config = DevanagariNormalizer(devanagari_rules=dev, unify_danda=dev).get_policy()

        tok_hash = TokenizerSerializer.compute_hash(tok.tokens, tok.merges)

        lang = getattr(tok, "language", None)
        if not lang:
            lang = "hi" if tok.script in ("devanagari", "hi", "en-hi") else "en"

        tok_version = getattr(tok, "version", "2.0")
        vocab_size = len(tok.tokens)
        merge_count = len(tok.merges)
        corpus_id = getattr(tok, "training_corpus_identity", "samanantar")
        corpus_rev = getattr(tok, "training_corpus_revision", "v0.2")
        pair_count = getattr(tok, "training_pair_count", 0)
        min_freq = getattr(tok, "min_frequency", 2)
        seg_version = getattr(AksharaSegmenter, "VERSION", "2.0")

        metadata = {
            # Standard snake_case
            "tokenizer_version": tok_version,
            "language": lang,
            "vocabulary_size": vocab_size,
            "merge_count": merge_count,
            "training_corpus_identity": corpus_id,
            "training_corpus_revision": corpus_rev,
            "training_corpus_identity_and_revision": f"{corpus_id}@{corpus_rev}",
            "training_pair_count": pair_count,
            "tokenizer_hash": tok_hash,
            "minimum_frequency": min_freq,
            "normalizer_configuration": norm_config,
            "normalization_policy": norm_config,  # backward compatibility alias
            "segmentation_implementation_version": seg_version,
            # Space-separated human-readable aliases matching reviewer prompt exactly
            "tokenizer version": tok_version,
            "vocabulary size": vocab_size,
            "merge count": merge_count,
            "training corpus identity and revision": f"{corpus_id}@{corpus_rev}",
            "training pair count": pair_count,
            "tokenizer hash": tok_hash,
            "minimum frequency": min_freq,
            "normalizer configuration": norm_config,
            "segmentation implementation version": seg_version,
        }

        # Merge any custom metadata already stored on tok
        if hasattr(tok, "metadata") and isinstance(tok.metadata, dict):
            for k, v in tok.metadata.items():
                if k not in metadata:
                    metadata[k] = v

        return {
            "version": TokenizerSerializer.VERSION,
            # Standard snake_case
            "tokenizer_version": tok_version,
            "language": lang,
            "vocabulary_size": vocab_size,
            "merge_count": merge_count,
            "training_corpus_identity": corpus_id,
            "training_corpus_revision": corpus_rev,
            "training_corpus_identity_and_revision": f"{corpus_id}@{corpus_rev}",
            "training_pair_count": pair_count,
            "tokenizer_hash": tok_hash,
            "minimum_frequency": min_freq,
            "normalizer_configuration": norm_config,
            "segmentation_implementation_version": seg_version,
            # Space-separated human-readable aliases matching reviewer prompt exactly
            "tokenizer version": tok_version,
            "vocabulary size": vocab_size,
            "merge count": merge_count,
            "training corpus identity and revision": f"{corpus_id}@{corpus_rev}",
            "training pair count": pair_count,
            "tokenizer hash": tok_hash,
            "minimum frequency": min_freq,
            "normalizer configuration": norm_config,
            "segmentation implementation version": seg_version,
            # Structural data
            "script": tok.script,
            "num_placeholders": tok.num_placeholders,
            "n_base": tok.n_base,
            "tokens": tok.tokens,
            "merges": [list(m) for m in tok.merges],
            "metadata": metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> "Tokenizer":
        from .tokenizer import Tokenizer, SPECIALS, N_SPECIALS
        from .akshara import AksharaSegmenter

        meta = d.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {}

        tok_version = (
            meta.get("tokenizer_version")
            or meta.get("tokenizer version")
            or d.get("tokenizer_version")
            or d.get("tokenizer version")
            or "2.0"
        )
        lang = (
            meta.get("language")
            or d.get("language")
            or ("hi" if d.get("script") in ("devanagari", "hi", "en-hi") else "en")
        )
        pair_count = (
            meta.get("training_pair_count")
            or meta.get("training pair count")
            or d.get("training_pair_count")
            or d.get("training pair count")
            or 0
        )
        corpus_id = (
            meta.get("training_corpus_identity")
            or d.get("training_corpus_identity")
            or "samanantar"
        )
        corpus_rev = (
            meta.get("training_corpus_revision")
            or d.get("training_corpus_revision")
            or "v0.2"
        )
        corpus_comb = (
            meta.get("training_corpus_identity_and_revision")
            or meta.get("training corpus identity and revision")
            or d.get("training_corpus_identity_and_revision")
            or d.get("training corpus identity and revision")
        )
        if corpus_comb and "@" in corpus_comb and (corpus_id == "samanantar" and corpus_rev == "v0.2"):
            parts = corpus_comb.split("@", 1)
            corpus_id, corpus_rev = parts[0], parts[1]

        min_freq = (
            meta.get("minimum_frequency")
            or meta.get("minimum frequency")
            or d.get("minimum_frequency")
            or d.get("minimum frequency")
            or 2
        )
        norm_config = (
            meta.get("normalizer_configuration")
            or meta.get("normalizer configuration")
            or meta.get("normalization_policy")
            or d.get("normalizer_configuration")
            or d.get("normalizer configuration")
        )
        tok_hash = (
            meta.get("tokenizer_hash")
            or meta.get("tokenizer hash")
            or d.get("tokenizer_hash")
            or d.get("tokenizer hash")
        )
        seg_version = (
            meta.get("segmentation_implementation_version")
            or meta.get("segmentation implementation version")
            or d.get("segmentation_implementation_version")
            or d.get("segmentation implementation version")
            or getattr(AksharaSegmenter, "VERSION", "2.0")
        )

        # Format 1: Claude's total function format (version 1)
        if "tokens" in d and "n_base" in d:
            merges = [tuple(m) for m in d.get("merges", [])]
            tok = Tokenizer(
                tokens=d["tokens"],
                merges=merges,
                n_base=d["n_base"],
                num_placeholders=d.get("num_placeholders", 0),
                script=d.get("script", "devanagari"),
                normalization_policy=norm_config,
                language=lang,
                version=tok_version,
                training_pair_count=pair_count,
                training_corpus_identity=corpus_id,
                training_corpus_revision=corpus_rev,
                min_frequency=min_freq,
                metadata=meta,
            )
            tok.tokenizer_hash = tok_hash or TokenizerSerializer.compute_hash(tok.tokens, tok.merges)
            tok.segmentation_implementation_version = seg_version
            return tok

        # Format 2: Legacy v2 dictionary format (joiner_policy, vocabulary, merges)
        if "vocabulary" in d:
            vocab = d["vocabulary"]
            # Reconstruct ordered tokens list from vocabulary dict
            sorted_items = sorted(vocab.items(), key=lambda kv: kv[1])
            tokens = [tok_item for tok_item, _ in sorted_items]

            # Reconstruct merges as (left_id, right_id, merged_id) if possible
            str_to_id = {tok_item: i for i, tok_item in enumerate(tokens)}
            merges: List[Tuple[int, int, int]] = []
            for item in d.get("merges", []):
                if len(item) == 2:
                    left, right = item[0], item[1]
                    merged_s = left + right
                    if left in str_to_id and right in str_to_id and merged_s in str_to_id:
                        merges.append((str_to_id[left], str_to_id[right], str_to_id[merged_s]))
                elif len(item) == 3:
                    merges.append((int(item[0]), int(item[1]), int(item[2])))

            num_placeholders = 0
            n_base = len(tokens)
            script = "devanagari" if d.get("language") in ("hi", "devanagari", "en-hi") else "latin"

            if not norm_config and "joiner_policy" in d:
                norm_config = {
                    "joiner_policy": d["joiner_policy"],
                    "preserve_zwj": d["joiner_policy"] != "remove_all",
                    "preserve_zwnj": d["joiner_policy"] != "remove_all",
                }

            tok = Tokenizer(
                tokens=tokens,
                merges=merges,
                n_base=n_base,
                num_placeholders=num_placeholders,
                script=script,
                is_legacy=True,
                normalization_policy=norm_config,
                language=lang,
                version=tok_version,
                training_pair_count=pair_count,
                training_corpus_identity=corpus_id,
                training_corpus_revision=corpus_rev,
                min_frequency=min_freq,
                metadata=meta,
            )
            tok.tokenizer_hash = tok_hash or TokenizerSerializer.compute_hash(tokens, merges)
            tok.segmentation_implementation_version = seg_version
            return tok

        raise ValueError(f"Unsupported tokenizer file schema: keys={list(d.keys())}")

    @classmethod
    def save(cls, tok: "Tokenizer", path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(cls.to_dict(tok), fh, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def load(cls, path) -> "Tokenizer":
        with open(Path(path), "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
