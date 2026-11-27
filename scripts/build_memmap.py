"""Pre-tokenize English-Hindi corpora into uint16 memmap shards (with optional Akshara BPE-dropout variants)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config, get_tokenizer_path
from research.memmap_data import build_memmap_from_tokenizers


def _load_pairs_and_weights(args) -> tuple[list[tuple[str, str]], list[float] | None]:
    pairs: list[tuple[str, str]] = []
    weights: list[float] | None = None

    if args.input_tsv:
        has_weights = False
        rows: list[list[str]] = []
        with args.input_tsv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    rows.append(row)
                    if len(row) >= 3:
                        has_weights = True
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
        if has_weights:
            weights = []
        for row in rows:
            pairs.append((row[0], row[1]))
            if weights is not None:
                weights.append(float(row[2]) if len(row) >= 3 and row[2] else 1.0)
        return pairs, weights

    from datasets import load_dataset, load_from_disk
    hf_path = Path(args.hf_dataset)
    if hf_path.exists() and hf_path.is_dir():
        ds = load_from_disk(str(hf_path))
    else:
        ds = load_dataset(args.hf_dataset, name=args.hf_config, split="train")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    for item in ds:
        if "translation" in item:
            pairs.append((item["translation"]["en"], item["translation"]["hi"]))
        else:
            pairs.append((item["src"], item["tgt"]))
    return pairs, None


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Build memory-mapped uint16 token datasets for fast NMT training.")
    parser.add_argument("--input-tsv", type=Path, default=None, help="Clean 2- or 3-column TSV (src \\t tgt [\\t weight])")
    parser.add_argument("--hf-dataset", type=str, default="ai4bharat/samanantar", help="HuggingFace dataset or saved disk path")
    parser.add_argument("--hf-config", type=str, default="hi", help="HF dataset config")
    parser.add_argument("--out-dir", type=Path, default=Path("data/memmap/samanantar"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=cfg["seq_len"])
    parser.add_argument("--variants", type=int, default=0, help="Number of offline source BPE-dropout variants (0 disables)")
    parser.add_argument("--dropout-p", type=float, default=0.05, help="Source Akshara BPE-dropout probability for variants")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes (Windows spawn safe)")
    parser.add_argument("--no-placeholders", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 13))
    args = parser.parse_args()

    tok_src = get_tokenizer_path(cfg, cfg["lang_src"])
    tok_tgt = tok_src if cfg.get("shared_vocabulary") else get_tokenizer_path(cfg, cfg["lang_tgt"])
    pairs, weights = _load_pairs_and_weights(args)
    print(f"Building memmap at {args.out_dir} ({len(pairs):,} pairs, variants={args.variants}, workers={args.workers})...")
    build_memmap_from_tokenizers(
        pairs=pairs,
        tok_src_path=tok_src,
        tok_tgt_path=tok_tgt,
        out_dir=args.out_dir,
        split=args.split,
        seq_len=args.seq_len,
        use_placeholders=not args.no_placeholders,
        n_src_variants=args.variants,
        src_dropout_p=args.dropout_p,
        seed=args.seed,
        n_proc=args.workers,
        quality_scores=weights,
    )
    print(f"Memmap build complete: {args.out_dir}")


if __name__ == "__main__":
    main()
