"""Real data-quality filtering CLI: Length ratios + LaBSE + CometKiwi (B2).

Cleans parallel TSVs, eliminates benchmark contamination, and optionally applies
neural semantic similarity (LaBSE) and reference-free quality estimation (CometKiwi).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.fetch_benchmarks import assert_no_benchmark_leakage
except ImportError:
    from fetch_benchmarks import assert_no_benchmark_leakage
from research.data import DatasetProvenance, ParallelCleaner, pair_fingerprint, write_datasheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter English-Hindi bitext using deterministic and neural filters.")
    parser.add_argument("input", type=Path, help="Input two-column TSV (English \\t Hindi)")
    parser.add_argument("output", type=Path, help="Clean output TSV")
    parser.add_argument("--source-name", default="CustomCorpus", help="Corpus provenance name")
    parser.add_argument("--source-url", default="local", help="Source URL or license identifier")
    parser.add_argument("--license", default="Research-only", help="License")
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--labse-min", type=float, default=None, help="Minimum LaBSE cosine similarity (e.g. 0.70)")
    parser.add_argument("--cometkiwi-min", type=float, default=None, help="Minimum CometKiwi QE score (e.g. 0.65)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cleaner = ParallelCleaner(min_chars=args.min_chars, max_chars=args.max_chars, max_length_ratio=args.max_length_ratio)
    
    # Read raw candidate pairs
    raw_pairs: list[tuple[str, str]] = []
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                raw_pairs.append((row[0].strip(), row[1].strip()))

    total_initial = len(raw_pairs)
    print(f"Loaded {total_initial:,} raw candidate pairs from {args.input}")

    # Step 1: Deterministic cleaning & deduplication
    stage1_kept: list[tuple[str, str]] = []
    seen = set()
    rejected_stats = {"empty_or_short": 0, "length_ratio": 0, "html": 0, "duplicate": 0, "leakage": 0}

    for src, tgt in raw_pairs:
        src_norm, tgt_norm = cleaner.normalize(src), cleaner.normalize(tgt)
        reason = cleaner.reason_to_reject(src_norm, tgt_norm)
        if reason:
            if reason in ("empty", "too_short", "too_long"):
                rejected_stats["empty_or_short"] += 1
            elif reason == "length_ratio":
                rejected_stats["length_ratio"] += 1
            elif reason == "html":
                rejected_stats["html"] += 1
            continue

        fp = pair_fingerprint(src_norm, tgt_norm)
        if fp in seen:
            rejected_stats["duplicate"] += 1
            continue

        seen.add(fp)
        stage1_kept.append((src_norm, tgt_norm))

    print(f"Stage 1 (Deterministic): Retained {len(stage1_kept):,} / {total_initial:,} pairs.")

    # Step 2: Ensure zero leakage against frozen benchmarks
    assert_no_benchmark_leakage(stage1_kept)

    # Step 3: Optional LaBSE filtering
    if args.labse_min is not None and stage1_kept:
        from research.adapters import labse_similarity
        print(f"Running LaBSE semantic similarity filter (threshold >= {args.labse_min})...")
        sources = [p[0] for p in stage1_kept]
        targets = [p[1] for p in stage1_kept]
        sims = labse_similarity(sources, targets)
        stage2_kept = [p for p, sim in zip(stage1_kept, sims) if sim >= args.labse_min]
        print(f"Stage 2 (LaBSE): Retained {len(stage2_kept):,} / {len(stage1_kept):,} pairs.")
    else:
        stage2_kept = stage1_kept

    # Step 4: Optional CometKiwi reference-free QE filtering
    if args.cometkiwi_min is not None and stage2_kept:
        from research.adapters import comet_scores
        print(f"Running CometKiwi reference-free QE filter (threshold >= {args.cometkiwi_min})...")
        sources = [p[0] for p in stage2_kept]
        targets = [p[1] for p in stage2_kept]
        scores = comet_scores(sources, targets, model_name="Unbabel/wmt22-cometkiwi-da")
        final_kept = [p for p, sc in zip(stage2_kept, scores) if sc >= args.cometkiwi_min]
        print(f"Stage 3 (CometKiwi): Retained {len(final_kept):,} / {len(stage2_kept):,} pairs.")
    else:
        final_kept = stage2_kept

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for src, tgt in final_kept:
            writer.writerow([src, tgt])

    # Save auditable datasheet
    datasheet_path = args.output.with_suffix(".datasheet.json")
    provenance = DatasetProvenance(
        name=args.source_name,
        source=args.source_url,
        license=args.license,
        language_pair="en-hi",
        pulled_at=datetime.now(timezone.utc).isoformat(),
        split="train_filtered",
    )
    statistics = {
        "initial_pairs": total_initial,
        "final_retained": len(final_kept),
        "retained_fraction": len(final_kept) / max(1, total_initial),
        "rejected_deterministic": rejected_stats,
        "labse_threshold": args.labse_min,
        "cometkiwi_threshold": args.cometkiwi_min,
    }
    write_datasheet(datasheet_path, provenance, statistics)
    print(f"Filtering complete! Saved {len(final_kept):,} pairs to {args.output}")
    print(f"Datasheet recorded at {datasheet_path}")


if __name__ == "__main__":
    main()
