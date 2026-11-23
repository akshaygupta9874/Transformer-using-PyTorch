"""Download and lock frozen benchmark test sets: IN22-Gen, IN22-Conv, FLORES-200 devtest (B1).

Strictly isolates test sets with SHA-256 contamination tracking so they are
never exposed to training, tokenization, or quality filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_DIR = Path("data/benchmarks")


def compute_pair_hash(src: str, tgt: str) -> str:
    cleaned = f"{' '.join(src.strip().split())}\0{' '.join(tgt.strip().split())}"
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def fetch_benchmark(dataset_name: str, config: str, split: str, output_tsv: Path) -> dict:
    """Download an external benchmark dataset and save as clean two-column TSV."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise RuntimeError("Install datasets library to fetch benchmarks.") from err

    print(f"Fetching {dataset_name} ({config}, split={split})...")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        ds = load_dataset(dataset_name, config, split=split)
    except Exception as err:
        # Never generate synthetic or mock benchmark data on failure: hard fail.
        if output_tsv.exists():
            try:
                output_tsv.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to fetch authentic benchmark dataset '{dataset_name}' (config={config}, split={split}) from HuggingFace: {err}\n"
            f"Frozen benchmark test sets must be authentic and cannot be populated with mock or synthetic data. "
            f"Please verify your network connection or authenticate with `huggingface-cli login` if required."
        ) from err

    pairs = []
    hashes = set()
    try:
        with output_tsv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in ds:
                # Handle standard HF schema formats
                src, tgt = "", ""
                if "translation" in row:
                    src = row["translation"].get("en", "")
                    tgt = row["translation"].get("hi", "")
                elif "sentence_en" in row and "sentence_hi" in row:
                    src, tgt = row["sentence_en"], row["sentence_hi"]
                elif "src" in row and "tgt" in row:
                    src, tgt = row["src"], row["tgt"]
                
                if src and tgt:
                    src_c = " ".join(src.strip().split())
                    tgt_c = " ".join(tgt.strip().split())
                    writer.writerow([src_c, tgt_c])
                    pairs.append((src_c, tgt_c))
                    hashes.add(compute_pair_hash(src_c, tgt_c))
    except Exception as err:
        if output_tsv.exists():
            try:
                output_tsv.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Error writing benchmark TSV '{output_tsv}': {err}") from err

    if not pairs:
        if output_tsv.exists():
            try:
                output_tsv.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Dataset '{dataset_name}' ({config}, split={split}) yielded 0 valid sentence pairs!"
        )

    metadata = {
        "name": dataset_name,
        "config": config,
        "split": split,
        "rows": len(pairs),
        "fingerprints_count": len(hashes),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "license": "Frozen research benchmark",
    }
    return metadata


def load_all_benchmark_fingerprints(benchmark_dir: Path = BENCHMARK_DIR) -> set[str]:
    """Collect all fingerprints across all downloaded frozen benchmarks."""
    fingerprints = set()
    if not benchmark_dir.exists():
        return fingerprints

    for tsv_path in benchmark_dir.glob("*.tsv"):
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    fingerprints.add(compute_pair_hash(row[0], row[1]))
    return fingerprints


def assert_no_benchmark_leakage(records: list[tuple[str, str]], benchmark_dir: Path = BENCHMARK_DIR) -> None:
    """Hard programmatic assertion preventing benchmark pairs from leaking into training."""
    frozen_fingerprints = load_all_benchmark_fingerprints(benchmark_dir)
    if not frozen_fingerprints:
        return

    leaked = []
    for src, tgt in records:
        h = compute_pair_hash(src, tgt)
        if h in frozen_fingerprints:
            leaked.append((src, tgt))

    if leaked:
        raise AssertionError(
            f"CONTAMINATION ERROR: Found {len(leaked)} training pairs matching frozen evaluation sets! "
            f"First leaked pair: {leaked[0]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and isolate frozen benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=BENCHMARK_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # 1. IN22-Gen & IN22-Conv (AI4Bharat)
    manifest["in22_gen"] = fetch_benchmark("ai4bharat/IN22-Gen", "en-hi", "test", args.output_dir / "in22_gen.tsv")
    manifest["in22_conv"] = fetch_benchmark("ai4bharat/IN22-Conv", "en-hi", "test", args.output_dir / "in22_conv.tsv")
    
    # 2. FLORES-200 devtest (Facebook/Meta)
    manifest["flores200_devtest"] = fetch_benchmark("facebook/flores", "eng_Latn-hin_Deva", "devtest", args.output_dir / "flores200_devtest.tsv")

    manifest_path = args.output_dir / "BENCHMARKS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Benchmark isolation manifest created at {manifest_path}")


if __name__ == "__main__":
    main()
