"""Download and lock frozen benchmark test sets: IN22-Gen, IN22-Conv, FLORES-200 devtest (B1).

Strictly isolates test sets with SHA-256 contamination tracking so they are
never exposed to training, tokenization, or quality filtering.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_DIR = Path("data/benchmarks")


def compute_pair_hash(src: str, tgt: str) -> str:
    cleaned = f"{' '.join(src.strip().split())}\0{' '.join(tgt.strip().split())}"
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def fetch_benchmark(
    dataset_name: str,
    config: str,
    split: str,
    output_tsv: Path,
    token: str | bool | None = None,
) -> dict:
    """Download an external benchmark dataset and save as clean two-column TSV."""
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise RuntimeError("Install datasets library to fetch benchmarks.") from err

    print(f"Fetching {dataset_name} ({config}, split={split})...")
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    
    hf_token = token if token is not None else (os.environ.get("HF_TOKEN") or True)
    try:
        ds = load_dataset(dataset_name, config, split=split, token=hf_token, trust_remote_code=True)
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


def fetch_flores200(output_tsv: Path, token: str | bool | None = None) -> dict:
    """Download FLORES-200 devtest, with automatic fallback to open Muennighoff mirror."""
    try:
        return fetch_benchmark("facebook/flores", "eng_Latn-hin_Deva", "devtest", output_tsv, token=token)
    except Exception as primary_err:
        print(f"Direct 'facebook/flores' fetch failed; using open FLORES-200 mirror ('Muennighoff/flores200')...")
        try:
            from datasets import load_dataset
            d_hi = load_dataset("Muennighoff/flores200", "hin_Deva", split="devtest", trust_remote_code=True)
            d_en = load_dataset("Muennighoff/flores200", "eng_Latn", split="devtest", trust_remote_code=True)
            output_tsv.parent.mkdir(parents=True, exist_ok=True)
            pairs = []
            hashes = set()
            with output_tsv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                for row_en, row_hi in zip(d_en, d_hi):
                    src_c = " ".join(row_en["sentence"].strip().split())
                    tgt_c = " ".join(row_hi["sentence"].strip().split())
                    if src_c and tgt_c:
                        writer.writerow([src_c, tgt_c])
                        pairs.append((src_c, tgt_c))
                        hashes.add(compute_pair_hash(src_c, tgt_c))
            metadata = {
                "name": "Muennighoff/flores200",
                "config": "eng_Latn-hin_Deva",
                "split": "devtest",
                "rows": len(pairs),
                "fingerprints_count": len(hashes),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "license": "Frozen research benchmark (CC-BY-SA 4.0)",
            }
            print(f"Successfully fetched {len(pairs):,} FLORES-200 sentence pairs.")
            return metadata
        except Exception as fallback_err:
            raise RuntimeError(
                f"Failed to fetch FLORES-200 from both primary and mirror sources: {primary_err} | Fallback: {fallback_err}"
            ) from fallback_err


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
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token (hf_...)")
    parser.add_argument("--flores-only", action="store_true", help="Download only FLORES-200 devtest benchmark")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # 1. FLORES-200 devtest (ungated via open mirror)
    manifest["flores200_devtest"] = fetch_flores200(args.output_dir / "flores200_devtest.tsv", token=args.token)

    if args.flores_only:
        print("\n[Notice] --flores-only specified. Skipping IN22 benchmarks.")
    else:
        # 2. IN22-Gen & IN22-Conv (AI4Bharat gated repository)
        try:
            manifest["in22_gen"] = fetch_benchmark(
                "ai4bharat/IN22-Gen", "en-hi", "test", args.output_dir / "in22_gen.tsv", token=args.token
            )
            manifest["in22_conv"] = fetch_benchmark(
                "ai4bharat/IN22-Conv", "en-hi", "test", args.output_dir / "in22_conv.tsv", token=args.token
            )
        except RuntimeError as e:
            print("\n" + "=" * 78)
            print("AUTHENTICATION NOTICE FOR IN22 (AI4Bharat):")
            print("ai4bharat/IN22-Gen and IN22-Conv are gated datasets on Hugging Face.")
            print("To download IN22:")
            print("  1. Visit: https://huggingface.co/datasets/ai4bharat/IN22-Gen and click 'Agree'.")
            print("  2. Get a free token at: https://huggingface.co/settings/tokens")
            print("  3. Run: python scripts/fetch_benchmarks.py --token <YOUR_HF_TOKEN>")
            print("     (Or run: huggingface-cli login)")
            print("FLORES-200 (1,012 sentences) was already downloaded successfully!")
            print("=" * 78 + "\n")

    manifest_path = args.output_dir / "BENCHMARKS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Benchmark isolation manifest created at {manifest_path}")


if __name__ == "__main__":
    main()
