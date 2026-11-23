"""Download the Samanantar Hindi configuration into the project data cache.

The external IN22 and FLORES benchmarks are intentionally not downloaded or
mixed here: they remain frozen evaluation-only resources.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Samanantar English-Hindi training data locally.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/samanantar_hi"))
    parser.add_argument("--max-samples", type=int, default=None, help="Use only for smoke tests; omit for the full training split.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("ai4bharat/samanantar", "hi", split="train", cache_dir=str(args.output / "hf_cache"))
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset.save_to_disk(str(args.output / "dataset"))
    manifest = {
        "name": "Samanantar",
        "config": "hi",
        "split": "train",
        "rows": len(dataset),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "license": "Verify the upstream dataset card before each use.",
        "frozen_evaluation_sets_excluded": ["IN22-Gen", "IN22-Conv", "FLORES-200 devtest"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Cached {len(dataset):,} Samanantar en-hi pairs at {args.output / 'dataset'}")


if __name__ == "__main__":
    main()
