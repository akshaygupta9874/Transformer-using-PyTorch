"""Create a clean, deduplicated parallel TSV with a provenance datasheet.

Input is deliberately local and explicit: downloading large corpora is an
operator decision, while this script makes every selection reproducible.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from research.data import DatasetProvenance, ParallelCleaner, pair_fingerprint, write_datasheet


def main():
    parser = argparse.ArgumentParser(description="Clean a two-column English-Hindi TSV.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    args = parser.parse_args()
    cleaner, seen, kept, rejected = ParallelCleaner(max_length_ratio=args.max_length_ratio), set(), 0, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", newline="") as source, args.output.open("w", encoding="utf-8", newline="") as target:
        reader, writer = csv.reader(source, delimiter="\t"), csv.writer(target, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                rejected += 1
                continue
            english, hindi = cleaner.normalize(row[0]), cleaner.normalize(row[1])
            fingerprint = pair_fingerprint(english, hindi)
            if cleaner.reason_to_reject(english, hindi) or fingerprint in seen:
                rejected += 1
                continue
            seen.add(fingerprint)
            writer.writerow([english, hindi])
            kept += 1
    provenance = DatasetProvenance(args.source_name, args.source_url, args.license, "en-hi", datetime.now(timezone.utc).isoformat(), "train")
    write_datasheet(args.output.with_suffix(".datasheet.json"), provenance, {"kept": kept, "rejected": rejected})
    print(f"Kept {kept:,} pairs; rejected {rejected:,}. Datasheet saved beside output.")


if __name__ == "__main__":
    main()
