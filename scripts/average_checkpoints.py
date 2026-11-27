"""CLI to average the last-k or specified model checkpoints using CheckpointAverager."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.optimization import CheckpointAverager


def main() -> None:
    parser = argparse.ArgumentParser(description="Average top-k or last-k checkpoints for improved NMT BLEU.")
    parser.add_argument("--dir", type=Path, default=Path("ai4bharat/samanantar_modern_weights"), help="Directory containing checkpoints")
    parser.add_argument("--pattern", type=str, default="tmodel_*.pt", help="Glob pattern for epoch checkpoints")
    parser.add_argument("--last-k", type=int, default=5, help="Number of latest checkpoints to average")
    parser.add_argument("--checkpoints", type=Path, nargs="*", default=None, help="Explicit list of checkpoint paths to average")
    parser.add_argument("--output", type=Path, default=None, help="Output averaged checkpoint path")
    args = parser.parse_args()

    if args.checkpoints:
        out_path = args.output or (args.checkpoints[-1].parent / f"tmodel_avg_{len(args.checkpoints)}.pt")
        CheckpointAverager.average_checkpoints(list(args.checkpoints), out_path)
    else:
        res = CheckpointAverager.average_last_k(
            args.dir,
            k=args.last_k,
            pattern=args.pattern,
            output_path=args.output,
        )
        if res is None:
            raise FileNotFoundError(f"No matching checkpoints found in {args.dir} with pattern {args.pattern}")


if __name__ == "__main__":
    main()
