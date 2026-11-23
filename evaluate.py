"""Comprehensive multi-metric evaluation runner with paired bootstrap significance (B10, B11, B18)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.evaluation import (
    comprehensive_error_taxonomy,
    paired_bootstrap_corpus,
    paired_bootstrap_sentence,
    score_corpus,
)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score English-to-Hindi model outputs with full metric suite and significance tests.")
    parser.add_argument("--references", type=Path, required=True, help="Reference translations file")
    parser.add_argument("--hypotheses", type=Path, required=True, help="System hypotheses file")
    parser.add_argument("--sources", type=Path, default=None, help="Optional source English file for copy/leak analysis")
    parser.add_argument("--baseline-hypotheses", type=Path, default=None, help="Optional baseline hypotheses for paired bootstrap significance")
    parser.add_argument("--omit-unavailable-bleu", action="store_true", help="Omit BLEU if SacreBLEU is unavailable instead of using pure-Python fallback")
    parser.add_argument("--output", type=Path, default=Path("evaluation.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    refs = read_lines(args.references)
    hyps = read_lines(args.hypotheses)
    srcs = read_lines(args.sources) if args.sources else None

    # 1. Standard corpus metrics (BLEU, chrF)
    metrics = score_corpus(hyps, refs, omit_unavailable_bleu=args.omit_unavailable_bleu)

    # 2. Automated linguistic error taxonomy
    taxonomy = comprehensive_error_taxonomy(hyps, refs, sources=srcs)

    results = {
        "metrics": metrics,
        "error_taxonomy": taxonomy,
    }

    # 3. Paired bootstrap significance testing against baseline if provided
    if args.baseline_hypotheses:
        base_hyps = read_lines(args.baseline_hypotheses)
        print(f"Running paired bootstrap significance testing ({args.bootstrap_samples} resamples)...")
        results["significance"] = {
            "chrf": paired_bootstrap_corpus(base_hyps, hyps, refs, metric_name="chrf", samples=args.bootstrap_samples),
            "bleu": paired_bootstrap_corpus(base_hyps, hyps, refs, metric_name="bleu", samples=args.bootstrap_samples),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Print formatted markdown summary table
    print("\n" + "=" * 50)
    print("           EVALUATION REPORT")
    print("=" * 50)
    print(f"| Metric | Score |")
    print(f"|---|---|")
    bleu_val = metrics.get("bleu")
    if bleu_val is None:
        print(f"| BLEU      | N/A (SacreBLEU unavailable) |")
    else:
        label = "SacreBLEU" if metrics.get("bleu_engine") == "sacrebleu" else "BLEU (py)"
        print(f"| {label:<9} | {bleu_val:.2f} |")
    print(f"| chrF2++   | {metrics['chrf']:.2f} |")
    
    if "significance" in results:
        chrf_sig = results["significance"]["chrf"]
        bleu_sig = results["significance"]["bleu"]
        print(f"\n--- Paired Bootstrap Significance (vs Baseline) ---")
        print(f"chrF delta: {chrf_sig['mean_delta']:+.2f} (95% CI: [{chrf_sig['ci_low']:+.2f}, {chrf_sig['ci_high']:+.2f}], p={chrf_sig['p_value']:.4f}, sig={chrf_sig['significant_p05']})")
        print(f"BLEU delta: {bleu_sig['mean_delta']:+.2f} (95% CI: [{bleu_sig['ci_low']:+.2f}, {bleu_sig['ci_high']:+.2f}], p={bleu_sig['p_value']:.4f}, sig={bleu_sig['significant_p05']})")

    print(f"\n--- Linguistic Error Flags (B18) ---")
    for flag, rate in taxonomy.get("rates", {}).items():
        count = taxonomy["counts"][flag]
        print(f"  {flag:<30}: {count} occurrences ({rate * 100:.1f}%)")
    print("=" * 50)
    print(f"Detailed results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
