"""Build English and Hindi tokenizers from Samanantar.

Usage:
    python build_tokenizers.py
    python build_tokenizers.py --max-samples 100000
"""

import argparse
import sys
import time
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config
from tokenizer import Tokenizer, load_benchmark_sentences


def load_training_data(config):
    dataset_kwargs = {"split": "train"}

    if config.get("dataset_config"):
        dataset_kwargs["name"] = config["dataset_config"]

    return load_dataset(
        config["datasource"],
        **dataset_kwargs,
    )


def build_tokenizers(max_samples=None):
    config = get_config()

    print("Loading Samanantar...", flush=True)

    dataset = load_training_data(config)

    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    print(
        f"Loaded {len(dataset):,} sentence pairs.",
        flush=True,
    )

    def texts_for(language):
        for item in dataset:
            if "translation" in item:
                yield item["translation"][language]

            elif "src" in item and "tgt" in item:
                if language == config["lang_src"]:
                    yield item["src"]
                else:
                    yield item["tgt"]

            else:
                raise KeyError(
                    f"Unsupported Samanantar columns: {list(item)}"
                )

    for language in (
        config["lang_src"],
        config["lang_tgt"],
    ):
        tokenizer_path = Path(
            config["tokenizer_file"].format(language)
        )

        vocab_size = (
            config
            .get("tokenizer_vocab_sizes", {})
            .get(language, 16000 if language == "en" else 40000)
        )

        total_sentences = len(dataset)

        print()
        print("=" * 70)
        print(
            f"Training {language} tokenizer | "
            f"{total_sentences:,} sentences | "
            f"vocab={vocab_size:,}",
            flush=True,
        )
        print("=" * 70, flush=True)

        started = time.perf_counter()

        # --------------------------------------------------------------
        # Progress state
        # --------------------------------------------------------------

        preprocessing_started = time.perf_counter()
        pair_index_started = None
        bpe_started = None

        last_preprocessed = 0
        last_indexed = 0

        # --------------------------------------------------------------
        # Preprocessing progress
        # --------------------------------------------------------------

        def report_preprocessing(sentence_count):
            nonlocal last_preprocessed
            nonlocal preprocessing_started
            nonlocal pair_index_started
            nonlocal bpe_started

            # Report every 1,000 sentences.
            if (
                sentence_count != total_sentences
                and sentence_count - last_preprocessed < 1000
            ):
                return

            last_preprocessed = sentence_count

            elapsed = (
                time.perf_counter()
                - preprocessing_started
            )

            rate = (
                sentence_count / elapsed
                if elapsed
                else 0.0
            )

            remaining = max(
                0,
                total_sentences - sentence_count,
            )

            eta = (
                remaining / rate
                if rate
                else 0.0
            )

            print(
                f"[PREPROCESS] "
                f"{sentence_count:,}/{total_sentences:,} "
                f"({sentence_count / total_sentences:.1%}) | "
                f"{rate:,.0f} sent/s | "
                f"ETA {eta / 60:.1f} min",
                flush=True,
            )

            if sentence_count >= total_sentences:
                pair_index_started = time.perf_counter()

        # --------------------------------------------------------------
        # Pair-index progress
        # --------------------------------------------------------------

        def report_pair_index(
            processed,
            total,
            unique_pairs,
        ):
            nonlocal last_indexed
            nonlocal bpe_started

            # Report every 1,000 sequences.
            if (
                processed != total
                and processed - last_indexed < 1000
            ):
                return

            last_indexed = processed

            if pair_index_started is None:
                return

            elapsed = (
                time.perf_counter()
                - pair_index_started
            )

            rate = (
                processed / elapsed
                if elapsed
                else 0.0
            )

            remaining = max(
                0,
                total - processed,
            )

            eta = (
                remaining / rate
                if rate
                else 0.0
            )

            print(
                f"[PAIR INDEX] "
                f"{processed:,}/{total:,} "
                f"({processed / total:.1%}) | "
                f"pairs={unique_pairs:,} | "
                f"{rate:,.0f} seq/s | "
                f"ETA {eta / 60:.1f} min",
                flush=True,
            )

            if processed >= total:
                bpe_started = time.perf_counter()

        # --------------------------------------------------------------
        # BPE progress
        # --------------------------------------------------------------

        def report(
            merge_count,
            vocabulary_size,
            pair_frequency,
            bpe_target,
        ):
            phase_started = (
                bpe_started
                or started
            )

            elapsed = (
                time.perf_counter()
                - phase_started
            )

            if merge_count == 0:
                print(
                    f"[BPE] Starting | "
                    f"vocab={vocabulary_size:,}/"
                    f"{bpe_target:,} | "
                    f"best_pair_freq={pair_frequency:,}",
                    flush=True,
                )
                return

            rate = (
                merge_count / elapsed
                if elapsed
                else 0.0
            )

            remaining = max(
                0,
                bpe_target - vocabulary_size,
            )

            eta = (
                remaining / rate
                if rate
                else 0.0
            )

            print(
                f"[BPE] "
                f"merges={merge_count:,} | "
                f"vocab={vocabulary_size:,}/"
                f"{bpe_target:,} | "
                f"best_pair_freq={pair_frequency:,} | "
                f"{rate:.2f} merges/s | "
                f"ETA {eta / 60:.1f} min",
                flush=True,
            )

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

        bench_sentences = load_benchmark_sentences(language=language)
        tokenizer = Tokenizer.train(
            texts_for(language),
            language=language,
            vocab_size=vocab_size,
            min_frequency=2,
            training_pair_count=total_sentences,
            corpus_identity=config.get("datasource", "samanantar"),
            corpus_revision=config.get("dataset_config", "v0.2"),
            benchmark_texts=bench_sentences,
            progress_callback=report,
            preprocessing_callback=report_preprocessing,
            initialization_callback=report_pair_index,
        )

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        print(
            f"Saving tokenizer → {tokenizer_path}",
            flush=True,
        )

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(tokenizer_path)

        if hasattr(tokenizer, "metadata") and "budget_utilization" in tokenizer.metadata:
            b_stats = tokenizer.metadata["budget_utilization"]
            print(
                f"[BUDGET SUMMARY] "
                f"Base vocab: {b_stats.get('base_vocab_size', 'N/A')} "
                f"({b_stats.get('occupied_percentage', 'N/A')}%) | "
                f"Remaining merges: {b_stats.get('remaining_merge_slots', 'N/A')} | "
                f"Corpus fertility: {b_stats.get('corpus_fertility', 'N/A')} tokens/word | "
                f"Benchmark fertility: {b_stats.get('benchmark_fertility', 'N/A')} tokens/word",
                flush=True,
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"✓ Saved {language} tokenizer | "
            f"vocab={tokenizer.get_vocab_size():,} | "
            f"time={elapsed / 60:.1f} min",
            flush=True,
        )

    print()
    print(
        "Special IDs: "
        "<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Samanantar English-Hindi tokenizers"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for a quick prototype",
    )

    args = parser.parse_args()

    build_tokenizers(args.max_samples)