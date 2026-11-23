import tempfile
import unittest
from pathlib import Path

from tokenizer import (
    BPETrainer,
    BudgetUtilizationReport,
    Tokenizer,
    compute_fertility,
    load_benchmark_sentences,
)


class BudgetUtilizationTests(unittest.TestCase):
    def setUp(self):
        self.hi_corpus = [
            "नमस्ते दुनिया",
            "भारत एक विशाल और सुंदर देश है",
            "ज्ञान और शिक्षा से ही समाज का विकास संभव है",
            "हम सब भारतीय हैं",
        ]
        self.val_texts = [
            "नमस्ते भारत",
            "शिक्षा ही शक्ति है",
        ]
        self.bench_texts = [
            "यह एक मानक वाक्य है",
            "कंप्यूटर विज्ञान एक आधुनिक विषय है",
        ]

    def test_load_benchmark_sentences(self):
        hi_bench = load_benchmark_sentences("hi", max_sentences=2)
        self.assertGreater(len(hi_bench), 0)
        self.assertIsInstance(hi_bench[0], str)

        en_bench = load_benchmark_sentences("en", max_sentences=2)
        self.assertGreater(len(en_bench), 0)
        self.assertIsInstance(en_bench[0], str)

    def test_budget_utilization_metrics_computation(self):
        trainer = BPETrainer(vocab_size=5000, min_frequency=2)
        report = trainer.measure_budget_utilization(
            self.hi_corpus,
            validation_texts=self.val_texts,
            benchmark_texts=self.bench_texts,
            print_report=False,
        )

        self.assertIsInstance(report, BudgetUtilizationReport)
        self.assertGreater(report.unique_atomic_units, 0)
        self.assertGreater(report.base_vocab_size, 0)
        self.assertEqual(report.configured_vocab_size, 5000)
        self.assertAlmostEqual(
            report.occupied_percentage,
            (report.base_vocab_size / 5000) * 100.0,
            places=2,
        )
        self.assertEqual(report.remaining_merge_slots, 5000 - report.base_vocab_size)
        self.assertGreaterEqual(report.rare_unit_count, 0)
        # Total function property: UNK rate must be exactly 0.0%
        self.assertEqual(report.corpus_unk_rate, 0.0)
        # Fallback rate within valid percentage range
        self.assertGreaterEqual(report.corpus_fallback_rate, 0.0)
        self.assertLessEqual(report.corpus_fallback_rate, 100.0)
        # Fertility: positive tokens/word
        self.assertGreater(report.corpus_fertility, 0.0)
        self.assertIsNotNone(report.validation_fertility)
        self.assertGreater(report.validation_fertility, 0.0)
        self.assertIsNotNone(report.benchmark_fertility)
        self.assertGreater(report.benchmark_fertility, 0.0)

        # Dictionary representation
        d = report.to_dict()
        self.assertIn("unique_atomic_units", d)
        self.assertIn("base_vocab_size", d)
        self.assertIn("configured_vocab_size", d)
        self.assertIn("occupied_percentage", d)
        self.assertIn("remaining_merge_slots", d)
        self.assertIn("rare_unit_count", d)
        self.assertIn("corpus_unk_rate", d)
        self.assertIn("corpus_fallback_rate", d)
        self.assertIn("corpus_fertility", d)
        self.assertIn("validation_fertility", d)
        self.assertIn("benchmark_fertility", d)

    def test_print_report_execution(self):
        trainer = BPETrainer(vocab_size=2000, min_frequency=1)
        report = trainer.measure_budget_utilization(
            self.hi_corpus,
            validation_texts=self.val_texts,
            benchmark_texts=self.bench_texts,
            print_report=True,
        )
        self.assertEqual(report.configured_vocab_size, 2000)

    def test_tokenizer_training_embeds_budget_metadata(self):
        tok = Tokenizer.train(
            self.hi_corpus,
            language="hi",
            vocab_size=3000,
            min_frequency=1,
            validation_texts=self.val_texts,
            benchmark_texts=self.bench_texts,
            print_budget=False,
        )

        self.assertIn("budget_utilization", tok.metadata)
        budget = tok.metadata["budget_utilization"]
        self.assertEqual(budget["configured_vocab_size"], 3000)
        self.assertGreater(budget["base_vocab_size"], 0)
        self.assertEqual(budget["corpus_unk_rate"], 0.0)
        self.assertIn("corpus_fertility", budget)
        self.assertIn("validation_fertility", budget)
        self.assertIn("benchmark_fertility", budget)

        # Verify serialization and persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tok.json"
            tok.save(save_path)
            loaded = Tokenizer.load(save_path)
            self.assertIn("budget_utilization", loaded.metadata)
            loaded_budget = loaded.metadata["budget_utilization"]
            self.assertEqual(loaded_budget["configured_vocab_size"], 3000)
            self.assertEqual(loaded_budget["base_vocab_size"], budget["base_vocab_size"])
            self.assertEqual(loaded_budget["corpus_unk_rate"], 0.0)

    def test_english_latin_budget_metrics(self):
        en_corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming linguistics and machine translation.",
            "PyTorch provides flexible deep learning abstractions.",
        ]
        trainer = BPETrainer(vocab_size=2000, min_frequency=1)
        trainer.cfg.script = "latin"
        report = trainer.measure_budget_utilization(
            en_corpus,
            validation_texts=["Quick tests pass."],
            benchmark_texts=["Standard benchmark."],
            print_report=False,
        )
        self.assertEqual(report.language, "en")
        self.assertEqual(report.corpus_unk_rate, 0.0)
        self.assertGreater(report.corpus_fertility, 0.0)

    def test_vocab_statistics_incorporates_budget(self):
        trainer = BPETrainer(vocab_size=4000, min_frequency=1)
        trainer.train(self.hi_corpus, print_budget=False)
        stats = trainer.vocab_statistics()
        self.assertIn("unique_base_units", stats)
        self.assertIn("merges_count", stats)
        self.assertIn("occupied_percentage", stats)
        self.assertIn("remaining_merge_slots", stats)
        self.assertIn("corpus_unk_rate", stats)
        self.assertIn("corpus_fertility", stats)


if __name__ == "__main__":
    unittest.main()
