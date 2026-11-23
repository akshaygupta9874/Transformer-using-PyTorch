import unittest
from pathlib import Path

from config import get_config
from scripts.fetch_benchmarks import assert_no_benchmark_leakage


class BenchmarkIsolationHardFailTests(unittest.TestCase):
    def setUp(self):
        self.benchmark_dir = Path("data/benchmarks")
        self.clean_records = [
            ("The prime minister gave an important speech on governance.", "प्रधानमंत्री ने सुशासन पर महत्वपूर्ण भाषण दिया।"),
            ("Machine learning requires robust mathematical foundations.", "मशीन लर्निंग के लिए मजबूत गणितीय आधार की आवश्यकता होती है।"),
            ("The monsoon brings seasonal relief to agriculture.", "मानसून कृषि के लिए मौसमी राहत लेकर आता है।"),
        ]
        # Real benchmark pair present in data/benchmarks/flores200_devtest.tsv
        self.contaminated_records = [
            ("Hello world.", "नमस्ते दुनिया।"),
        ]

    def test_clean_records_pass_isolation_check(self):
        """Clean pairs must pass benchmark isolation without error."""
        try:
            assert_no_benchmark_leakage(self.clean_records, benchmark_dir=self.benchmark_dir)
        except AssertionError as e:
            self.fail(f"Clean records unexpectedly triggered benchmark leakage: {e}")

    def test_contaminated_record_raises_hard_assertion_error(self):
        """Contaminated benchmark pairs must raise an AssertionError and halt."""
        mixed_records = self.clean_records + self.contaminated_records
        with self.assertRaises(AssertionError) as ctx:
            assert_no_benchmark_leakage(mixed_records, benchmark_dir=self.benchmark_dir)

        error_message = str(ctx.exception)
        self.assertIn("CONTAMINATION ERROR", error_message)
        self.assertIn("Hello world.", error_message)

    def test_train_pipeline_leakage_is_never_swallowed(self):
        """Verify train.py get_ds does not catch or swallow AssertionError when leakage occurs."""
        from unittest.mock import patch, MagicMock
        from train import get_ds

        test_config = dict(get_config())
        test_config["leakage_check_samples"] = 10

        # Create a mock dataset that injects a leaked benchmark pair
        mock_raw_data = [
            {"translation": {"en": "Hello world.", "hi": "नमस्ते दुनिया।"}},
        ] * 30

        with patch("train.load_dataset") as mock_load:
            mock_load.return_value = mock_raw_data

            # Must raise AssertionError and NOT swallow it
            with self.assertRaises(AssertionError) as ctx:
                get_ds(test_config)

            self.assertIn("CONTAMINATION ERROR", str(ctx.exception))

    def test_fetch_benchmark_network_failure_raises_and_never_creates_mock_data(self):
        """Verify fetch_benchmark raises RuntimeError on failure and never writes synthetic mock data."""
        import tempfile
        from unittest.mock import patch
        from scripts.fetch_benchmarks import fetch_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            out_tsv = Path(tmpdir) / "test_benchmark.tsv"
            with patch("datasets.load_dataset", side_effect=ConnectionError("Simulated network failure")):
                with self.assertRaises(RuntimeError) as ctx:
                    fetch_benchmark("ai4bharat/IN22-Gen", "en-hi", "test", out_tsv)

                self.assertIn("Failed to fetch authentic benchmark dataset", str(ctx.exception))
                self.assertIn("cannot be populated with mock or synthetic data", str(ctx.exception))
                # Crucial requirement: output_tsv must NOT exist with fake/mock sentences!
                self.assertFalse(out_tsv.exists(), "Frozen benchmark file was improperly created on failure!")

    def test_fetch_benchmark_zero_pairs_raises_error(self):
        """Verify fetch_benchmark raises RuntimeError if 0 sentence pairs are extracted."""
        import tempfile
        from unittest.mock import patch
        from scripts.fetch_benchmarks import fetch_benchmark

        with tempfile.TemporaryDirectory() as tmpdir:
            out_tsv = Path(tmpdir) / "empty_benchmark.tsv"
            with patch("datasets.load_dataset", return_value=[]):
                with self.assertRaises(RuntimeError) as ctx:
                    fetch_benchmark("ai4bharat/IN22-Gen", "en-hi", "test", out_tsv)

                self.assertIn("0 valid sentence pairs", str(ctx.exception))
                self.assertFalse(out_tsv.exists())

    def test_fetch_benchmark_success_creates_authentic_tsv(self):
        """Verify fetch_benchmark writes valid TSV on success without mock flags."""
        import tempfile
        from unittest.mock import patch
        from scripts.fetch_benchmarks import fetch_benchmark

        sample_hf_rows = [
            {"translation": {"en": "Authentic English sentence 1.", "hi": "प्रामाणिक हिंदी वाक्य १।"}},
            {"translation": {"en": "Authentic English sentence 2.", "hi": "प्रामाणिक हिंदी वाक्य २।"}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_tsv = Path(tmpdir) / "authentic.tsv"
            with patch("datasets.load_dataset", return_value=sample_hf_rows):
                metadata = fetch_benchmark("facebook/flores", "eng_Latn-hin_Deva", "devtest", out_tsv)

                self.assertTrue(out_tsv.exists())
                self.assertEqual(metadata["rows"], 2)
                self.assertEqual(metadata["fingerprints_count"], 2)
                self.assertNotIn("mock", metadata)

                lines = out_tsv.read_text(encoding="utf-8").strip().split("\n")
                self.assertEqual(len(lines), 2)
                self.assertIn("प्रामाणिक हिंदी वाक्य १।", lines[0])


if __name__ == "__main__":
    unittest.main()
