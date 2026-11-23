import tempfile
import unittest
from pathlib import Path

from config import get_config
from tokenizer import (
    Tokenizer,
    TokenizerValidationError,
    validate_tokenizer_metadata,
)
from tokenizer.serializer import TokenizerSerializer


class TokenizerValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = get_config()

    def test_existing_tokenizers_pass_validation(self):
        """Verify standard 16k/40k tokenizers on disk pass validation against experiment config."""
        hi_path = Path("data/tokenizers/tokenizer_hi.akshara.v2.json")
        en_path = Path("data/tokenizers/tokenizer_en.akshara.v2.json")

        if hi_path.exists():
            tok_hi = Tokenizer.load(hi_path)
            is_valid, reasons = validate_tokenizer_metadata(
                tok_hi, self.config, expected_lang="hi", tokenizer_path=hi_path
            )
            self.assertTrue(is_valid, f"Hindi tokenizer failed validation: {reasons}")

        if en_path.exists():
            tok_en = Tokenizer.load(en_path)
            is_valid, reasons = validate_tokenizer_metadata(
                tok_en, self.config, expected_lang="en", tokenizer_path=en_path
            )
            self.assertTrue(is_valid, f"English tokenizer failed validation: {reasons}")

    def test_expected_language_mismatch_detected(self):
        """Verify passing English tokenizer when Hindi is expected is caught."""
        tok = Tokenizer.train(["Hello world"], language="en", vocab_size=2000, print_budget=False)
        is_valid, reasons = validate_tokenizer_metadata(
            tok, self.config, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Language mismatch" in r for r in reasons))

        # Check raise_on_error
        with self.assertRaises(TokenizerValidationError):
            validate_tokenizer_metadata(tok, self.config, expected_lang="hi", raise_on_error=True)

    def test_vocab_size_mismatch_detected(self):
        """Verify vocabulary size mismatch (e.g. 16k vs 40k) is caught."""
        # Train a 2000 vocab tokenizer when 40000 is expected for Hindi
        tok = Tokenizer.train(["नमस्ते भारत"], language="hi", vocab_size=2000, print_budget=False)
        is_valid, reasons = validate_tokenizer_metadata(
            tok, self.config, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Configured vocabulary size mismatch" in r for r in reasons))

    def test_realized_vocabulary_may_be_below_configured_cap(self):
        """A tokenizer that learned fewer merges than its configured cap remains valid."""
        tok = Tokenizer.train(
            ["नमस्ते भारत"],
            language="hi",
            vocab_size=40000,
            print_budget=False,
        )
        self.assertLess(tok.get_vocab_size(), 40000)

        is_valid, reasons = validate_tokenizer_metadata(
            tok, self.config, expected_lang="hi"
        )
        self.assertTrue(is_valid, f"Realized vocabulary below cap should pass: {reasons}")

    def test_training_corpus_mismatch_strict(self):
        """Verify training corpus mismatch in strict mode."""
        tok = Tokenizer.train(
            ["नमस्ते भारत"],
            language="hi",
            vocab_size=40000,
            corpus_identity="custom_private_corpus",
            print_budget=False,
        )
        is_valid, reasons = validate_tokenizer_metadata(
            tok, self.config, expected_lang="hi", strict=True
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Training corpus mismatch" in r for r in reasons))

    def test_tokenizer_hash_tampering_detected(self):
        """Verify tampering with tokens or merges triggers hash integrity failure."""
        tok = Tokenizer.train(["नमस्ते भारत"], language="hi", vocab_size=2000, print_budget=False)
        tok_dict = TokenizerSerializer.to_dict(tok)

        # Alter hash in metadata
        tok_dict["metadata"]["tokenizer_hash"] = "0" * 64
        loaded_tampered = TokenizerSerializer.from_dict(tok_dict)

        is_valid, reasons = validate_tokenizer_metadata(
            loaded_tampered, self.config, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("hash integrity failure" in r for r in reasons))

    def test_pair_count_validation(self):
        """Verify negative or sub-threshold pair count is caught."""
        tok = Tokenizer.train(["नमस्ते"], language="hi", vocab_size=2000, print_budget=False)
        tok.training_pair_count = -10
        is_valid, reasons = validate_tokenizer_metadata(
            tok, self.config, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid negative training pair count" in r for r in reasons))

        # Sub-threshold check
        cfg_with_min = dict(self.config)
        cfg_with_min["min_tokenizer_pairs"] = 50000
        tok.training_pair_count = 100
        is_valid, reasons = validate_tokenizer_metadata(
            tok, cfg_with_min, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("below required experiment threshold" in r for r in reasons))

    def test_benchmark_leakage_detected(self):
        """Verify benchmark leakage (e.g. FLORES-200 / IN22 contamination) is caught."""
        leaked_corpora = [
            "flores200_devtest",
            "in22_conv",
            "in22_gen",
            "benchmark_eval_v1",
        ]
        for leaked_corpus in leaked_corpora:
            tok = Tokenizer.train(
                ["नमस्ते भारत"],
                language="hi",
                vocab_size=40000,
                corpus_identity=leaked_corpus,
                print_budget=False,
            )
            is_valid, reasons = validate_tokenizer_metadata(
                tok, self.config, expected_lang="hi"
            )
            self.assertFalse(is_valid)
            self.assertTrue(
                any("BENCHMARK LEAKAGE DETECTED" in r for r in reasons),
                f"Failed to catch benchmark leakage for {leaked_corpus}",
            )

        # Explicit metadata flag
        clean_tok = Tokenizer.train(["नमस्ते भारत"], language="hi", vocab_size=40000, print_budget=False)
        clean_tok.metadata["benchmark_leakage"] = True
        is_valid, reasons = validate_tokenizer_metadata(
            clean_tok, self.config, expected_lang="hi"
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("BENCHMARK LEAKAGE DETECTED" in r for r in reasons))

    def test_train_py_get_or_build_tokenizer_prevents_stale_reuse(self):
        """Verify train.py get_or_build_tokenizer blocks stale tokenizer reuse."""
        from train import get_or_build_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a stale English tokenizer at the Hindi target path
            stale_path = Path(tmpdir) / "stale_tokenizer.json"
            stale_tok = Tokenizer.train(["Hello English"], language="en", vocab_size=2000, print_budget=False)
            stale_tok.save(stale_path)

            test_config = dict(self.config)
            test_config["tokenizer_file"] = str(Path(tmpdir) / "stale_tokenizer.json")

            mock_ds = [{"translation": {"en": "Hello", "hi": "नमस्ते"}}]

            # Must raise TokenizerValidationError rather than silently reusing English tokenizer
            with self.assertRaises(TokenizerValidationError):
                get_or_build_tokenizer(test_config, mock_ds, "hi")

            # If retrain_invalid_tokenizer is enabled, it rebuilds a fresh one
            test_config["retrain_invalid_tokenizer"] = True
            rebuilt = get_or_build_tokenizer(test_config, mock_ds, "hi")
            self.assertIsNotNone(rebuilt)
            self.assertEqual(rebuilt.language, "hi")


if __name__ == "__main__":
    unittest.main()
