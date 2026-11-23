import tempfile
import unittest
from pathlib import Path
from tokenizer.normalizer import DevanagariNormalizer, unify_brahmic_script
from tokenizer.akshara import AksharaSegmenter


class NormalizerTests(unittest.TestCase):
    def test_nukta_decomposition(self):
        norm = DevanagariNormalizer(decompose_nukta=True)
        # क़ (\u0958) should decompose to क (\u0915) + ़ (\u093C)
        result = norm.normalize("क़िताब")
        self.assertEqual(result, "\u0915\u093c\u093f\u0924\u093e\u092c")

    def test_danda_normalization(self):
        norm = DevanagariNormalizer(normalize_danda=True)
        text = "यह एक वाक्य है| दूसरा वाक्य||"
        normalized = norm.normalize(text)
        self.assertIn("।", normalized)
        self.assertIn("॥", normalized)
        self.assertNotIn("|", normalized)

    def test_colon_to_visarga(self):
        norm_off = DevanagariNormalizer(normalize_colon_to_visarga=False)
        self.assertEqual(norm_off.normalize("अत:"), "अत:")
        norm_on = DevanagariNormalizer(normalize_colon_to_visarga=True)
        self.assertEqual(norm_on.normalize("अत:"), "अतः")
        self.assertGreater(norm_on.colon_visarga_count, 0)

    def test_brahmic_script_unification(self):
        # Bengali 'ক' (\u0995) offset from Bengali base (0x0980) is 0x15.
        # Devanagari base (0x0900) + 0x15 is 'क' (\u0915).
        bengali_text = "ক"
        devanagari_translit = unify_brahmic_script(bengali_text, target_script="hi")
        self.assertEqual(devanagari_translit, "क")

    def test_placeholder_atomic_segmentation(self):
        units = AksharaSegmenter.segment("यह ⟨NUM_0⟩ और ⟨URL_1⟩ है")
        self.assertIn("⟨NUM_0⟩", units)
        self.assertIn("⟨URL_1⟩", units)

    def test_joiner_policy_modes(self):
        eyelash = "र्\u200dक"
        half_form = "क्\u200cष"
        stray = "\u200dनमस्ते कि\u200dताब भारत\u200c"

        # 1. preserve_canonical: legitimate joiners after virama kept, stray dropped
        norm_canon = DevanagariNormalizer(joiner_policy="preserve_canonical")
        self.assertEqual(norm_canon.normalize(eyelash), eyelash)
        self.assertEqual(norm_canon.normalize(half_form), half_form)
        self.assertEqual(norm_canon.normalize(stray), "नमस्ते किताब भारत")

        # 2. remove_all: all ZWJ/ZWNJ stripped
        norm_rem = DevanagariNormalizer(joiner_policy="remove_all")
        self.assertEqual(norm_rem.normalize(eyelash), "र्क")
        self.assertEqual(norm_rem.normalize(half_form), "क्ष")
        self.assertEqual(norm_rem.normalize(stray), "नमस्ते किताब भारत")

        # 3. preserve_all: keep all joiners verbatim
        norm_all = DevanagariNormalizer(joiner_policy="preserve_all")
        self.assertEqual(norm_all.normalize(eyelash), eyelash)
        self.assertEqual(norm_all.normalize(half_form), half_form)
        self.assertIn("\u200d", norm_all.normalize(stray))

    def test_normalization_policy_metadata_and_roundtrip(self):
        import tempfile
        from pathlib import Path
        from tokenizer import Tokenizer

        norm = DevanagariNormalizer(
            joiner_policy="preserve_canonical",
            decompose_nukta=True,
            unify_danda=True,
            normalize_colon_to_visarga=False,
        )
        policy = norm.get_policy()
        self.assertEqual(policy["normalizer_version"], "3.1")
        self.assertEqual(policy["joiner_policy"], "preserve_canonical")
        self.assertTrue(policy["preserve_zwj"])
        self.assertTrue(policy["preserve_zwnj"])
        self.assertEqual(policy["nukta_decomposition"], "decompose_to_base_plus_nukta")
        self.assertEqual(policy["danda_handling"], "unify_pipe_to_danda")
        self.assertEqual(policy["colon_visarga_handling"], "preserve_ascii_colon")
        self.assertEqual(policy["nfc_policy"], "canonical_nfc")
        self.assertEqual(policy["pretokenization_policy"], "placeholder_masking_script_run_akshara")

        # Reconstructed from policy
        reconstructed = DevanagariNormalizer.from_policy(policy)
        self.assertEqual(reconstructed.get_policy(), policy)

        # Tokenizer serialization round-trip
        tok = Tokenizer.train(["किताब किताब", "भारत में Python"], vocab_size=1000, min_frequency=1)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tok_meta.json"
            tok.save(p)
            loaded = Tokenizer.load(p)
            self.assertEqual(loaded.normalizer.get_policy(), tok.normalizer.get_policy())

    def test_comprehensive_serializer_metadata_fields(self):
        from tokenizer import Tokenizer, TokenizerSerializer

        tok = Tokenizer.train(["किताब किताब", "भारत में Python"], vocab_size=1000, min_frequency=2)
        d = TokenizerSerializer.to_dict(tok)

        expected_meta_keys = [
            "tokenizer_version",
            "language",
            "vocabulary_size",
            "merge_count",
            "training_corpus_identity",
            "training_corpus_revision",
            "training_corpus_identity_and_revision",
            "training_pair_count",
            "tokenizer_hash",
            "minimum_frequency",
            "normalizer_configuration",
            "segmentation_implementation_version",
        ]
        for key in expected_meta_keys:
            self.assertIn(key, d, f"Missing root key: {key}")
            self.assertIn(key, d["metadata"], f"Missing metadata key: {key}")

        expected_space_aliases = [
            "tokenizer version",
            "vocabulary size",
            "merge count",
            "training corpus identity and revision",
            "training pair count",
            "tokenizer hash",
            "minimum frequency",
            "normalizer configuration",
            "segmentation implementation version",
        ]
        for alias in expected_space_aliases:
            self.assertIn(alias, d, f"Missing root space alias: {alias}")
            self.assertIn(alias, d["metadata"], f"Missing metadata space alias: {alias}")

        self.assertEqual(len(d["tokenizer_hash"]), 64)
        self.assertEqual(d["language"], "hi")
        self.assertEqual(d["minimum_frequency"], 2)
        self.assertEqual(d["training_pair_count"], 2)
        self.assertEqual(d["segmentation_implementation_version"], "2.0")

        # Test disk roundtrip
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test_tok_meta.json"
            TokenizerSerializer.save(tok, p)
            loaded = TokenizerSerializer.load(p)
            self.assertEqual(loaded.language, "hi")
            self.assertEqual(loaded.version, "2.0")
            self.assertEqual(loaded.training_pair_count, 2)
            self.assertEqual(loaded.training_corpus_identity, "samanantar")
            self.assertEqual(loaded.training_corpus_revision, "v0.2")
            self.assertEqual(loaded.min_frequency, 2)
            self.assertEqual(loaded.tokenizer_hash, d["tokenizer_hash"])
            self.assertEqual(loaded.segmentation_implementation_version, "2.0")


if __name__ == "__main__":
    unittest.main()
