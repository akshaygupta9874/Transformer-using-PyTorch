import tempfile
import unittest
from pathlib import Path

from tokenizer import AksharaSegmenter, BPETrainer, Tokenizer


class AksharaTests(unittest.TestCase):
    def test_basic_orthographic_units(self):
        self.assertEqual(AksharaSegmenter.segment("किताब"), ["कि", "ता", "ब"])
        self.assertEqual(AksharaSegmenter.segment("भारत"), ["भा", "र", "त"])
        self.assertEqual(AksharaSegmenter.segment("प्रकाश"), ["प्र", "का", "श"])

    def test_conjuncts_and_nukta(self):
        self.assertEqual(AksharaSegmenter.segment("क्ष त्र ज्ञ"), ["क्ष", " ", "त्र", " ", "ज्ञ"])
        self.assertEqual(AksharaSegmenter.segment("क्ष्ण त्स्न्य उत्स्फूर्त"), ["क्ष्ण", " ", "त्स्न्य", " ", "उ", "त्स्फू", "र्त"])
        self.assertEqual(AksharaSegmenter.segment("क़ ख़ ग़ ज़ ड़ ढ़ फ़")[0::2], ["क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़"])

    def test_joiners_are_preserved_in_logical_order(self):
        self.assertEqual(AksharaSegmenter.segment("र्\u200dक"), ["र्\u200dक"])
        self.assertEqual(AksharaSegmenter.segment("क्\u200cष"), ["क्\u200cष"])

    def test_atomic_entity_placeholders(self):
        self.assertEqual(AksharaSegmenter.segment("⟨NUM_0⟩"), ["⟨NUM_0⟩"])
        self.assertEqual(AksharaSegmenter.segment("⟨URL_1⟩"), ["⟨URL_1⟩"])
        self.assertEqual(AksharaSegmenter.segment("⟨ENTITY_TEST_99⟩"), ["⟨ENTITY_TEST_99⟩"])
        text = "यह ⟨NUM_0⟩ और ⟨URL_1⟩ है"
        self.assertEqual(
            AksharaSegmenter.segment(text),
            ["य", "ह", " ", "⟨NUM_0⟩", " ", "औ", "र", " ", "⟨URL_1⟩", " ", "है"]
        )

    def test_code_mixing_round_trip(self):
        text = "मैं आज coding कर रहा हूँ।"
        self.assertEqual(AksharaSegmenter.segment(text), ["मैं", " ", "आ", "ज", " ", "c", "o", "d", "i", "n", "g", " ", "क", "र", " ", "र", "हा", " ", "हूँ", "।"])


class BPETests(unittest.TestCase):
    def test_merges_are_deterministic(self):
        trainer = BPETrainer(vocab_size=8, min_frequency=2).train([["कि", "ता", "ब"], ["कि", "ता", "ब"]])
        self.assertEqual(trainer.apply(["कि", "ता", "ब"]), ["किताब"])
        self.assertEqual(trainer.vocab_statistics(40)["unique_base_units"], 3)


class TokenizerTests(unittest.TestCase):
    def test_special_ids_fallback_and_serialization(self):
        tokenizer = Tokenizer.train(["किताब किताब", "भारत में Python"], vocab_size=32, min_frequency=1)
        self.assertEqual([tokenizer.token_to_id(token) for token in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]], [0, 1, 2, 3])
        original_ids = tokenizer.encode("भारत में Python").ids
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hindi.json"
            tokenizer.save(path)
            loaded = Tokenizer.load(path)
            self.assertEqual(tokenizer.text_to_tokens("किताब"), loaded.text_to_tokens("किताब"))
            self.assertEqual(original_ids, loaded.encode("भारत में Python").ids)

    def test_vocab_is_frozen_for_unseen_text(self):
        tokenizer = Tokenizer.train(["किताब"], vocab_size=16, min_frequency=1)
        size = tokenizer.get_vocab_size()
        tokenizer.encode("अज्ञात")
        self.assertEqual(tokenizer.get_vocab_size(), size)

    def test_word_boundaries_round_trip_and_do_not_merge_across_words(self):
        tokenizer = Tokenizer.train(["घर घर"], vocab_size=24, min_frequency=1)
        tokens = tokenizer.text_to_tokens("घर घर")
        self.assertEqual(sum(token.startswith("▁") for token in tokens), 2)
        self.assertEqual(tokenizer.decode(tokenizer.encode("घर घर").ids), "घर घर")

    def test_decode_replaces_malformed_generated_byte_sequences(self):
        tokenizer = Tokenizer.train(["भारत"], vocab_size=40000, min_frequency=1)
        malformed_byte_id = tokenizer.byte_base + 0xDE

        self.assertEqual(tokenizer.decode([malformed_byte_id]), "\uFFFD")
        self.assertEqual(
            tokenizer.decode([malformed_byte_id, tokenizer.token_to_id("▁")]),
            "\uFFFD ",
        )

    def test_vocabulary_budget_is_never_exceeded(self):
        tokenizer = Tokenizer.train(["अ आ इ ई उ ऊ"], vocab_size=8, min_frequency=1)
        self.assertLessEqual(tokenizer.get_vocab_size(), 8)

    def test_unseen_akshara_is_not_split_into_combining_marks(self):
        tokenizer = Tokenizer.train(["क"], vocab_size=16, min_frequency=1)
        ids = tokenizer.encode("ज्ञ").ids
        self.assertIn(tokenizer.token_to_id("<UNK>"), ids)
        self.assertLessEqual(len(ids), 2)  # word marker plus one Akshara-level UNK

    def test_hierarchical_fallback_constituent_code_points(self):
        # When an akshara conjunct is not in vocab, it must fall back to constituent
        # base characters and combining marks before emitting <UNK>.
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "▁", "क", "ज", "्", "ञ"]
        tokenizer = Tokenizer(tokens=tokens, merges=[], n_base=len(tokens), script="devanagari", is_legacy=True)
        ids = tokenizer.tokens_to_ids(["ज्ञ"])
        self.assertEqual(ids, [tokenizer.token_to_id("ज"), tokenizer.token_to_id("्"), tokenizer.token_to_id("ञ")])
        self.assertNotIn(tokenizer.token_to_id("<UNK>"), ids)

    def test_hierarchical_fallback_partial_constituents(self):
        # When some constituent code points exist and one is missing, known characters are preserved
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "▁", "क", "ज", "्"]
        tokenizer = Tokenizer(tokens=tokens, merges=[], n_base=len(tokens), script="devanagari", is_legacy=True)
        ids = tokenizer.tokens_to_ids(["ज्ञ"])
        self.assertEqual(ids, [tokenizer.token_to_id("ज"), tokenizer.token_to_id("्"), tokenizer.token_to_id("<UNK>")])

    def test_hierarchical_fallback_akshara_with_matra_decomposition(self):
        # When an akshara with matra is not in vocab, it decomposes to base consonant + matra
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "▁", "क", "ि", "ता", "ब"]
        tokenizer = Tokenizer(tokens=tokens, merges=[], n_base=len(tokens), script="devanagari", is_legacy=True)
        ids = tokenizer.tokens_to_ids(["▁किताब"])
        expected = [
            tokenizer.token_to_id("▁"),
            tokenizer.token_to_id("क"),
            tokenizer.token_to_id("ि"),
            tokenizer.token_to_id("ता"),
            tokenizer.token_to_id("ब"),
        ]
        self.assertEqual(ids, expected)
        self.assertNotIn(tokenizer.token_to_id("<UNK>"), ids)

    def test_hierarchical_fallback_exact_token_precedence(self):
        # Level 0 exact learned token must take precedence over subparts
        tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "▁", "क", "ि", "ता", "ब", "▁किताब"]
        tokenizer = Tokenizer(tokens=tokens, merges=[], n_base=len(tokens), script="devanagari", is_legacy=True)
        ids = tokenizer.tokens_to_ids(["▁किताब"])
        self.assertEqual(ids, [tokenizer.token_to_id("▁किताब")])


if __name__ == "__main__":
    unittest.main()
