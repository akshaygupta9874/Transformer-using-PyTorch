import unittest
import unicodedata
from tokenizer import (
    AksharaSegmenter,
    DevanagariNormalizer,
    Tokenizer,
    WORD_MARK,
    ZWJ,
    ZWNJ,
    VIRAMA,
    NUKTA,
)


class SegmentationCorpusTests(unittest.TestCase):
    def setUp(self):
        self.normalizer = DevanagariNormalizer(devanagari_rules=True, unify_danda=True)
        # Train a representative tokenizer for end-to-end round-trip checks
        self.tokenizer = Tokenizer.train(
            [
                "नमस्ते विश्व किताब भारत में Python",
                "यह एक अत्यंत कठिन परीक्षा है।",
                "त्स्न्य र्त्स्र्य शास्त्रेषु",
            ],
            vocab_size=1000,
            min_frequency=1,
        )

    # ----------------------------------------------------------------------
    # 1. Three-or-more-consonant conjuncts
    # ----------------------------------------------------------------------
    def test_01_three_or_more_consonant_conjuncts(self):
        """Verify that multi-consonant clusters (3, 4, 5 consonants) form indivisible atomic units."""
        clusters = {
            # 3 consonants
            "स्त्र": ["स्त्र"],        # s + t + r
            "स्म्र": ["स्म्र"],        # s + m + r
            "त्स्म": ["त्स्म"],        # t + s + m
            "क्ष्म": ["क्ष्म"],        # k + sh + m
            "ष्ण्य": ["ष्ण्य"],        # sh + n + y
            "ञ्छ्र": ["ञ्छ्र"],        # ny + ch + r
            # 4 consonants
            "त्स्न्य": ["त्स्न्य"],    # t + s + n + y
            "त्त्र्य": ["त्त्र्य"],    # t + t + r + y
            "र्द्ध्व": ["र्द्ध्व"],    # r + d + dh + v
            "ङ्क्त्य": ["ङ्क्त्य"],    # ng + k + t + y
            "ष्ट्र्य": ["ष्ट्र्य"],    # sh + t + r + y
            # 5 consonants
            "र्त्स्र्य": ["र्त्स्र्य"],  # r + t + s + r + y
        }
        for cluster, expected in clusters.items():
            segmented = AksharaSegmenter.segment(cluster)
            self.assertEqual(segmented, expected, f"Failed on cluster {cluster}")

        # Inflected words with complex conjuncts
        self.assertEqual(
            AksharaSegmenter.segment("त्स्न्येभ्यः"),
            ["त्स्न्ये", "भ्यः"],
        )
        self.assertEqual(
            AksharaSegmenter.segment("शास्त्रेषु"),
            ["शा", "स्त्रे", "षु"],
        )
        self.assertEqual(
            AksharaSegmenter.segment("उत्स्फूर्त"),
            ["उ", "त्स्फू", "र्त"],
        )

    # ----------------------------------------------------------------------
    # 2. All nukta consonants
    # ----------------------------------------------------------------------
    def test_02_all_nukta_consonants(self):
        """Verify that all classical and extended nukta consonants segment atomically."""
        nukta_chars = ["क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़", "य़", "ऩ", "ऱ", "ल़", "ष़"]
        for ch in nukta_chars:
            segmented = AksharaSegmenter.segment(ch)
            self.assertEqual(segmented, [ch], f"Nukta consonant {ch} must be atomic")

            # Check decomposed input (base + nukta U+093C)
            normalized = self.normalizer.normalize(ch)
            self.assertEqual(AksharaSegmenter.segment(normalized), [normalized])

        # Words containing nuktas with matras and conjuncts
        nukta_words = {
            "फ़्रेंच": ["फ़्रें", "च"],
            "ज़्यादती": ["ज़्या", "द", "ती"],
            "क़िस्सा": ["क़ि", "स्सा"],
            "लड़का": ["ल", "ड़", "का"],
            "पढ़ाई": ["प", "ढ़ा", "ई"],
        }
        for word, expected in nukta_words.items():
            self.assertEqual(AksharaSegmenter.segment(word), expected)

    # ----------------------------------------------------------------------
    # 3. ZWJ / ZWNJ before, after, and in illegitimate positions
    # ----------------------------------------------------------------------
    def test_03_zwj_zwnj_legitimate_and_illegitimate_positions(self):
        """Verify legitimate joiner conjuncts and graceful handling of stray joiners."""
        # Legitimate eyelash-ra (र + virama + ZWJ + consonant)
        eyelash = "र्" + ZWJ + "क"
        self.assertEqual(AksharaSegmenter.segment(eyelash), [eyelash])

        # Legitimate half-form cluster (consonant + virama + ZWNJ + consonant)
        half_form = "क्" + ZWNJ + "ष"
        self.assertEqual(AksharaSegmenter.segment(half_form), [half_form])

        # Illegitimate stray joiner at start of word
        stray_start = ZWJ + "नमस्ते"
        norm_start = self.normalizer.normalize(stray_start)
        self.assertEqual(norm_start, "नमस्ते")
        self.assertEqual(AksharaSegmenter.segment(norm_start), ["न", "म", "स्ते"])

        # Illegitimate stray joiner after vowel/matra
        stray_mid = "कि" + ZWJ + "ताब"
        norm_mid = self.normalizer.normalize(stray_mid)
        self.assertEqual(norm_mid, "किताब")
        self.assertEqual(AksharaSegmenter.segment(norm_mid), ["कि", "ता", "ब"])

        # Illegitimate stray joiner at end of word
        stray_end = "भारत" + ZWNJ
        norm_end = self.normalizer.normalize(stray_end)
        self.assertEqual(norm_end, "भारत")
        self.assertEqual(AksharaSegmenter.segment(norm_end), ["भा", "र", "त"])

    # ----------------------------------------------------------------------
    # 4. Combining marks, chandrabindu, visarga, reph
    # ----------------------------------------------------------------------
    def test_04_combining_marks_chandrabindu_visarga_reph(self):
        """Verify that modifiers (chandrabindu, anusvara, visarga, reph) attach to preceding syllables."""
        # Chandrabindu (ँ)
        self.assertEqual(AksharaSegmenter.segment("हँसना"), ["हँ", "स", "ना"])
        self.assertEqual(AksharaSegmenter.segment("आँख"), ["आँ", "ख"])
        self.assertEqual(AksharaSegmenter.segment("हूँ"), ["हूँ"])
        self.assertEqual(AksharaSegmenter.segment("गाँव"), ["गाँ", "व"])

        # Anusvara (ं)
        self.assertEqual(AksharaSegmenter.segment("संसार"), ["सं", "सा", "र"])
        self.assertEqual(AksharaSegmenter.segment("गंगा"), ["गं", "गा"])
        self.assertEqual(AksharaSegmenter.segment("अंगूर"), ["अं", "गू", "र"])

        # Visarga (ः)
        self.assertEqual(AksharaSegmenter.segment("दुःख"), ["दुः", "ख"])
        self.assertEqual(AksharaSegmenter.segment("अतः"), ["अ", "तः"])
        self.assertEqual(AksharaSegmenter.segment("पुनः"), ["पु", "नः"])
        self.assertEqual(AksharaSegmenter.segment("प्रायः"), ["प्रा", "यः"])

        # Reph (र्)
        self.assertEqual(AksharaSegmenter.segment("धर्म"), ["ध", "र्म"])
        self.assertEqual(AksharaSegmenter.segment("कर्म"), ["क", "र्म"])
        self.assertEqual(AksharaSegmenter.segment("सर्वश्रेष्ठ"), ["स", "र्व", "श्रे", "ष्ठ"])

        # Stacked marks: Consonant + Nukta + Matra + Anusvara
        self.assertEqual(AksharaSegmenter.segment("ज़िंदगी"), ["ज़िं", "द", "गी"])

        # Vedic tone marks (U+0951 Svarita, U+0952 Anudatta)
        vedic = "क" + "\u0951" + "\u0952"
        self.assertEqual(AksharaSegmenter.segment(self.normalizer.normalize(vedic)), [self.normalizer.normalize(vedic)])

    # ----------------------------------------------------------------------
    # 5. Numbers in Devanagari and Latin scripts
    # ----------------------------------------------------------------------
    def test_05_numbers_devanagari_and_latin(self):
        """Verify character-level preservation of numbers in Devanagari and Latin."""
        # Devanagari numbers
        dev_num = "१२३४५"
        self.assertEqual(AksharaSegmenter.segment(dev_num), ["१", "२", "३", "४", "५"])
        self.assertEqual(AksharaSegmenter.segment("२०२६"), ["२", "०", "२", "६"])
        self.assertEqual(AksharaSegmenter.segment("३.१४"), ["३", ".", "१", "४"])

        # Latin numbers
        latin_num = "12345"
        self.assertEqual(AksharaSegmenter.segment(latin_num), ["1", "2", "3", "4", "5"])
        self.assertEqual(AksharaSegmenter.segment("2026"), ["2", "0", "2", "6"])
        self.assertEqual(AksharaSegmenter.segment("3.14"), ["3", ".", "1", "4"])

        # Numbers with currency and percent symbols
        self.assertEqual(AksharaSegmenter.segment("₹५००"), ["₹", "५", "०", "०"])
        self.assertEqual(AksharaSegmenter.segment("$100"), ["$", "1", "0", "0"])
        self.assertEqual(AksharaSegmenter.segment("10%"), ["1", "0", "%"])
        self.assertEqual(AksharaSegmenter.segment("१०%"), ["१", "०", "%"])

        # Boundary between digits and Devanagari words
        self.assertEqual(
            AksharaSegmenter.segment("100लोग"),
            ["1", "0", "0", "लो", "ग"],
        )

    # ----------------------------------------------------------------------
    # 6. URLs and email addresses
    # ----------------------------------------------------------------------
    def test_06_urls_and_email_addresses(self):
        """Verify URLs, email addresses, and entity placeholders."""
        # Copy-preservation placeholders
        self.assertEqual(AksharaSegmenter.segment("⟨URL_0⟩"), ["⟨URL_0⟩"])
        self.assertEqual(AksharaSegmenter.segment("⟨EMAIL_1⟩"), ["⟨EMAIL_1⟩"])
        self.assertEqual(AksharaSegmenter.segment("⟨ENTITY_TEST_99⟩"), ["⟨ENTITY_TEST_99⟩"])

        # URL segmenting deterministically without dropping characters
        url = "https://google.com/search?q=nmt#top"
        url_seg = AksharaSegmenter.segment(url)
        self.assertEqual("".join(url_seg), url)

        # Email segmenting deterministically
        email = "user.name+tag@example.co.in"
        email_seg = AksharaSegmenter.segment(email)
        self.assertEqual("".join(email_seg), email)

        # Sentence with embedded entity placeholders
        sentence = "अधिक जानकारी के लिए ⟨URL_0⟩ देखें या ⟨EMAIL_1⟩ पर संपर्क करें।"
        seg_sent = AksharaSegmenter.segment(sentence)
        self.assertIn("⟨URL_0⟩", seg_sent)
        self.assertIn("⟨EMAIL_1⟩", seg_sent)
        self.assertEqual("".join(seg_sent), sentence)

    # ----------------------------------------------------------------------
    # 7. Hinglish and mixed-script words
    # ----------------------------------------------------------------------
    def test_07_hinglish_and_mixed_script(self):
        """Verify compound script words and Hinglish code-switching."""
        # Compound script terms
        compounds = {
            "WiFiराउटर": ["WiFi", "राउटर"],
            "COVID19टीकाकरण": ["COVID", "19", "टीकाकरण"],
            "E-कॉमर्स": ["E", "-", "कॉमर्स"],
            "WhatsAppमैसेज": ["WhatsApp", "मैसेज"],
            "2Gस्पेक्ट्रम": ["2", "G", "स्पेक्ट्रम"],
        }
        for term, expected_split in compounds.items():
            self.assertEqual(self.normalizer.script_split(term), expected_split)

        # Hinglish sentence segmentation
        hinglish = "main kal Delhi ja raha hoon"
        self.assertEqual(
            AksharaSegmenter.segment(hinglish),
            list(hinglish),
        )

        # Code-mixed sentence
        code_mix = "मैं आज coding कर रहा हूँ।"
        expected_mix = ["मैं", " ", "आ", "ज", " ", "c", "o", "d", "i", "n", "g", " ", "क", "र", " ", "र", "हा", " ", "हूँ", "।"]
        self.assertEqual(AksharaSegmenter.segment(code_mix), expected_mix)

    # ----------------------------------------------------------------------
    # 8. Punctuation and whitespace boundaries
    # ----------------------------------------------------------------------
    def test_08_punctuation_and_whitespace_boundaries(self):
        """Verify Dandas, punctuation, and varied whitespace separators."""
        # Purna Viram (।) and Dirgha Viram (॥)
        text_danda = "यह पहला वाक्य है। यह दूसरा वाक्य है॥"
        seg_danda = AksharaSegmenter.segment(text_danda)
        self.assertIn("।", seg_danda)
        self.assertIn("॥", seg_danda)
        self.assertEqual("".join(seg_danda), text_danda)

        # Standard punctuation
        punct_text = 'उसने कहा, "क्या बात है?" (उत्तर: हाँ/नहीं!)'
        seg_punct = AksharaSegmenter.segment(punct_text)
        self.assertEqual("".join(seg_punct), punct_text)

        # Diverse whitespace: tab, newline, non-breaking space (U+00A0), zero-width space (U+200B)
        ws_text = "शब्द1\tशब्द2\nशब्द3\u00A0शब्द4\u200Bशब्द5"
        norm_ws = self.normalizer.normalize(ws_text)
        seg_ws = AksharaSegmenter.segment(norm_ws)
        # All original words must be preserved
        reconstructed = "".join(seg_ws)
        self.assertIn("शब्द1", reconstructed)
        self.assertIn("शब्द2", reconstructed)
        self.assertIn("शब्द3", reconstructed)

    # ----------------------------------------------------------------------
    # 9. Very long and malformed Unicode strings
    # ----------------------------------------------------------------------
    def test_09_very_long_and_malformed_unicode_strings(self):
        """Verify tokenizer stability, totality, and lossless round-trip on long and malformed inputs."""
        # 14,000 character stream
        long_text = "नमस्ते विश्व! यह एक बहुत बड़ा परीक्षण है। " * 500
        seg_long = AksharaSegmenter.segment(long_text)
        self.assertEqual("".join(seg_long), long_text)

        # Consecutive viramas
        consec_virama = "्््"
        self.assertEqual(AksharaSegmenter.segment(consec_virama), ["्", "्", "्"])

        # Consecutive matras
        consec_matra = "ााा"
        self.assertEqual(AksharaSegmenter.segment(consec_matra), ["ा", "ा", "ा"])

        # Orphaned matra at start of word
        orphaned = "िकताब"
        self.assertEqual(AksharaSegmenter.segment(orphaned), ["ि", "क", "ता", "ब"])

        # Emojis and multi-byte Unicode
        emoji_text = "नमस्ते दुनिया 🚀🔥🤖"
        seg_emoji = AksharaSegmenter.segment(emoji_text)
        self.assertIn("🚀", seg_emoji)
        self.assertIn("🔥", seg_emoji)
        self.assertIn("🤖", seg_emoji)
        self.assertEqual("".join(seg_emoji), emoji_text)

        # End-to-end Tokenizer Totality and Lossless Round-Trip Check
        test_samples = [
            "त्स्न्य र्त्स्र्य शास्त्रेषु",
            "फ़्रेंच ज़िंदगी लड़का",
            "र्‍क क्‌ष नमस्ते",
            "हँसना दुःख गंगा धर्म",
            "१२३४५ ₹५०० 10% 3.14",
            "https://google.com ⟨URL_0⟩",
            "COVID19टीकाकरण WiFiराउटर",
            "नमस्ते दुनिया 🚀🔥🤖",
        ]
        self.assertTrue(self.tokenizer.verify_totality())
        for sample in test_samples:
            enc = self.tokenizer.encode(sample)
            dec = self.tokenizer.decode(enc.ids)
            norm = self.tokenizer.normalize(sample)
            self.assertEqual(
                dec,
                norm,
                f"Round-trip mismatch for {sample}: {dec} != {norm}",
            )


if __name__ == "__main__":
    unittest.main()
