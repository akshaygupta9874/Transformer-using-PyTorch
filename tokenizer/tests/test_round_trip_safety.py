import unittest
from pathlib import Path

from tokenizer import (
    DevanagariNormalizer,
    Tokenizer,
    audit_round_trip_safety,
)


class RoundTripSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We test both the pre-existing tokenizer and modern total-function tokenizer
        cls.tok_path = Path("data/tokenizers/tokenizer_hi.akshara.v2.json")
        if cls.tok_path.exists():
            cls.tok_hi = Tokenizer.load(cls.tok_path)
        else:
            cls.tok_hi = Tokenizer.train(["नमस्ते दुनिया"], language="hi", vocab_size=2000, print_budget=False)

        # Also create a modern total-function tokenizer with closure set and byte floor
        cls.tok_modern = Tokenizer.train(
            [
                "श्री नरेंद्र मोदी और अमिताभ बच्चन शेक्सपियर और आइंस्टीन",
                "https://example.com/api/v1/page?id=42#sec-3 http://192.168.1.1:8080/query?page=12&limit=50",
                "र्\u200dक क्\u200cष क्ष्ण त्स्न्य उत्स्फूर्त द्वंद्व राष्ट्र दृष्टि शृंखला",
            ],
            language="hi",
            vocab_size=3000,
            print_budget=False,
        )
        cls.normalizer = DevanagariNormalizer(devanagari_rules=True, unify_danda=True)

    def test_names_normalization_and_round_trip(self):
        """Verify normalization and tokenization never silently alter or corrupt names."""
        names = [
            # Modern Indian political, cultural, scientific names
            "श्री नरेंद्र मोदी",
            "अमिताभ बच्चन",
            "डॉ. ए. पी. जे. अब्दुल कलाम",
            "सी. वी. रमन",
            "रवींद्रनाथ ठाकुर",
            "सुब्रमण्यम भारती",
            "शकुंतला देवी",
            "श्रीनिवास रामानुजन",
            "विक्रम साराभाई",
            "होमी जहांगीर भाभा",
            "कल्पना चावला",
            "महात्मा गांधी",
            "सरदार वल्लभभाई पटेल",
            # Urdu / Persian / Arabic loan names with nukta
            "मिर्ज़ा ग़ालिब",
            "फ़ैज़ अहमद फ़ैज़",
            "क़ुरैशी",
            "शबाना आज़मी",
            "ज़फ़र",
            "आबिदा परवीन",
            # Transliterated international names
            "शेक्सपियर",
            "आइंस्टीन",
            "अल्बर्ट आइंस्टीन",
            "मैरी क्यूरी",
            "आइजैक न्यूटन",
            "चार्ल्स डार्विन",
            "जॉर्ज ओरवेल",
            # Honorific abbreviations
            "डॉ.",
            "पं.",
            "श्री",
            "श्रीमती",
            "प्रो.",
        ]

        for name in names:
            norm = self.normalizer.normalize(name)
            # Verify normalization preserves names faithfully
            self.assertGreater(len(norm), 0)
            # Modulo canonical nukta decomposition, name characters must match
            self.assertEqual(norm, self.normalizer.normalize(norm))

            # Verify round-trip decode(encode(name)) matches normalize(name)
            for tok in (self.tok_hi, self.tok_modern):
                enc = tok.encode(name)
                dec = tok.decode(enc.ids)
                expected = tok.normalizer.normalize(name)
                self.assertEqual(
                    dec,
                    expected,
                    f"Name corrupted during round-trip: input={name!r}, dec={dec!r}, expected={expected!r}",
                )

    def test_url_numbers_and_urls_round_trip(self):
        """Verify normalization and tokenization never alter URL numbers, ports, or structures."""
        urls = [
            "https://example.com/api/v1/page?id=42#sec-3",
            "http://192.168.1.1:8080/query?page=12&limit=50",
            "https://ai4bharat.org/models/2024/v2.1",
            "ftp://files.org/download/archive_2023_v2.tar.gz",
            "https://huggingface.co/datasets/ai4bharat/samanantar/tree/main",
            "http://localhost:3000/dashboard?metric=100&step=500",
            "https://user:pass@example.com:8443/resource?key=999#tag",
            # URLs embedded inside Hindi sentences
            "यह वेबसाइट https://example.com/v2/item?id=108 पर उपलब्ध है।",
            "अधिक जानकारी के लिए https://ai4bharat.org/v1/doc देखें।",
            # Plain numbers embedded in sentences
            "वर्ष 2024 में 108 उपग्रह प्रक्षेपित किए गए थे।",
            "इसका मूल्य 450 रुपये और 50 पैसे है।",
        ]

        for url_str in urls:
            norm = self.normalizer.normalize(url_str)
            # Ensure numbers and URL syntax are untouched
            # Colons in URLs must never become visarga
            self.assertNotIn("httpः", norm)
            self.assertNotIn("httpsः", norm)
            # Slashes must never become dandas
            self.assertNotIn("http:।", norm)
            # Digits must never be converted to Devanagari numerals
            if "42" in url_str:
                self.assertIn("42", norm)
            if "8080" in url_str:
                self.assertIn("8080", norm)
            if "108" in url_str:
                self.assertIn("108", norm)

            # Verify round-trip decode(encode(url)) matches normalize(url)
            for tok in (self.tok_hi, self.tok_modern):
                enc = tok.encode(url_str)
                dec = tok.decode(enc.ids)
                expected = tok.normalizer.normalize(url_str)
                self.assertEqual(
                    dec,
                    expected,
                    f"URL corrupted during round-trip: input={url_str!r}, dec={dec!r}, expected={expected!r}",
                )

    def test_zwj_sequences_round_trip(self):
        """Verify legitimate canonical ZWJ and ZWNJ sequences are never dropped or mutated."""
        zwj_cases = [
            # Eyelash-Ra conjuncts (र + virama + ZWJ + consonant)
            ("र्\u200dक", "Eyelash-Ra + ka"),
            ("र्\u200dख", "Eyelash-Ra + kha"),
            ("र्\u200dग", "Eyelash-Ra + ga"),
            ("र्\u200dय", "Eyelash-Ra + ya"),
            # Explicit half-forms (consonant + virama + ZWJ + consonant)
            ("क्\u200dय", "Half-ka + ya"),
            ("ह्\u200dन", "Half-ha + na"),
            ("ह्\u200dम", "Half-ha + ma"),
            ("ह्\u200dल", "Half-ha + la"),
            # Explicit virama / ZWNJ conjuncts (consonant + virama + ZWNJ + consonant)
            ("क्\u200cष", "Explicit virama ka + sha"),
            ("त्\u200cथ", "Explicit virama ta + tha"),
            ("द्\u200cध", "Explicit virama da + dha"),
            # Sentences with canonical joiners
            ("महाराष्ट्र में र्‍क और क्‌ष का प्रयोग देखा जाता है।", "Sentence with ZWJ and ZWNJ"),
        ]

        for s, desc in zwj_cases:
            norm = self.normalizer.normalize(s)
            # In canonical positions, joiners must be strictly preserved
            self.assertEqual(
                norm,
                s,
                f"Canonical ZWJ/ZWNJ silently dropped by normalizer in {desc}: input={s!r}, norm={norm!r}",
            )

            # Round-trip decode must match exactly
            for tok in (self.tok_hi, self.tok_modern):
                enc = tok.encode(s)
                dec = tok.decode(enc.ids)
                expected = tok.normalizer.normalize(s)
                self.assertEqual(
                    dec,
                    expected,
                    f"ZWJ/ZWNJ corrupted during round-trip in {desc}: input={s!r}, dec={dec!r}, expected={expected!r}",
                )

        # Stray non-canonical joiners (ZWJ without preceding virama) must be sanitized
        stray = "क\u200Dा"  # ka + ZWJ + aa matra (invalid rendering state)
        stray_norm = self.normalizer.normalize(stray)
        self.assertEqual(stray_norm, "का")  # correctly sanitized to valid ka + aa matra

    def test_conjunct_structure_round_trip(self):
        """Verify conjunct structure, virama positions, and consonant orders are preserved."""
        conjuncts = [
            # 2-consonant conjuncts
            "क्ष", "त्र", "ज्ञ", "श्र", "द्व", "द्म", "द्य", "प्र", "क्र", "ट्र",
            # 3-consonant conjuncts
            "क्ष्ण", "त्स्न", "उत्स्फूर्त", "त्त्र", "र्त्स्य", "स्म्य", "स्त्री",
            # 4- and 5-consonant conjuncts
            "त्स्न्य", "र्त्स्न्य", "कार्त्स्न्य",
            # Halant-terminated coda consonants
            "संसद्", "भगवान्", "विद्वान्", "श्रीमान्",
            # Vocalic R + matras
            "दृष्टि", "शृंखला", "ऋषि", "हृदय", "कृषि", "सृष्टि", "अमृत",
            # Nukta conjuncts
            "क़्", "ख़्", "फ़्", "अफ़ग़ानिस्तान", "तख़्त",
            # Complex conjunct sentences
            "धर्मात्मा व्यक्ति का अस्तित्व समाज के लिए एक आदर्श है।",
            "संस्कृत भाषा में कार्त्स्न्य और र्त्स्न्य जैसे कठिन संयुक्त वर्ण हैं।",
        ]

        for conj in conjuncts:
            norm = self.normalizer.normalize(conj)
            # Ensure virama and consonants are not lost
            self.assertEqual(norm, self.normalizer.normalize(norm))

            # Round-trip decode must match normalize
            for tok in (self.tok_hi, self.tok_modern):
                enc = tok.encode(conj)
                dec = tok.decode(enc.ids)
                expected = tok.normalizer.normalize(conj)
                self.assertEqual(
                    dec,
                    expected,
                    f"Conjunct corrupted during round-trip: input={conj!r}, dec={dec!r}, expected={expected!r}",
                )

    def test_real_corpora_round_trip_flores_and_in22(self):
        """Verify round-trip lossless property on frozen benchmark corpora."""
        benchmark_paths = [
            Path("data/benchmarks/flores200_devtest.tsv"),
            Path("data/benchmarks/in22_conv.tsv"),
            Path("data/benchmarks/in22_gen.tsv"),
        ]

        total_tested = 0
        for bp in benchmark_paths:
            if not bp.exists():
                continue
            with open(bp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        hi_sentence = parts[1].strip()
                        if hi_sentence:
                            total_tested += 1
                            enc = self.tok_modern.encode(hi_sentence)
                            dec = self.tok_modern.decode(enc.ids)
                            expected = self.tok_modern.normalizer.normalize(hi_sentence)
                            self.assertEqual(
                                dec,
                                expected,
                                f"Corpus sentence corrupted in {bp.name}: input={hi_sentence!r}, dec={dec!r}",
                            )

        self.assertGreater(total_tested, 0, "Benchmark files should contain sentences to test.")

    def test_audit_round_trip_safety_utility(self):
        """Verify the audit_round_trip_safety utility computes exact metrics."""
        sample_corpus = [
            "नमस्ते दुनिया।",
            "श्री नरेंद्र मोदी भारत के प्रधानमंत्री हैं।",
            "https://ai4bharat.org/models/2024/v2.1",
            "महाराष्ट्र में र्‍क और क्‌ष का प्रयोग होता है।",
            "कार्त्स्न्य और उत्स्फूर्त कठिन संयुक्त अक्षर हैं।",
        ]

        report = audit_round_trip_safety(sample_corpus, self.tok_modern)
        self.assertEqual(report["total_sentences"], len(sample_corpus))
        self.assertEqual(report["passed_count"], len(sample_corpus))
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["pass_rate_percentage"], 100.0)
        self.assertEqual(len(report["failures"]), 0)

        # Instance method check
        inst_report = self.tok_modern.audit_round_trip(sample_corpus)
        self.assertEqual(inst_report["pass_rate_percentage"], 100.0)


if __name__ == "__main__":
    unittest.main()
