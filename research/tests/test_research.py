import unittest
import torch

from research.byte_tokenizer import ByteTokenizer
from research.curriculum import CurriculumSampler, compute_akshara_complexity, compute_length_difficulty
from research.decoding import (
    greedy_speculative_accept,
    mbr_rerank,
    pure_python_chrf,
    length_penalty,
)
from research.deployment import distillation_loss
from research.evaluation import (
    comprehensive_error_taxonomy,
    paired_bootstrap,
    paired_bootstrap_corpus,
    paired_bootstrap_sentence,
    pure_python_bleu,
    score_corpus,
)
from research.knn_mt import KNNMTDatastore, KNNMTDecoder
from research.placeholders import substitute, substitute_parallel


class ResearchUtilitiesTests(unittest.TestCase):
    def test_byte_tokenizer_round_trip(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.decode(tokenizer.encode("नमस्ते hello").ids), "नमस्ते hello")

    def test_placeholders_preserve_urls_and_devanagari_numbers(self):
        text, mapping = substitute("Cost १२.५ at https://example.org/a")
        self.assertEqual(mapping.restore(text), "Cost १२.५ at https://example.org/a")
        source, target, mapping = substitute_parallel("Read https://x.test", "https://x.test पढ़ें")
        self.assertIn("⟨URL_0⟩", source)
        self.assertEqual(mapping.restore(target), "https://x.test पढ़ें")

    def test_placeholders_deva_vs_latin_numeral_alignment(self):
        source = "In 2024, the total cost was 123."
        target = "२०२४ में, कुल लागत १२३ थी।"
        s_sub, t_sub, mapping = substitute_parallel(source, target)
        self.assertIn("⟨NUM_0⟩", s_sub)
        self.assertIn("⟨NUM_1⟩", s_sub)
        self.assertIn("⟨NUM_0⟩", t_sub)
        self.assertIn("⟨NUM_1⟩", t_sub)
        # Verify default restore preserves native target Devanagari numerals
        self.assertEqual(mapping.restore(t_sub), target)
        # Verify explicit script conversions
        self.assertEqual(mapping.restore(t_sub, script="latin"), "2024 में, कुल लागत 123 थी।")
        self.assertEqual(mapping.restore(s_sub, script="devanagari"), "In २०२४, the total cost was १२३.")

    def test_placeholders_reordered_target_spans(self):
        # English: SVO (give 5 apples to 2 children) vs Hindi: SOV (२ बच्चों को ५ सेब)
        source = "He gave 5 apples to 2 children."
        target = "उसने २ बच्चों को ५ सेब दिए।"
        s_sub, t_sub, mapping = substitute_parallel(source, target)
        # In source: 5 is NUM_0, 2 is NUM_1
        self.assertIn("⟨NUM_0⟩ apples to ⟨NUM_1⟩ children", s_sub)
        # In target: 2 should be NUM_1, 5 should be NUM_0
        self.assertIn("⟨NUM_1⟩ बच्चों को ⟨NUM_0⟩ सेब", t_sub)
        self.assertEqual(mapping.restore(t_sub), target)

    def test_placeholders_repeated_values_do_not_collide(self):
        # Repeated identical values in source and target
        source = "Room 1 on floor 1 costs $100."
        target = "कमरा १ मंजिल १ पर १०० डॉलर का है।"
        s_sub, t_sub, mapping = substitute_parallel(source, target)
        self.assertIn("⟨NUM_0⟩", s_sub)
        self.assertIn("⟨NUM_1⟩", s_sub)
        self.assertIn("⟨NUM_2⟩", s_sub)
        self.assertIn("⟨NUM_0⟩", t_sub)
        self.assertIn("⟨NUM_1⟩", t_sub)
        self.assertIn("⟨NUM_2⟩", t_sub)
        self.assertEqual(mapping.restore(t_sub), target)

    def test_placeholders_currency_percentage_and_dates(self):
        # Currency and percentages
        source = "Invested $10,000 for a 50% return."
        target = "१०,००० डॉलर निवेश करके ५०% लाभ कमाया।"
        s_sub, t_sub, mapping = substitute_parallel(source, target)
        self.assertIn("⟨NUM_0⟩", s_sub)
        self.assertIn("⟨NUM_1⟩", s_sub)
        self.assertIn("⟨NUM_0⟩", t_sub)
        self.assertIn("⟨NUM_1⟩", t_sub)
        self.assertEqual(mapping.restore(t_sub), target)

        # Date format differences (24/11/2026 -> २४-११-२०२६)
        src_date = "The deadline is 24/11/2026."
        tgt_date = "अंतिम तिथि २४/११/२०२६ है।"
        s_date_sub, t_date_sub, date_map = substitute_parallel(src_date, tgt_date)
        self.assertEqual(date_map.restore(t_date_sub), tgt_date)

    def test_placeholders_unrelated_target_number_preserved(self):
        # Target has an extra number not found in source
        source = "He bought 3 pens."
        target = "उसने 5 रुपये में 3 पेन खरीदे।"
        s_sub, t_sub, mapping = substitute_parallel(source, target)
        self.assertIn("⟨NUM_0⟩", s_sub)
        self.assertIn("5", t_sub)  # 5 was not in source, so it remains untouched
        self.assertIn("⟨NUM_0⟩", t_sub)  # 3 was in source, aligned to NUM_0
        self.assertEqual(mapping.restore(t_sub), target)

    def test_mbr_and_speculative_greedy(self):
        self.assertEqual(mbr_rerank(["a", "aa", "aaa"], lambda a, b: -abs(len(a) - len(b))), "aa")
        accepted, count = greedy_speculative_accept([1, 2], lambda prefix: 1 if len(prefix) == 1 else 9, [0])
        self.assertEqual((accepted, count), ([0, 1, 9], 1))

    def test_pure_python_chrf(self):
        score_identical = pure_python_chrf("नमस्ते", "नमस्ते")
        self.assertAlmostEqual(score_identical, 1.0, places=2)
        score_diff = pure_python_chrf("नमस्ते", "किताब")
        self.assertLess(score_diff, 0.5)

    def test_bootstrap_detects_clear_improvement(self):
        result = paired_bootstrap([0.0] * 20, [1.0] * 20, samples=200, seed=1)
        self.assertGreater(result["mean_delta"], 0)
        self.assertGreater(result["ci_low"], 0)
        self.assertTrue(result["significant_p05"])

    def test_knn_datastore_and_decoder(self):
        d_model = 16
        vocab_size = 50
        datastore = KNNMTDatastore(d_model=d_model)
        
        # Add keys and values
        k1 = torch.randn(d_model)
        k2 = torch.randn(d_model)
        datastore.add(k1, val=10)
        datastore.add(k2, val=25)
        datastore.consolidate()
        
        self.assertEqual(len(datastore), 2)
        sims, neighbor_ids = datastore.search(k1, k=1)
        self.assertEqual(neighbor_ids[0, 0].item(), 10)

        # Test decoder interpolation
        decoder = KNNMTDecoder(datastore, k=2, temperature=1.0, lambda_weight=0.5)
        query = k1.unsqueeze(0)
        knn_dist = decoder.compute_knn_distribution(query, vocab_size)
        self.assertGreater(knn_dist[0, 10].item(), 0.0)

        model_logits = torch.zeros((1, vocab_size))
        interp = decoder.interpolate(model_logits, knn_dist)
        self.assertEqual(interp.shape, (1, vocab_size))

    def test_curriculum_sampler_and_metrics(self):
        len_diff = compute_length_difficulty(10, 15, max_seq_len=50)
        self.assertAlmostEqual(len_diff, 25 / 100)

        akshara_diff = compute_akshara_complexity("उत्स्फूर्त")
        self.assertGreater(akshara_diff, 0.0)

        # Test CurriculumSampler annealing
        scores = [0.1 * i for i in range(100)]
        sampler = CurriculumSampler(scores, warmup_fraction=0.5, total_steps=1000)
        sampler.set_step(0)
        early_samples = list(iter(sampler))
        self.assertEqual(len(early_samples), 100)

        # When warmed up, progress is 1.0
        sampler.set_step(600)
        self.assertEqual(sampler.get_progress(), 1.0)

    def test_distillation_loss(self):
        vocab_size = 10
        seq_len = 5
        student = torch.randn(1, seq_len, vocab_size)
        teacher = torch.randn(1, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (1, seq_len))
        loss = distillation_loss(student, teacher, labels, pad_id=0, alpha=0.5, temperature=2.0)
        self.assertGreater(loss.item(), 0.0)

    def test_comprehensive_error_taxonomy(self):
        hyps = ["यह एक वाक्य है।", "short"]
        refs = ["यह 123 वाक्य है।", "This is a very long sentence that was translated too short."]
        srcs = ["This is 123 sentence.", "Long english source."]
        tax = comprehensive_error_taxonomy(hyps, refs, sources=srcs)
        self.assertIn("number_omission_total", tax["counts"])
        self.assertIn("under_translation", tax["counts"])

    def test_pure_python_bleu_computation(self):
        # 1. Exact match produces 100.0
        h_exact = ["The quick brown fox jumps over the lazy dog."]
        r_exact = ["The quick brown fox jumps over the lazy dog."]
        self.assertAlmostEqual(pure_python_bleu(h_exact, r_exact), 100.0, places=2)

        # 2. Devanagari sentences with danda
        h_deva = ["नमस्ते भारत। यह एक परीक्षण है।"]
        r_deva = ["नमस्ते भारत। यह एक परीक्षण है।"]
        self.assertAlmostEqual(pure_python_bleu(h_deva, r_deva), 100.0, places=2)

        # 3. Partial overlap produces genuine intermediate score (not 0.0 or 100.0)
        h_part = ["The quick brown fox jumps"]
        r_part = ["The quick brown dog jumps over the lazy fox"]
        score_part = pure_python_bleu(h_part, r_part)
        self.assertGreater(score_part, 15.0)
        self.assertLess(score_part, 50.0)

        # 4. Total mismatch produces 0.0
        h_diff = ["apple orange banana"]
        r_diff = ["car plane train"]
        self.assertEqual(pure_python_bleu(h_diff, r_diff), 0.0)

        # 5. Short sentences (2 tokens) adapt effective order without crashing
        self.assertAlmostEqual(pure_python_bleu(["नमस्ते भारत"], ["नमस्ते भारत"]), 100.0, places=2)

    def test_score_corpus_fallback_bleu_not_silent_zero(self):
        # When SacreBLEU is unavailable, identical hypotheses must NOT silently report 0.0 BLEU
        hyps = ["नमस्ते भारत।", "यह एक अच्छा दिन है।"]
        refs = ["नमस्ते भारत।", "यह एक अच्छा दिन है।"]

        metrics = score_corpus(hyps, refs)
        self.assertIn("bleu", metrics)
        self.assertIn("chrf", metrics)
        self.assertAlmostEqual(metrics["bleu"], 100.0, places=2)
        self.assertAlmostEqual(metrics["chrf"], 100.0, places=2)
        self.assertEqual(metrics["bleu_engine"], "pure_python")
        self.assertFalse(metrics["sacrebleu_available"])

    def test_score_corpus_omit_unavailable_bleu(self):
        # When omit_unavailable_bleu=True, BLEU is marked None/unavailable
        hyps = ["नमस्ते भारत।"]
        refs = ["नमस्ते भारत।"]

        metrics = score_corpus(hyps, refs, omit_unavailable_bleu=True)
        self.assertIsNone(metrics["bleu"])
        self.assertEqual(metrics["bleu_engine"], "unavailable")
        self.assertFalse(metrics["sacrebleu_available"])

    def test_paired_bootstrap_corpus_bleu_fallback(self):
        # Verify paired bootstrap on BLEU metric uses fallback rather than returning chrF deltas
        base = ["The cat on mat.", "Good morning."]
        cand = ["The cat sat on the mat.", "Good morning world."]
        refs = ["The cat sat on the mat.", "Good morning world."]

        result = paired_bootstrap_corpus(base, cand, refs, metric_name="bleu", samples=100, seed=42)
        self.assertEqual(result["metric"], "bleu")
        self.assertGreater(result["mean_delta"], 0.0)
        self.assertIn("ci_low", result)
        self.assertIn("ci_high", result)

    def test_evaluate_copy_preservation(self):
        from research.evaluation import evaluate_copy_preservation

        sources = [
            "Visit https://example.org and see 123 items.",
            "Meeting on 24/11/2026 at room 5.",
            "Plain text without numbers.",
        ]
        # Hypothesis 1: Exact URL + Devanagari numerals (123 -> १२३)
        # Hypothesis 2: Preserves date numbers (24, 11, 2026) but misses room 5
        # Hypothesis 3: Perfect (no numbers/urls to preserve)
        hypotheses = [
            "https://example.org पर जाएँ और १२३ आइटम देखें।",
            "बैठक 24/11/2026 को होगी।",
            "साधारण पाठ।",
        ]

        metrics = evaluate_copy_preservation(sources, hypotheses)
        self.assertEqual(metrics["url_copy_rate"], 1.0)
        self.assertEqual(metrics["number_copy_rate"], 0.8)  # 4 out of 5 numbers preserved
        self.assertEqual(metrics["total_urls"], 1)
        self.assertEqual(metrics["preserved_urls"], 1)
        self.assertEqual(metrics["total_numbers"], 5)
        self.assertEqual(metrics["preserved_numbers"], 4)

    def test_token_frequency_loss_analyzer(self):
        from research.optimization import TokenFrequencyLossAnalyzer
        from tokenizer import Tokenizer

        tok = Tokenizer.train(["एक दो तीन चार पाँच छह सात आठ नौ दस"], language="hi", vocab_size=50)
        v_size = tok.get_vocab_size()
        analyzer = TokenFrequencyLossAnalyzer(tok)

        logits = torch.randn(2, 4, v_size)
        # Ensure labels are within valid range [0, v_size - 1]
        labels = torch.tensor([[min(2, v_size - 1), min(3, v_size - 1), 0, 0],
                               [min(1, v_size - 1), min(v_size - 1, 10), min(v_size - 2, 8), 0]])

        report = analyzer.analyze(logits, labels)
        self.assertIn("head", report)
        self.assertIn("rare", report)
        self.assertIn("overall", report)
        self.assertGreater(report["overall"]["count"], 0)
        self.assertGreater(report["overall"]["loss"], 0.0)

    def test_target_aware_loss_and_label_smoothing_comparison(self):
        from research.optimization import TargetAwareLoss, compare_label_smoothing

        vocab_size = 20
        pad_id = 0
        loss_fn = TargetAwareLoss(vocab_size=vocab_size, pad_id=pad_id, label_smoothing=0.1)

        logits = torch.randn(4, vocab_size)
        targets = torch.tensor([1, 2, 5, pad_id])

        loss_val = loss_fn(logits, targets)
        self.assertGreater(loss_val.item(), 0.0)

        # Entity weighted loss
        loss_entity = TargetAwareLoss(
            vocab_size=vocab_size,
            pad_id=pad_id,
            label_smoothing=0.1,
            entity_token_ids=[5],
            entity_weight_multiplier=2.0,
        )
        loss_entity_val = loss_entity(logits, targets)
        self.assertGreater(loss_entity_val.item(), 0.0)

        # Label smoothing comparison (0.0 vs 0.05 vs 0.1)
        comp = compare_label_smoothing(logits, targets, smoothings=(0.0, 0.05, 0.1), pad_id=pad_id)
        self.assertIn(0.0, comp)
        self.assertIn(0.05, comp)
        self.assertIn(0.1, comp)
        self.assertEqual(comp[0.0]["target_confidence"], 1.0)
        self.assertEqual(comp[0.05]["target_confidence"], 0.95)
        self.assertEqual(comp[0.1]["target_confidence"], 0.90)


if __name__ == "__main__":
    unittest.main()
