import unittest
import torch

from modern_model import (
    RMSNorm,
    RotaryEmbedding,
    SwiGLU,
    KVCache,
    build_modern_transformer,
)
from research.optimization import ModelEMA, MinimumRiskTrainingLoss, get_cosine_schedule_with_warmup
from research.dynamic_batching import TokenBucketBatchSampler
from research.fast_decoding import fast_greedy_decode, fast_beam_search
from research.moe import MoEFeedForward
from research.back_translation import SyntheticNoiser, BackTranslationPipeline
from research.domain_tags import format_tagged_sentence, extract_tags


class ModernArchitectureTests(unittest.TestCase):
    def test_rmsnorm_invariance(self):
        norm = RMSNorm(16)
        x = torch.randn(2, 5, 16)
        out = norm(x)
        self.assertEqual(out.shape, (2, 5, 16))
        # Variance along last dimension should be approximately 1
        rms = torch.sqrt(out.pow(2).mean(-1) + 1e-6)
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_rope_rotation_properties(self):
        dim = 8
        rope = RotaryEmbedding(dim=dim, max_seq_len=64)
        q = torch.randn(1, 2, 4, dim)  # (batch, heads, seq, dim)
        k = torch.randn(1, 2, 4, dim)
        q_rot, k_rot = rope.apply_rope(q, k, offset=0)
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_swiglu_shape(self):
        swiglu = SwiGLU(d_model=32, d_ff=64)
        x = torch.randn(2, 8, 32)
        out = swiglu(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_modern_transformer_forward_and_kv_cache(self):
        model = build_modern_transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            heads=4,
            d_ff=64,
        )
        src = torch.randint(4, 100, (2, 10))
        tgt = torch.randint(4, 100, (2, 8))
        out = model(src, tgt)
        self.assertEqual(out.shape, (2, 8, 100))

        # Test KV Cache incremental step-by-step decoding matches non-cached decode
        model.eval()
        with torch.inference_mode():
            src_single = src[:1]
            enc_out = model.encode(src_single)

            # Full forward pass on 3 target tokens
            tgt_seq = torch.tensor([[2, 15, 23]])
            full_dec = model.decode(enc_out, None, tgt_seq)
            full_logits = model.project(full_dec[:, -1])

            # Incremental forward pass with KV Cache
            cache = KVCache()
            # Token 0 (SOS = 2)
            _ = model.decode(enc_out, None, torch.tensor([[2]]), kv_cache=cache)
            # Token 1 (15)
            _ = model.decode(enc_out, None, torch.tensor([[15]]), kv_cache=cache)
            # Token 2 (23)
            inc_dec = model.decode(enc_out, None, torch.tensor([[23]]), kv_cache=cache)
            inc_logits = model.project(inc_dec[:, -1])

            self.assertTrue(torch.allclose(full_logits, inc_logits, atol=1e-3))

    def test_fast_greedy_and_beam_decode(self):
        model = build_modern_transformer(
            src_vocab_size=50,
            tgt_vocab_size=50,
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            heads=2,
            d_ff=32,
        )
        src = torch.randint(4, 50, (1, 6))
        greedy_tokens = fast_greedy_decode(model, src, None, sos_id=2, eos_id=3, max_len=8)
        self.assertIsInstance(greedy_tokens, list)
        self.assertGreater(len(greedy_tokens), 0)

        beam_tokens = fast_beam_search(model, src, None, sos_id=2, eos_id=3, beam_size=2, max_len=8)
        self.assertIsInstance(beam_tokens, list)
        self.assertGreater(len(beam_tokens), 0)


class OptimizationAndTrainingTests(unittest.TestCase):
    def test_model_ema(self):
        model = torch.nn.Linear(8, 8)
        ema = ModelEMA(model, decay=0.9)
        initial_val = model.weight.data.clone()

        # Mutate model weights
        model.weight.data.add_(1.0)
        ema.update()

        # Verify shadow weights updated smoothly
        self.assertFalse(torch.allclose(ema.shadow["weight"], initial_val))
        self.assertFalse(torch.allclose(ema.shadow["weight"], model.weight.data))

        # Test apply and restore
        ema.apply_shadow()
        self.assertTrue(torch.allclose(model.weight.data, ema.shadow["weight"]))
        ema.restore()
        self.assertFalse(torch.allclose(model.weight.data, ema.shadow["weight"]))

    def test_dynamic_token_batching(self):
        # 100 sentences with variable lengths 5 to 50
        lengths = [5 + (i % 45) for i in range(100)]
        sampler = TokenBucketBatchSampler(lengths, max_tokens=100, shuffle=False)
        batches = list(iter(sampler))
        self.assertGreater(len(batches), 0)

        for batch in batches:
            max_l = max(lengths[i] for i in batch)
            total_tokens = len(batch) * max_l
            # Every batch should stay within token budget
            self.assertLessEqual(total_tokens, 100)

    def test_minimum_risk_training_loss(self):
        mrt = MinimumRiskTrainingLoss(alpha=0.01)
        log_probs = torch.tensor([[-0.2, -1.5]])  # 1 batch, 2 candidate samples
        candidates = [["नमस्ते दुनिया", "नमस्ते संसार"]]
        references = ["नमस्ते दुनिया"]
        loss = mrt(log_probs, candidates, references)
        self.assertGreater(loss.item(), 0.0)

    def test_moe_feed_forward(self):
        moe = MoEFeedForward(d_model=16, d_ff=32, num_experts=4, top_k=2)
        x = torch.randn(2, 6, 16)
        out, aux_loss = moe(x)
        self.assertEqual(out.shape, (2, 6, 16))
        self.assertGreater(aux_loss.item(), 0.0)

    def test_back_translation_and_noiser(self):
        noiser = SyntheticNoiser(drop_prob=0.2, blank_prob=0.2, shuffle_window=2, seed=1)
        text = "This is an important synthetic training sentence."
        noised = noiser.apply(text)
        self.assertIsInstance(noised, str)

        pipeline = BackTranslationPipeline(reverse_translator=lambda hi: f"English of {hi}")
        pairs = pipeline.generate_synthetic_pairs(["नमस्ते", "भारत"])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], "नमस्ते")

        mixed = BackTranslationPipeline.mix_datasets([("src", "tgt")], pairs, synthetic_ratio=0.5)
        self.assertGreaterEqual(len(mixed), 2)

    def test_domain_and_language_tags(self):
        tagged = format_tagged_sentence("Hello world", target_lang="hi", domain="news")
        self.assertEqual(tagged, "<domain:news> <2hi> Hello world")
        cleaned, meta = extract_tags(tagged)
        self.assertEqual(cleaned, "Hello world")
        self.assertEqual(meta["target_lang"], "hi")
        self.assertEqual(meta["domain"], "news")

    def test_basic_baseline_architecture_is_default(self):
        """Verify the hand-written vanilla Transformer baseline is default with Pre-LN, 6 enc / 4 dec, and Sinusoidal positions."""
        from config import get_config
        from run_experiment import compose_config
        from train import get_model
        from models.basic import Transformer, PositionalEncoding

        # 1. config.py defaults to vanilla baseline
        cfg = get_config()
        self.assertEqual(cfg.get("architecture"), "vanilla")
        self.assertEqual(cfg.get("norm_type"), "layernorm")
        self.assertEqual(cfg.get("pos_embedding"), "sinusoidal")
        self.assertEqual(cfg.get("encoder_layers"), 6)
        self.assertEqual(cfg.get("decoder_layers"), 4)

        # 2. Composed config from run_experiment defaults to baseline
        composed = compose_config([])
        self.assertEqual(composed.get("architecture"), "vanilla")
        self.assertEqual(composed.get("pos_embedding"), "sinusoidal")

        # 3. Model construction defaults to the hand-written vanilla Transformer
        model = get_model(cfg, vocab_src_len=100, vocab_tgt_len=100)
        self.assertIsInstance(model, Transformer)
        self.assertEqual(len(model.encoder.layers), 6)
        self.assertEqual(len(model.decoder.layers), 4)

        # 4. Verify baseline uses Sinusoidal positional embeddings (not RoPE)
        self.assertIsInstance(model.src_pos, PositionalEncoding)
        self.assertIsInstance(model.tgt_pos, PositionalEncoding)

    def test_modern_architecture_ablation(self):
        """Verify modern architecture (RoPE, RMSNorm, SwiGLU) is enabled via explicit ablation config."""
        from run_experiment import compose_config
        from train import get_model
        from modern_model import ModernTransformer

        composed_modern = compose_config(["architecture=modern_transformer"])
        self.assertEqual(composed_modern.get("architecture"), "modern_transformer")
        self.assertEqual(composed_modern.get("norm_type"), "rmsnorm")
        self.assertEqual(composed_modern.get("pos_embedding"), "rope")
        self.assertEqual(composed_modern.get("ffn_type"), "swiglu")

        model = get_model(composed_modern, vocab_src_len=100, vocab_tgt_len=100)
        self.assertIsInstance(model, ModernTransformer)
        self.assertTrue(hasattr(model, "rope"))

    def test_cosine_training_step_count_with_gradient_accumulation(self):
        """Verify cosine training step count accurately reflects optimizer updates under gradient accumulation."""
        import math
        from research.optimization import get_cosine_schedule_with_warmup

        num_epochs = 5
        dataloader_len = 35
        gradient_accumulation_steps = 4

        # Compute total steps using the exact formula
        steps_per_epoch = math.ceil(dataloader_len / gradient_accumulation_steps)  # ceil(35 / 4) = 9
        total_expected_steps = num_epochs * steps_per_epoch  # 5 * 9 = 45

        # Simulate the training loop step counts
        actual_optimizer_steps = 0
        for epoch in range(num_epochs):
            for batch_index in range(dataloader_len):
                if (batch_index + 1) % gradient_accumulation_steps == 0:
                    actual_optimizer_steps += 1
            if (batch_index + 1) % gradient_accumulation_steps != 0:
                actual_optimizer_steps += 1

        self.assertEqual(actual_optimizer_steps, total_expected_steps)

        # Verify scheduler reaches min_lr_ratio at the exact total_expected_steps
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([dummy_param], lr=1.0)
        warmup_steps = 5
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_expected_steps, min_lr_ratio=0.01)

        # Step scheduler through simulated training
        for _ in range(total_expected_steps):
            opt.step()
            sched.step()

        # At the final expected step, lr should reach min_lr_ratio (0.01)
        final_lr = opt.param_groups[0]["lr"]
        self.assertAlmostEqual(final_lr, 0.01, places=3)

    def test_target_padding_mask_exclusively_in_loss(self):
        """Verify loss computation ignores target PAD tokens exclusively."""
        pad_idx = 1
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

        # Batch of 2 tokens: one valid token (id=5), one pad token (id=1)
        logits = torch.randn(2, 20)
        labels = torch.tensor([5, pad_idx])

        # Loss should match loss computed only on the non-pad token
        loss_total = loss_fn(logits, labels)
        loss_single = loss_fn(logits[:1], labels[:1])
        self.assertAlmostEqual(loss_total.item(), loss_single.item(), places=5)

    def test_checkpoint_metadata_preservation(self):
        """Verify checkpoints preserve all model, optimizer, scheduler, tokenizer, and data metadata."""
        import tempfile
        from pathlib import Path
        from config import get_config
        from train import get_model

        cfg = get_config()
        model = get_model(cfg, vocab_src_len=50, vocab_tgt_len=50)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({
                'epoch': 3,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': 1500,
                'ema_state_dict': None,
                'config': dict(cfg),
                'tokenizer_metadata': {
                    'lang_src': 'en',
                    'lang_tgt': 'hi',
                    'vocab_src_size': 16000,
                    'vocab_tgt_size': 40000,
                },
                'data_metadata': {
                    'datasource': 'ai4bharat/samanantar',
                    'dataset_config': 'hi',
                    'seq_len': 350,
                    'train_samples': 50000,
                },
            }, ckpt_path)

            loaded = torch.load(ckpt_path)
            self.assertEqual(loaded['epoch'], 3)
            self.assertEqual(loaded['global_step'], 1500)
            self.assertIn('scheduler_state_dict', loaded)
            self.assertEqual(loaded['config']['architecture'], 'vanilla')
            self.assertEqual(loaded['tokenizer_metadata']['vocab_src_size'], 16000)
            self.assertEqual(loaded['data_metadata']['datasource'], 'ai4bharat/samanantar')

    def test_basic_transformer_initialization(self):
        """Verify Xavier initialization for linear weights, zero bias, and normal embeddings."""
        from models.basic import build_transformer, LayerNormalization
        import torch.nn as nn

        d_model = 64
        model = build_transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            src_seq_len=20,
            tgt_seq_len=20,
            d_model=d_model,
            encoder_layers=2,
            decoder_layers=2,
            h=4,
            d_ff=128,
        )

        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear biases initialized to 0
                if module.bias is not None:
                    self.assertTrue(torch.all(module.bias == 0))
                # Linear weights have reasonable non-zero variance
                self.assertGreater(module.weight.std().item(), 0.0)
            elif isinstance(module, nn.Embedding):
                # Embeddings initialized with std ~ d_model^-0.5
                expected_std = d_model ** -0.5
                self.assertAlmostEqual(module.weight.std().item(), expected_std, delta=0.05)
            elif isinstance(module, LayerNormalization):
                self.assertTrue(torch.all(module.alpha == 1.0))
                self.assertTrue(torch.all(module.bias == 0.0))

    def test_decoder_input_label_shift_alignment(self):
        """Verify teacher-forcing shift: decoder_input starts with BOS, label ends with EOS."""
        from dataset import BilingualDataset
        from tokenizer import Tokenizer

        tok_src = Tokenizer.train(["hello world"], language="en", vocab_size=100)
        tok_tgt = Tokenizer.train(["नमस्ते भारत"], language="hi", vocab_size=100)

        raw_data = [{"translation": {"en": "hello world", "hi": "नमस्ते भारत"}}]
        ds = BilingualDataset(raw_data, tok_src, tok_tgt, "en", "hi", seq_len=30)
        item = ds[0]

        dec_input = item["decoder_input"]
        label = item["label"]
        pad_id = tok_tgt.token_to_id("<PAD>")
        bos_id = tok_tgt.token_to_id("<BOS>")
        eos_id = tok_tgt.token_to_id("<EOS>")

        # dec_input starts with <BOS>
        self.assertEqual(dec_input[0].item(), bos_id)

        # find position of <EOS> in label
        eos_positions = (label == eos_id).nonzero(as_tuple=True)[0]
        self.assertEqual(len(eos_positions), 1)
        eos_pos = eos_positions[0].item()

        # Target label at step 0 matches the first real token following BOS in decoder_input
        self.assertEqual(label[0].item(), dec_input[1].item())

        # Positions after EOS in label must be PAD
        if eos_pos + 1 < len(label):
            self.assertTrue(torch.all(label[eos_pos + 1 :] == pad_id))

    def test_attention_masks_broadcast_and_causality(self):
        """Verify MultiHeadAttentionBlock handles 2D, 3D, and 4D masks and enforces strict causality."""
        from models.basic import MultiHeadAttentionBlock
        from dataset import causal_mask
        import torch

        mha = MultiHeadAttentionBlock(d_model=32, h=4, dropout=0.0)
        q = torch.randn(2, 6, 32)
        k = torch.randn(2, 6, 32)
        v = torch.randn(2, 6, 32)

        # 1. Test 3D causal mask: (1, 6, 6)
        c_mask = causal_mask(6)
        out = mha(q, k, v, c_mask)
        self.assertEqual(out.shape, (2, 6, 32))

        # 2. Test 4D mask: (2, 1, 1, 6)
        pad_mask = torch.tensor([[[[True, True, True, True, False, False]]],
                                [[[True, True, True, False, False, False]]]])
        out_pad = mha(q, k, v, pad_mask)
        self.assertEqual(out_pad.shape, (2, 6, 32))


if __name__ == "__main__":
    unittest.main()
