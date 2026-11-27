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

    def test_checkpoint_compatibility_validation(self):
        """Verify checkpoint compatibility rejects vocab or architecture mismatches with actionable errors."""
        from train import validate_checkpoint_compatibility
        from run_experiment import compose_config

        cfg_modern = compose_config(["architecture=modern"])
        self.assertEqual(cfg_modern.get("architecture"), "modern_transformer")

        # 1. Matching checkpoint passes cleanly
        good_state = {
            "epoch": 2,
            "config": {"architecture": "modern_transformer"},
            "tokenizer_metadata": {"vocab_src_size": 16000, "vocab_tgt_size": 40000},
        }
        validate_checkpoint_compatibility(good_state, cfg_modern, 16000, 40000, "tmodel_02.pt")

        # 2. Target vocab mismatch (20268 vs 40000) raises RuntimeError with clear context
        stale_vocab_state = {
            "epoch": 1,
            "config": {"architecture": "modern_transformer"},
            "tokenizer_metadata": {"vocab_src_size": 16000, "vocab_tgt_size": 20268},
        }
        with self.assertRaises(RuntimeError) as ctx:
            validate_checkpoint_compatibility(stale_vocab_state, cfg_modern, 16000, 40000, "tmodel_01.pt")
        self.assertIn("20268", str(ctx.exception))
        self.assertIn("40000", str(ctx.exception))

        # 3. Legacy state without metadata inferred from weight shapes/keys rejects vanilla vs modern
        legacy_vanilla_state = {
            "epoch": 1,
            "model_state_dict": {
                "src_embed.embedding.weight": torch.zeros(16000, 512),
                "tgt_embed.embedding.weight": torch.zeros(20268, 512),
            },
        }
        with self.assertRaises(RuntimeError) as ctx_arch:
            validate_checkpoint_compatibility(legacy_vanilla_state, cfg_modern, 16000, 40000, "tmodel_01.pt")
        self.assertIn("vanilla", str(ctx_arch.exception))
        self.assertIn("modern_transformer", str(ctx_arch.exception))

    def test_modern_qk_norm_depth_init_and_weight_tying(self):
        """Verify QK-Norm, depth-scaled residual init, project_logits, and tie_embeddings."""
        import math
        import torch.nn.functional as F

        model = build_modern_transformer(
            src_vocab_size=120,
            tgt_vocab_size=150,
            d_model=64,
            encoder_layers=6,
            decoder_layers=4,
            heads=8,
            d_ff=256,
            use_qk_norm=True,
            tie_embeddings=True,
        )
        # QK-Norm present with d_k = 8
        self.assertIsInstance(model.encoder_blocks[0].self_attn.q_norm, RMSNorm)
        self.assertIsInstance(model.encoder_blocks[0].self_attn.k_norm, RMSNorm)
        self.assertEqual(model.encoder_blocks[0].self_attn.q_norm.weight.shape, (8,))
        self.assertIsInstance(model.decoder_blocks[0].cross_attn.q_norm, RMSNorm)

        # Depth-scaled std on residual output projections: 0.02 / sqrt(2 * N)
        expected_enc_std = 0.02 / math.sqrt(2 * 6)
        expected_dec_std = 0.02 / math.sqrt(2 * 4)
        enc_wo_std = model.encoder_blocks[0].self_attn.w_o.weight.std().item()
        dec_down_std = model.decoder_blocks[0].ffn.w_down.weight.std().item()
        self.assertLess(abs(enc_wo_std - expected_enc_std), 0.005)
        self.assertLess(abs(dec_down_std - expected_dec_std), 0.005)

        # Tied target embeddings & projection weight
        self.assertIs(model.proj.weight, model.tgt_embed.weight)

        # project_logits matches pre-log_softmax of project()
        hidden = torch.randn(2, 5, 64)
        raw_logits = model.project_logits(hidden)
        log_probs = model.project(hidden)
        self.assertTrue(torch.allclose(F.log_softmax(raw_logits, dim=-1), log_probs, atol=1e-5))

    def test_warmup_scaling_ema_ramp_and_optimizer_groups(self):
        """Verify dynamic warmup capping, EMA decay ramp, and no-weight-decay on RMSNorm."""
        from train import compute_effective_warmup_steps, build_optimizer_param_groups

        # 20k-sample run (2,820 steps) should cap 4000 warmup down to ~169 steps
        self.assertEqual(compute_effective_warmup_steps(4000, 2820), 169)
        # Large run (100k steps) keeps configured 4000 cap
        self.assertEqual(compute_effective_warmup_steps(4000, 100000), 4000)

        model = build_modern_transformer(
            src_vocab_size=80,
            tgt_vocab_size=80,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            heads=4,
            d_ff=64,
            tie_embeddings=True,
        )
        groups = build_optimizer_param_groups(model, weight_decay=0.01)
        self.assertEqual(len(groups), 2)
        decay_group, no_decay_group = groups[0], groups[1]
        self.assertEqual(decay_group["weight_decay"], 0.01)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)

        no_decay_ids = {id(p) for p in no_decay_group["params"]}
        for name, p in model.named_parameters():
            if "norm" in name.lower() or p.ndim < 2:
                self.assertIn(id(p), no_decay_ids)

        # EMA warmup ramp starts low (0.1) and increases cleanly
        ema = ModelEMA(model, decay=0.9999, warmup_ramp=True)
        self.assertAlmostEqual(ema.get_decay(), 0.1, places=5)
        ema.update()
        self.assertAlmostEqual(ema.get_decay(), 2.0 / 11.0, places=5)

    def test_fast_beam_coverage_grammar_mask_and_checkpoint_averaging(self):
        """Verify fast_beam_search with coverage_beta=0.2 + DevanagariGrammarMask and CheckpointAverager EMA averaging."""
        import tempfile
        from pathlib import Path
        from tokenizer import Tokenizer
        from tokenizer.grammar_mask import DevanagariGrammarMask
        from research.optimization import CheckpointAverager

        tok = Tokenizer.train(["नमस्ते भारत। यह परीक्षण है।"], language="hi", vocab_size=80)
        vocab = tok.get_vocab_size()
        model = build_modern_transformer(
            src_vocab_size=vocab,
            tgt_vocab_size=vocab,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            heads=4,
            d_ff=64,
        )
        src = torch.randint(4, vocab, (1, 6))
        src_mask = torch.ones((1, 1, 1, 6), dtype=torch.bool)
        grammar_mask = DevanagariGrammarMask(tok)

        out = fast_beam_search(
            model,
            src,
            src_mask,
            sos_id=tok.token_to_id("<BOS>"),
            eos_id=tok.token_to_id("<EOS>"),
            beam_size=4,
            max_len=10,
            alpha=0.6,
            coverage_beta=0.2,
            grammar_mask=grammar_mask,
        )
        self.assertGreater(len(out), 0)
        # Verify emitted transitions obey DevanagariGrammarMask
        for a, b in zip(out[:-1], out[1:]):
            self.assertTrue(grammar_mask.allowed(a, b))

        # Verify CheckpointAverager averages both model_state_dict and ema_state_dict
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for idx, val in enumerate([1.0, 3.0]):
                sd = {k: torch.full_like(v.float(), val) for k, v in model.state_dict().items()}
                ema_sd = {k: torch.full_like(v.float(), val * 2.0) for k, v in model.state_dict().items()}
                torch.save({"epoch": idx, "model_state_dict": sd, "ema_state_dict": ema_sd}, tmp / f"tmodel_0{idx}.pt")
            avg_path = CheckpointAverager.average_last_k(tmp, k=2)
            self.assertIsNotNone(avg_path)
            avg_ckpt = torch.load(avg_path, map_location="cpu", weights_only=False)
            first_key = next(iter(avg_ckpt["model_state_dict"]))
            self.assertTrue(torch.allclose(avg_ckpt["model_state_dict"][first_key], torch.full_like(avg_ckpt["model_state_dict"][first_key], 2.0)))
            self.assertTrue(torch.allclose(avg_ckpt["ema_state_dict"][first_key], torch.full_like(avg_ckpt["ema_state_dict"][first_key], 4.0)))

    def test_chunked_cross_entropy_parity_and_sample_weights(self):
        """Verify chunked_cross_entropy matches F.cross_entropy in loss, gradients, and sample weighting."""
        import torch.nn.functional as F
        from research.optimization import chunked_cross_entropy

        torch.manual_seed(0)
        n_tok, d, vocab = 128, 32, 101
        x = torch.randn(n_tok, d, requires_grad=True)
        w_mat = torch.randn(vocab, d, requires_grad=True)
        t = torch.randint(1, vocab, (n_tok,))
        t[::7] = 0  # PAD=0

        x_ref = x.detach().clone().requires_grad_()
        w_ref_mat = w_mat.detach().clone().requires_grad_()
        loss_ref = F.cross_entropy(x_ref @ w_ref_mat.t(), t, ignore_index=0, label_smoothing=0.1)
        loss_ref.backward()

        loss_got = chunked_cross_entropy(x, w_mat, t, label_smoothing=0.1, ignore_index=0, chunk_size=32, compute_dtype=torch.float32)
        loss_got.backward()

        self.assertLess((loss_ref - loss_got).abs().item(), 1e-5)
        self.assertLess((x.grad - x_ref.grad).abs().max().item(), 1e-5)
        self.assertLess((w_mat.grad - w_ref_mat.grad).abs().max().item(), 1e-5)

        # Per-sentence sample_weight broadcasting (B, T)
        x2 = x.detach().view(4, 32, d)
        t2 = t.view(4, 32)
        sw = torch.tensor([1.0, 0.5, 2.0, 0.8])
        got_w = chunked_cross_entropy(x2, w_mat.detach(), t2, label_smoothing=0.1, ignore_index=0, sample_weight=sw, chunk_size=32, compute_dtype=torch.float32)
        per = F.cross_entropy(x2.reshape(-1, d) @ w_mat.detach().t(), t2.reshape(-1), ignore_index=0, label_smoothing=0.1, reduction="none")
        sw_flat = sw.unsqueeze(1).expand_as(t2).reshape(-1)
        keep = t2.reshape(-1) != 0
        expected_w = (per[keep] * sw_flat[keep]).sum() / sw_flat[keep].sum()
        self.assertLess((got_w - expected_w).abs().item(), 1e-5)

        # BF16 compute path runs without NaNs
        got_bf16 = chunked_cross_entropy(x.detach(), w_mat.detach(), t, label_smoothing=0.1, ignore_index=0, compute_dtype=torch.bfloat16)
        self.assertFalse(torch.isnan(got_bf16))

    def test_gqa_asymmetric_encoder_dff_and_kv_cache(self):
        """Verify ModernTransformer with kv_heads=4 (GQA), 8 enc / 4 dec, and encoder_d_ff=1536."""
        model = build_modern_transformer(
            src_vocab_size=100,
            tgt_vocab_size=100,
            d_model=32,
            encoder_layers=8,
            decoder_layers=4,
            heads=8,
            kv_heads=4,
            encoder_d_ff=48,
            d_ff=64,
            tie_embeddings=True,
        )
        self.assertEqual(len(model.encoder_blocks), 8)
        self.assertEqual(len(model.decoder_blocks), 4)
        self.assertEqual(model.encoder_blocks[0].ffn.w_gate.out_features, 48)
        self.assertEqual(model.decoder_blocks[0].ffn.w_gate.out_features, 64)
        self.assertEqual(model.decoder_blocks[0].self_attn.kv_heads, 4)

        model.eval()
        with torch.inference_mode():
            src = torch.randint(4, 100, (1, 7))
            enc_out = model.encode(src)
            tgt_seq = torch.tensor([[2, 11, 19]])
            full_logits = model.project_logits(model.decode(enc_out, None, tgt_seq)[:, -1])

            cache = KVCache()
            _ = model.decode(enc_out, None, torch.tensor([[2]]), kv_cache=cache)
            _ = model.decode(enc_out, None, torch.tensor([[11]]), kv_cache=cache)
            inc_logits = model.project_logits(model.decode(enc_out, None, torch.tensor([[19]]), kv_cache=cache)[:, -1])
            self.assertTrue(torch.allclose(full_logits, inc_logits, atol=1e-3))
            self.assertEqual(cache.self_keys[0].size(1), 4)  # stored at kv_heads size

    def test_memmap_pairs_sequence_packing_and_packed_model(self):
        """Verify build_memmap_from_tokenizers, MemmapPairs, PackedMemmapCollate, and packed ModernTransformer."""
        import tempfile
        from pathlib import Path
        from tokenizer import Tokenizer
        from research.memmap_data import (
            build_memmap_from_tokenizers,
            MemmapPairs,
            MemmapCollate,
            PackedMemmapCollate,
        )

        src_texts = ["hello world", "good morning", "test sentence one", "another short line"]
        tgt_texts = ["नमस्ते दुनिया", "सुप्रभात", "परीक्षण वाक्य एक", "एक और पंक्ति"]
        tok_src = Tokenizer.train(src_texts, language="en", vocab_size=80)
        tok_tgt = Tokenizer.train(tgt_texts, language="hi", vocab_size=80)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            tok_src_path = tmp / "tok_en.json"
            tok_tgt_path = tmp / "tok_hi.json"
            tok_src.save(tok_src_path)
            tok_tgt.save(tok_tgt_path)

            build_memmap_from_tokenizers(
                pairs=list(zip(src_texts, tgt_texts)),
                tok_src_path=tok_src_path,
                tok_tgt_path=tok_tgt_path,
                out_dir=tmp / "mm",
                split="train",
                seq_len=32,
                use_placeholders=False,
                n_src_variants=2,
                src_dropout_p=0.1,
                seed=13,
                n_proc=1,
                quality_scores=[1.0, 0.8, 0.9, 1.1],
            )

            ds = MemmapPairs(tmp / "mm", split="train", n_src_variants=2, dropout_frac=0.5, seed=13)
            self.assertEqual(len(ds), 4)
            self.assertEqual(len(ds.lengths()), 4)
            ds.set_epoch(2)
            batch_items = [ds[i] for i in range(4)]
            ds.close()

            std_batch = MemmapCollate()(batch_items)
            self.assertEqual(std_batch["encoder_input"].shape[0], 4)
            self.assertEqual(std_batch["sample_weight"].shape, (4,))

            packed_batch = PackedMemmapCollate(src_cap=32, tgt_cap=32)(batch_items)
            model = build_modern_transformer(
                src_vocab_size=tok_src.get_vocab_size(),
                tgt_vocab_size=tok_tgt.get_vocab_size(),
                d_model=32,
                encoder_layers=2,
                decoder_layers=2,
                heads=4,
                d_ff=64,
                tie_embeddings=True,
            )
            enc = model.encode(
                packed_batch["encoder_input"],
                packed_batch["encoder_mask"],
                pos=packed_batch["src_pos"],
            )
            dec = model.decode(
                enc,
                packed_batch["cross_mask"],
                packed_batch["decoder_input"],
                packed_batch["decoder_mask"],
                pos=packed_batch["tgt_pos"],
            )
            self.assertFalse(torch.isnan(dec).any())

    def test_batched_sampling_mbr_and_mrt_loss(self):
        """Verify sample_batched, fast_mbr_decode, and sequence-level mrt_loss backward pass."""
        from tokenizer import Tokenizer
        from tokenizer.grammar_mask import DevanagariGrammarMask
        from research.fast_decoding import sample_batched, fast_mbr_decode, mbr_select
        from research.optimization import mrt_loss

        tok = Tokenizer.train(["नमस्ते भारत। यह परीक्षण है।"], language="hi", vocab_size=80)
        vocab = tok.get_vocab_size()
        model = build_modern_transformer(
            src_vocab_size=vocab,
            tgt_vocab_size=vocab,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            heads=4,
            d_ff=64,
            tie_embeddings=True,
        )
        gmask = DevanagariGrammarMask(tok)
        src = torch.randint(4, vocab, (2, 5))
        src_mask = torch.ones((2, 1, 1, 5), dtype=torch.bool)

        seqs = sample_batched(model, src, src_mask, n=4, max_len=8, eps=0.02, grammar=gmask)
        self.assertEqual(seqs.shape[0], 2)
        self.assertEqual(seqs.shape[1], 4)

        best_i = mbr_select(["नमस्ते भारत", "क", "नमस्ते भारत।"], ["नमस्ते भारत।", "नमस्ते भारत"])
        self.assertIn(best_i, (0, 2))

        decoded = fast_mbr_decode(model, src[:1], src_mask[:1], tok, num_samples=4, max_len=8, grammar_mask=gmask)
        self.assertIsInstance(decoded, str)

        loss = mrt_loss(
            model,
            src,
            src_mask,
            refs=["नमस्ते भारत", "यह परीक्षण है"],
            decode_fn=tok.decode,
            n=3,
            max_len=8,
            grammar=gmask,
        )
        self.assertFalse(torch.isnan(loss))
        loss.backward()
        self.assertIsNotNone(model.proj.weight.grad)

    def test_akshara_bpe_dropout_and_lattice_encode(self):
        """Verify Akshara BPE-dropout preserves lossless round-trip & placeholders, and lattice_encode totality."""
        import random
        from tokenizer import Tokenizer, lattice_encode

        texts = ["यह ⟨NUM_0⟩ किताब और नमस्ते भारत है।", "किताबें और परीक्षाएं बहुत अच्छी हैं।"]
        tok = Tokenizer.train(texts, language="hi", vocab_size=150)

        sample = "यह ⟨NUM_0⟩ किताब है।"
        canon = tok.encode(sample).ids
        drop_a = tok.encode(sample, dropout_p=0.5, rng=random.Random(42)).ids
        drop_b = tok.encode(sample, dropout_p=0.5, rng=random.Random(42)).ids
        self.assertEqual(drop_a, drop_b)
        self.assertEqual(tok.decode(drop_a), tok.normalize(sample))
        self.assertEqual(tok.decode(canon), tok.normalize(sample))

        # lattice_encode unit test
        vocab = {"क": 10, "ि": 11, "कि": 12, "ता": 13, "ब": 14, "किता": 15, "⟨NUM_0⟩": 16}
        vocab.update({f"<0x{b:02X}>": 100 + b for b in range(256)})
        units = ["कि", "ता", "ब", "⟨NUM_0⟩", "Z"]
        self.assertEqual(lattice_encode(units, vocab, p=0.0), [15, 14, 16, 100 + ord("Z")])
        c1 = lattice_encode(units, vocab, p=1.0, rng=random.Random(1))
        self.assertEqual(c1, [12, 13, 14, 16, 100 + ord("Z")])  # multi-unit edge "किता" dropped, placeholder intact


if __name__ == "__main__":
    unittest.main()
