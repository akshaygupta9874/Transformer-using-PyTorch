# en→hi NMT: four optimisation areas — findings and drop-ins

All code is in this folder and runs against the exact APIs in your spec. **32 tests pass** (CPU, PyTorch 2.15 in my sandbox).
What I could *not* test: your real model/tokenizer/`MemmapPairs`, CUDA/BF16 on your RTX 5060, Windows itself, and whether any change moves BLEU/chrF. Spots where I had to guess at code I can't see are marked **ADAPT**.

| File | Replaces / adds |
|---|---|
| `research/chunked_losses.py` | `_ChunkedLinearCE` + `chunked_cross_entropy` (adds z-loss, `normalizer`, `return_info`), new `chunked_rdrop_loss`, `chunked_nll_sum`, `chunked_token_logprobs` |
| `research/accum.py` | token-normalised accumulation loop (`windows`, `train_window`) |
| `research/dynamic_batching.py` | megabatch-shuffled `TokenBucketBatchSampler` |
| `research/optim_groups.py` | param groups that understand tied embeddings |
| `research/logits_utils.py` | single-`log_softmax` helpers for greedy / beam / sampling / kNN-MT |
| `research/tie_diagnostics.py` | measures the two gradient paths into the tied `W` |
| `scripts/build_memmap_stream.py`, `dataset_cache.py` | streaming builder, cached dataset |

---

## 1. Weight tying (`models/modern.py`)

**Is there a ~22.6× gradient imbalance on the shared `W`? No — not at init.** I measured it on a toy 4-layer Pre-RMSNorm tied decoder (d=512, V=40k, your init recipe, Zipf-random tokens), splitting the gradient into the embedding path and the projection path (`research/tie_diagnostics.py`):

| init std | `embed_scale` | embedding rms | sublayer-output rms | ‖g_emb‖ / ‖g_proj‖ | init logit std |
|---|---|---|---|---|---|
| 0.02 | 22.63 (**yours**) | 0.452 | 0.023 | **0.64** | 0.45 |
| 0.02 | 1.0 | 0.020 | 0.035 | 1.14 | 0.45 |
| 0.0442 (d⁻½) | 22.63 | 0.998 | 0.023 | 0.27 | 1.01 |
| 0.0442 | 1.0 | 0.044 | 0.031 | 1.46 | 1.00 |

Why the factor cancels: while the embedding dominates the residual stream, the pre-final-norm rms is ∝ `embed_scale`, and the RMSNorm Jacobian divides the gradient by it. Also, AdamW normalises step size, so a larger raw gradient cannot become a larger step; the relative update per element is `lr/|W|` in both roles.

**Caveats:** at init only, a toy, random targets. As sublayer outputs grow, the embedding path's share can only shrink. Run `tied_grad_split` on the real model at step 0 and at ~1–2k steps.

**Recommendations**
- **Keep `embed_scale=√d`, std 0.02.** Switching to `std=d⁻½` makes balance worse (0.27) and doubles initial logit spread; `embed_scale=1` is also fine but there is no gradient-based reason to move. The real effect of `√d` is that the raw token vector dominates the residual stream (~20× the sublayer writes) early on; warmup covers it.
- **Weight decay: 0 on tied `W` and `src_embed`** (`build_param_groups` does this by default; flag `decay_embeddings=True` reverts). Decoupled decay shrinks *every* row every step whether or not it received gradient; `src_embed` rows for rare English subwords get no gradient most steps. Total shrink ≈ `exp(-λ·Σlr)`: e.g. 100k steps at mean lr 1.5e-4 and λ=0.01 → ~14%. Small either way; this is low-risk, not a proven win.
- **z-loss: yes, `1e-4`.** Cross-entropy with label smoothing is *exactly invariant* to adding a constant to all logits of a position (`lse`, `z_y` and `mean(z)` all shift by c and cancel), so nothing in the loss pins the logit offset, and BF16 resolution worsens as magnitudes grow (my chunk logits are BF16 matmul outputs cast to FP32, so rounding happens before the FP32 `logsumexp`). z-loss pins `lse`.
  Exact change, implemented in `_ChunkedLinearCE`:
  - forward: `l = lse - (1-eps)*zy - eps*z.mean(-1);  if z_loss: l = l + z_loss * lse.square()`
  - backward: `gz = softmax * (1 + 2*z_loss*lse) - eps/V;  gz[y] -= (1-eps)`
  
  Verified by float64 `gradcheck` and against naive autograd. Log `info["lse_mean"]`/`["lse_max"]` to watch it. `1e-4` is the PaLM value; I haven't tuned it on your data.
- An FP32-output matmul (`torch.mm(..., out_dtype=torch.float32)`) may remove the BF16 logit rounding on CUDA; the op isn't implemented on CPU so I couldn't test it and did not use it.

## 2. Sampler and accumulation

**Accumulation.** `loss_mb_mean / ga` gives each micro-batch equal weight regardless of token count. `train_window` pulls `ga` micro-batches on CPU, counts valid tokens (or Σ`sample_weight`) with no GPU sync, then runs each with `reduction="sum", normalizer=window_total`. Only one micro-batch is on the GPU at a time; clipping happens once per window. Test (toy, micro-batches of very different size): new gradients equal one big-batch token-mean gradient to 1e-6; the old scheme differs by up to 0.21.
The same `normalizer` works for `chunked_rdrop_loss`. **ADAPT:** `default_forward_hidden` assumes batch keys `src, src_mask, tgt_in, tgt_mask, tgt_out[, weight]`.

**Sampler.** Permute → megabatches of 100k → sort each by length → greedy maximal batches (`count × max_len ≤ max_tokens`, `count ≤ 128`) → shuffle batch order. `set_epoch(epoch, start_batch=0)`; the plan is built once per epoch and cached, so `__len__` + `__iter__` cost one pass.
- 10.1M rows: **1.1 s and 0.2 GB per epoch plan**.
- Padding efficiency (synthetic lognormal lengths, 400k rows): 0.998 vs 0.999 for a global sort. With a 2k megabatch it drops to 0.925, so **keep the megabatch ≥ ~50k**. Check yours with `padding_efficiency()`.
- Batch membership: <2% of batches repeat between epochs.
- `num_buckets` is accepted and ignored. Use `expected_len` (epoch-0 count) for the scheduler's total steps; other epochs differ by a few batches.

## 3. Double `log_softmax` and chunked R-Drop

**`chunked_rdrop_loss(h1, h2, W, target, label_smoothing, rdrop_alpha, ...)`** computes `ce_weight·(CE1+CE2) + α·½(KL(p1‖p2)+KL(p2‖p1))` chunk by chunk with a custom backward (derivation in the module docstring). Matches naive autograd in float64 to 1e-9; `gradcheck` passes. `ce_weight=0.5` means `(CE1+CE2)/2`; use 1.0 for the paper form — **match whatever `train.py` does now**. Default `chunk_size=256`.
Memory (arithmetic, not measured on GPU): one FP32 `(N,V)` tensor for a 4,096-token micro-batch is 655 MB, and the naive path keeps several per branch; the chunked path holds ~8 chunk-sized tensors (~41 MB each at 256). Cost: one extra projection matmul per branch in backward. The two decoder passes still double activation memory.

**Should `project()` return raw logits? Don't redefine it — rename it.** A silent semantic flip means any forgotten caller (`knn_mt`, `mrt_loss`, eval scripts) treats logits as log-probs with no error. Keep `project_logits` as the primitive, add `log_probs(x, temperature)`, make `project()` a warning shim, migrate callers, then delete it so stragglers fail loudly. Patch text is in `research/logits_utils.py`.
- greedy: `argmax` on raw logits (no softmax)
- beam/MBR: one FP32 `log_softmax`
- sampling: temperature, mask, top-p/top-k on logits, one softmax
- kNN-MT needs real log-probs (`interpolate_knn_log_probs`)

Two things in your list are not correctness bugs: `log_softmax` is idempotent, so the extra call in `beam_search_decode` only wastes a pass; and `softmax(log_softmax(z)/T) == softmax(z/T)`, so `fast_nucleus_sample` is right unless it ends in a bare `.exp()`.
Mask order: my default is "log_softmax then mask" (scores stay true model log-probs); `renormalize_mask=True` masks first. **Verify** that `DevanagariGrammarMask.apply` writes `-inf` and that this matches your current beam behaviour.
For validation use `chunked_nll_sum`; for MRT/MBR token scores use `chunked_token_logprobs`.

## 4. Streaming builder and cached dataset

`build_memmap(pairs_iterator, out_dir, ...)` / CLI: chunk generator → spawn `Pool` with `apply_async` and a bounded deque (explicit back-pressure; `Pool.imap` would drain the whole generator into its queue) → append to `.bin` files; results are written in order, atomically renamed from `.partial`.
- Synthetic 1M-pair run: **peak RSS 114 MB, ≤4 chunks in flight**. The only O(N) state is lengths + quality (~120 MB at 10.1M).
- Output is byte-identical for 1 or 3 workers; `--mode srcdrop` writes row-aligned BPE-dropout shards (RNG seeded by chunk index).
- Real throughput is unknown (my fake tokenizer ran 95k pairs/s); try `--limit 50000` first.
- **ADAPT:** `--tok-factory module:callable(path)`; `--subst` return shape `(src', tgt', ...)`; file names (top of the script); BOS/EOS flags; dropout `rng` type (I use `random.Random`). The `hf` source is untested (no network).

`CachedBilingualDataset` runs the builder once into `cache_dir`, keyed by tokenizer hashes + flags + `seq_len` + input fingerprint (hit/miss tested), then serves `__getitem__` as two memmap slices and exposes `.lengths` for the sampler. If `MemmapPairs` already reads this format, point it at `cache_dir` instead.

## Suggested order
1. `build_param_groups`, 2. z-loss `1e-4` + new `chunked_cross_entropy`, 3. `train_window` + new sampler (re-derive total steps from `expected_len`), 4. log `tied_grad_split` at step 0 and ~2k, 5. rename `project`, 6. `chunked_rdrop_loss` before Phase 2, 7. streaming builder.
