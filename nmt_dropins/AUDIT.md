# Audit round 2 — merge verdicts, hidden bugs, 8 GB settings, accuracy roadmap

## 0. Scope and limits (read first)
- I can see **only what your message describes**, not `train.py`, `models/modern.py`, `research/memmap_data.py`, `research/optimization.py` or `research/fast_decoding.py`. So I did **not** write replacement files or unified diffs for them: a diff against code I haven't read would be fiction. Section 3 gives integration sketches keyed to your function names; paste the files listed in §7 and I'll produce exact diffs.
- Everything below marked **measured** was run in my CPU sandbox (PyTorch 2.15). Anything about your RTX 5060, Windows itself, or BLEU/chrF++ is **not measured**; those are priors plus a script to measure them (`research/profile_sweep.py`).
- **47 tests** now pass in the sandbox (32 from round 1 + 15 new).

## 1. Merge verdicts for `nmt_dropins/`

| Module | Verdict | Conditions / changes |
|---|---|---|
| `research/chunked_losses.py` | **Merge now** | Replaces `chunked_cross_entropy`; new args are keyword-only. Use `z_loss=1e-4`. Pass `compute_dtype=torch.bfloat16` explicitly if you call it *outside* the autocast block (otherwise it follows `hidden.dtype`, and your residual stream is likely FP32 → slow FP32 matmul + 2× memory). It accepts `(B,T)` per-token weights, so the "`token_weighting` disables chunked CE" restriction can go. |
| `research/accum.py` | **Merge, adapting** | Don't paste over your loop if it carries EMA/MRT/logging/scaler logic; fold in `windows()`, `window_normalizer()` and the `normalizer=` loss call. Added this round: non-finite-gradient skip (a BF16 spike no longer poisons AdamW moments and EMA). `ema.update(model)` is my guess at your `ModelEMA` API. |
| `research/dynamic_batching.py` | **Merge** | Check that `lengths[i]` means what your old sampler used (max vs sum of src/tgt). Use `expected_len` for scheduler steps. Feed it **exact** lengths (see bug 6). |
| `research/optim_groups.py` | **Merge** | Only if `build_optimizer_param_groups` has no extra LR groups; otherwise port the rule (no decay on `nn.Embedding` incl. tied weight, no decay on ndim<2). |
| `research/logits_utils.py` | **Merge in steps** | 1) add `log_probs()`; 2) migrate `fast_decoding.py`, `decoding.py`, `knn_mt.py`, `train.py`; 3) delete `project()` so stragglers fail loudly. |
| `research/packed_ref.py` | **Use as a test oracle, not runtime** | Run `check_packed_equivalence` against your `packed_masks`/`pack_pairs`. Keep `use_packed_batches: false` (see §4). |
| `research/tie_diagnostics.py` | One-off script | Run at step 0 and ~2k. |
| `scripts/build_memmap_stream.py` | **Merge, after a round-trip test** | Build 10k pairs with the old and new builders and diff the `.bin`/offset files. Adapt file names, BOS/EOS flags, `--tok-factory`. |
| `dataset_cache.py` | **Skip if `memmap_dir` is set** | For staging runs without a memmap it replaces `BilingualDataset` and supplies exact lengths. |
| `research/eval_tools.py`, `profile_sweep.py`, `shared_epoch.py` | **New, merge** | See §2. |

## 2. Hidden bugs and edge cases
**Found in my own round-1 code (fixed in this round's zip):**
1. **`np.memmap` pickles by value.** `CachedBilingualDataset` pickled to **33 MB for 27 MB of token data** (**measured**), and that copy goes to *every* spawn worker; at 10M rows that is gigabytes × 4 workers on a 24 GB machine. Fixed with `__getstate__`/`__setstate__` (pickle now 0.0 MB) plus `close()`. **Run `assert_spawn_safe(ds, collate, sampler)` (in `tests/test_spawn_safety.py`) on your `MemmapPairs` — any dataset holding an `np.memmap` has this problem unless it re-opens lazily.**
2. **No guard against inf/NaN gradients** → added (`skipped=1.0`, optimizer untouched).
3. **Per-row `sample_weight` is wrong for packed rows** (a row holds several documents). Use `expand_doc_weights(doc_ids, doc_weights)` → `(B,T)`.
4. My test harness itself hit the classic Windows trap: a spawn run from a script with no `if __name__ == "__main__":` guard re-executed itself in every worker.

**In the code you describe (reasoned or measured as stated; not verified against your files):**
5. **Persistent workers never see parent-side attribute changes** (**measured**: parent set `epoch` 0,1,2 → workers saw `[0]` every time). If `MemmapPairs` picks the `.v1/.v2` BPE-dropout shard from an epoch attribute, it silently uses one shard forever with `persistent_workers=True`. Use `research/shared_epoch.py` (tested with spawn + persistent workers). The batch sampler is unaffected (it lives in the main process).
6. **`_estimate_pair_lengths` with `len(txt)//4`**: any under-estimate makes real batches exceed `max_tokens` (OOM risk on 8 GB); over-estimates waste capacity. Use exact token lengths from the memmap offsets (`np.diff`) or `CachedBilingualDataset.lengths`. *(Reasoned, not measured.)*
7. **Packed R-Drop second pass (`train.py:847`)**: structural fix = one `forward_hidden(model, batch)` used for **both** passes (see `packed_forward_hidden`), so masks and positions cannot diverge. My negative controls show the equivalence test **fails** when positions are dropped or masks leak across documents (**measured**), so it would catch this bug.
8. **Fully masked attention rows**: on my PyTorch 2.15 **CPU** build, SDPA returns zeros (finite grads) for an all-False row (**measured**), but I cannot test your CUDA backends, and older versions gave NaN. The reference masks guarantee ≥1 allowed key per row. Check that your `(tgt_doc == 0)[:, None, :, None]` mask is a *row-zeroing* multiply applied **after** attention, not something that creates all-False rows; paste `packed_masks` and I'll check it against the oracle.
9. **Validation drift**: val rows from a random split change when `max_train_samples` changes, so 1M→2M val losses aren't comparable, and val shares near-duplicates with train. `hash_split(src)` is keyed on source text only (stable when the dataset grows; **tested**).
10. **Placeholder smoke files**: `evaluate_benchmark` warns when a benchmark has <100 lines, so a 3-line smoke file can't masquerade as IN22-Gen.
11. **Windows text encoding**: `open()` without `encoding="utf-8"` uses the ANSI code page on Windows and will mangle Devanagari TSVs; grep your repo for `open(` lacking `encoding=`. Console prints of Devanagari need `PYTHONUTF8=1`. *(General Windows behaviour, not seen in your code.)*
12. **File replacement**: `os.replace` onto a file that is still memory-mapped fails on Windows. Build into a new directory and switch the config; call `close()` first.
13. **BF16 checklist for `models/modern.py`** (cannot verify, please confirm): RMSNorm and QK-Norm computed in FP32; RoPE cos/sin tables FP32; residual stream dtype entering `decoder_final_norm` (FP32 is good); boolean masks only; no `.half()` on logits. The chunked CE already does `logsumexp` in FP32.

## 3. Integration sketches (keyed to your names; **not verified diffs**)
- **Loop**: replace `loss / ga` + per-micro-batch backward with
  `for window in windows(train_loader, ga): stats = train_window(model, window, optimizer, device=device, forward_hidden=fwd, z_loss=1e-4, rdrop_alpha=cfg.rdrop_alpha, scheduler=scheduler, ema=ema)`.
  Scheduler total steps = `ceil(batch_sampler.expected_len / ga) * epochs`.
- **Per epoch**: `batch_sampler.set_epoch(e); shared_epoch.set(e)`.
- **R-Drop**: delete the branch that disables chunked CE; `chunked_rdrop_loss` replaces the two `(B*T,40000)` tensors. One `fwd` for both passes.
- **Optimizer**: `make_optimizer(model, lr=3e-4, eps=1e-9, betas=<your current>)`.
- **Validation**: `evaluate_benchmark(translate_fn, "data/benchmarks/in22_gen.tsv", {"chrF++": ..., "BLEU": ...})` every N steps on a 200–500 sentence slice; full sets at checkpoints. Log `copy_rate`.
- **Lengths**: from offsets, not `len(txt)//4`.

## 4. Settings for the 8 GB RTX 5060 (priors; verify with `profile_sweep.py`)

| Setting | Recommendation | Basis | Confidence |
|---|---|---|---|
| `kv_heads` | **8 (keep `null`)** | GQA saves 4.2M params (5.5%) and nothing in training speed. KV-cache arithmetic: 2 KB per token per layer × 4 layers = 8 KB/token; beam 4 × batch 32 × 350 tokens ≈ 0.36 GB, irrelevant on 8 GB. | Medium. A 1M-pair A/B would settle quality. |
| decoder `d_ff` | **2048 (keep)** | Parameter accounting: encoder ≈27.3M + decoder ≈21.0M + embeddings ≈28.7M ≈ 77.0M (matches your 76.92M). The tied 40k output projection alone is ≈20.5M MAC/token, about the whole decoder stack, so cutting FFN to 1536 saves ≈3.1M params (≈4% of total) for a risk to the capacity-limited side. | Medium |
| `use_packed_batches` | **false** | My sampler's padding efficiency was 0.998 vs 0.999 for a global sort (synthetic lognormal lengths; **check `padding_efficiency()` on your real lengths**). Packing can recover ≲2% while adding the mask/position/weight failure modes above. Revisit only if real efficiency < ~0.95. | Medium-high |
| `max_tokens_per_batch` | **Find it, don't guess:** `find_max_tokens(fits)` with ~12% VRAM headroom | Static memory ≈1.7 GB by arithmetic (weights 0.31 + grads 0.31 + Adam 0.62 + EMA 0.31 + BF16 weight cache ≈0.15); activations unknown. Windows/WDDM also holds back some VRAM. | — |
| effective tokens/step | **~24–32k padded tokens** → `ga ≈ round(28k / max_tokens)` | Transformer-base used ≈25k src + 25k tgt tokens/step; you are at ≈16k (4096×4) if `max_tokens` counts padded tokens per side. Larger batches usually help from-scratch runs; confirm with grad-norm noise and dev loss. | Low-medium |
| `compile_model` | **false** until A/B | Token bucketing gives many shapes → recompiles unless shapes are bucketed/`dynamic=True`; Windows Triton support for your build is something I can't verify. Never compile through `_ChunkedLinearCE`. A/B 200 steps (tokens/s, VRAM). | Low |
| `z_loss` | **1e-4** | See ANALYSIS.md §1. Log `lse_mean/lse_max`. | Medium |
| `ce_chunk_size` | 1024 (512 if tight) | Chunk transient ≈0.5 GB at 1024. Recompute makes the projection ≈4 matmuls instead of 3 regardless of chunk size; measure the tokens/s cost in the sweep. | — |
| EMA | `0.9999` is **never reached** on a 1M run | With your ramp `min(0.9999,(1+s)/(10+s))` decay hits 0.9999 only at step ≈90k; a 1M-pair epoch is on the order of 2k steps (≈30 tokens/pair, 16k tokens/step; assumptions). Pick EMA vs last-5 averaging on dev chrF++, not by habit. | Medium |

**Phases.** *Phase 1*: CE (ε=0.1) + z-loss, source BPE-dropout p≈0.05–0.1 (alternating `.v1/.v2` per epoch via `SharedEpoch`), mild quality weights (normalize to mean 1, clip), EMA on, cosine to ~5% of peak. *Phase 2* (last ~10% of the budget, from the Phase-1 average): chunked R-Drop. For reference, as I recall the R-Drop paper used α=5 with `CE1+CE2`, which is ≈2.5 in your `(CE1+CE2)/2` convention, so 0.3–0.5 is much weaker; try α ∈ {0.5, 1, 2.5} (verify the paper's number). Then checkpoint averaging, then MBR at inference. MRT last: expensive, least certain.

## 5. Accuracy roadmap (ordered by expected ROI per unit effort; gains are priors, not promises)
1. **Measure properly first.** Real benchmark files (not smoke), hash-split dev set, chrF++ + BLEU + copy-rate, and a neural metric (COMET-family) on a subset. Make every decision on dev, report on benchmarks once. IN22-Conv is 1,503 conversational sentences (CC-BY-4.0, per its dataset card); evaluate it separately from IN22-Gen.
2. **Data filtering ablation on the 1M slice** (3–4 short runs): none vs LaBSE thresholds vs LaBSE + script-purity + length ratio. Free extra filter: `copy_preservation(src, tgt)` on *training pairs*. Pairs whose digits/URLs disagree are likely misaligned; down-weight rather than drop when numbers are written as words.
3. **Domain gap.** Samanantar is mined data of mixed register; IN22-Conv is conversational. Check the per-source composition of your slice. IndicTrans2's BPCC is 230M pairs including 644K manually translated ones ([paper](https://arxiv.org/abs/2305.16307v1)); its HF page is gated, so check access and license before relying on it, and confirm no overlap with your frozen benchmarks (your leakage assertion covers this).
4. **Back-translation** (Hindi monolingual → synthetic English): train hi→en with the same stack, translate monolingual Hindi, mix in at a capped ratio with lower weight (you have no spare special ID for a tag, so use the quality-weight column). Usually the largest single gain for low/mid-resource, but at 10M pairs expect less; most valuable if the monolingual text matches IN22-Conv's register. Monolingual source: IndicCorp ([paper](https://arxiv.org/pdf/2212.05409)); I did not verify download terms.
5. **Schedule/scale-up.** Use 1M/2M for ablations and LR choice (short range test over peak LR 3e-4/5e-4/8e-4), then train the 10.2M run **from scratch** with the final recipe. If you must warm-start, re-warm to ≈0.5× peak. Cosine to ~5% of peak; average the tail.
6. **Decoding.** Tune beam α/β on dev (grid), then try MBR (chrF++ utility, N=16 ε-samples) on dev; it costs ~16× decoding, so use it for final outputs.
7. **Placeholder integrity.** Track a *placeholder drop rate* (source `⟨NUM_i⟩`/`⟨URL_i⟩` missing from the raw hypothesis, before restoration). If it is >~0.5%, add a decoding constraint or an MBR penalty; if tiny, skip.

## 6. Windows `spawn` checklist
`if __name__ == "__main__":` guard and no heavy module-level work (every worker re-imports); lazy tokenizer loading; datasets re-open memmaps in `__setstate__` (test: pickle < 5 MB); `SharedEpoch` for per-epoch state seen by persistent workers; `close()` before replacing/deleting mmapped files; `encoding="utf-8"` everywhere; `PYTHONUTF8=1`; exclude the data dir from antivirus real-time scanning (first-touch cost on multi-GB memmaps; general experience, unmeasured); budget ~0.3–0.5 GB RAM per worker for the torch import (typical, unmeasured).

## 7. What to paste back
**Files:** `train.py` (loop, `run_validation`, `build_optimizer_param_groups`, `_estimate_pair_lengths`, the R-Drop block near line 847); `research/memmap_data.py` (`MemmapPairs`, `PackedMemmapCollate`, `pack_pairs`, `packed_masks`); the attention + `encode`/`decode` signatures from `models/modern.py`; `ModelEMA` + `chunked_cross_entropy` call sites in `research/optimization.py`; the beam loop in `research/fast_decoding.py`; `configs/modern.yaml`.
**Measurements:** (a) `profile_sweep` rows for kv_heads {null,4} × `max_tokens` {2048, 3072, 4096}, plus `find_max_tokens` result; (b) a 300-step log with per-step `loss, ce, z, lse_mean, lse_max, grad_norm, lr, tokens/step, step_ms, data_wait_ms` (time blocked on the DataLoader; >5% means loader-bound); (c) `nvidia-smi dmon -s mu -d 5` during training; (d) `padding_efficiency()` and the length histogram of the real 1M slice; (e) `tied_grad_split` at step 0 and ~2k; (f) line counts of `data/benchmarks/*.tsv`; (g) 200 IN22-Gen samples from `write_samples_tsv`; (h) the placeholder drop rate; (i) Windows Task Manager RSS per worker at `num_workers=4`.
