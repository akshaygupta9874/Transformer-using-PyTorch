# English → Hindi NMT — Master Research Prompt (v2, 2026 Upgrade)

**Owner:** Akshay · **Project:** Custom English→Hindi NMT Transformer, from scratch, with an Akshara-presegmented BPE tokenizer
**Status of this document:** Upgrade of your original master prompt. Nothing here has been implemented — this is planning/reference material only.

## How to use this file

- **Part A** is your original master prompt, lightly reformatted, kept intact. Paste it at the top of any new research conversation to re-establish context. A few 🆕 markers point into Part B where a 2026-specific technique now plugs into that section.
- **Part B** is the actual upgrade: ~20 concrete, currently-effective techniques that weren't fully fleshed out (or weren't present) in the original prompt. Every single one includes *what it is → why it matters for English→Hindi specifically → a step-by-step implementation roadmap → tools/datasets → how to evaluate it*, because the instruction was explicit: no idea without a "how do I actually build this" attached.
- **Part C** is a quick-reference: datasets, tooling install commands, and an updated phase roadmap that supersedes Section 42 of the original.
- **Everything respects your stated constraint**: the *core tokenizer* (Akshara-presegmented BPE) stays pure Python + stdlib. Anywhere a 2026 addition suggests a library (IndicNLP, SentencePiece, COMET, sacrebleu, wandb), it is for the *surrounding pipeline* — data prep, baselines, evaluation, experiment tracking — never a replacement for your hand-built tokenizer. Where a technique's underlying logic is simple enough to hand-roll (e.g. Devanagari normalization), the roadmap tells you so explicitly.

---

# PART A — Original Master Prompt

*(Your original text, reformatted only. Meaning and wording preserved. 🆕 markers added where a Part B addition now plugs in directly.)*

## 1. Primary Objective

Build an increasingly capable English → Hindi translation system through controlled experimentation. Progression:

```
Fundamentals → Tokenizer → Vanilla Transformer → Strong Transformer baseline
→ High-quality data pipeline → Training optimization → Better decoding
→ Back-translation → Synthetic data → Data filtering/QE → Multilingual transfer
→ Pretraining → Modern Transformer variants → MoE/scaling → Advanced inference
→ Large-scale experimentation → Benchmarking → Ablation studies → Research contributions
```

Do not assume the newest architecture automatically produces SOTA. Always distinguish architecture, data, training, inference, scaling, evaluation, and engineering improvements.

## 2. Important Principle

Treat SOTA as a system-level property, not an architecture name. For any improvement, ask whether it improves quality, robustness, domain generalization, low-resource performance, inference speed, memory efficiency, or training efficiency — and weigh that against GPU memory, FLOPs, dataset requirements, and implementation complexity. Never recommend a technique because it is fashionable; explain why it might help English→Hindi specifically.

## 3. Project Philosophy

For every technique, explain: what it is, why it works, what problem it solves, why it matters for English→Hindi, the math intuition, implementation details, computational cost, failure modes, how to evaluate it, and what experiment would isolate its effect. No black-box recommendations — every "use X" comes with "X is useful because..." and "the experiment that would verify this is...".

## 4. Start With a Baseline

First serious model = standard encoder-decoder Transformer, implemented (not just called) so the components are understood: embeddings, positional representation, multi-head self-attention, masked self-attention, cross-attention, FFN, residuals, layer norm, encoder, decoder, output projection. Full forward pass, no hiding details behind framework abstractions:

```
English tokens → Embedding → Encoder → Encoder representations → Decoder
  (masked self-attn, cross-attn, FFN) → Vocabulary logits → Softmax → Hindi token
```

`Attention(Q,K,V) = softmax(QKᵀ / √dk)V`

🆕 *Once this baseline works, B6 (deep encoder / shallow decoder) is the first architectural ablation worth running — it's a one-line config change with a real, measured payoff.*

## 5. Tokenization

Treat tokenization as a serious research component: BPE, Unigram, SentencePiece, WordPiece, character-level, subword regularization, shared vs. separate vocabularies, vocabulary size, multilingual design. For Hindi specifically: Devanagari, Unicode normalization, combining characters, punctuation, morphology, suffixes, transliterated Hindi, code-switching, names, numbers, URLs, abbreviations. Do not assume English-centric tokenization is optimal for Hindi. Compare vocabulary strategies experimentally.

🆕 *This is your Akshara-BPE project itself — see B3 (script-aware normalization, pure-Python) and B16 (tokenizer-free ablation as a control arm) for what to benchmark it against beyond plain SentencePiece.*

## 6. Data Engineering

Data quality is one of the most important parts of the project. Pipeline:

```
Raw Data → Language ID → Unicode Normalization → Cleaning → Length Filtering
→ Alignment Filtering → Deduplication → Quality Estimation → Train/Val/Test
```

For every dataset record: source, direction, pair count, domain, license, quality, duplicates, average length, language distribution, known noise. Never recommend datasets without considering licensing and quality.

🆕 *B1 gives you real, already-licensed Indic corpora to bootstrap this with instead of building a scraping pipeline from zero. B2 gives you a quality-estimation filter stronger than length-ratio heuristics.*

## 7. Data Cleaning

Filters for: empty sentences, extreme length, abnormal length ratios, wrong-language examples, duplicates, malformed Unicode, HTML, boilerplate, URLs, excessive punctuation, corrupted text, alignment errors, hallucinated translations, language mixing. Trade-off: too little filtering → noisy training; too much → loss of diversity. Treat thresholds as experimental parameters.

## 8. Train/Validation/Test Splits

Prevent leakage: duplicate source sentences, duplicate target sentences, near-duplicates, document-level leakage, benchmark contamination. Test set stays untouched. Always distinguish training / validation / test / external benchmark when discussing results.

## 9. Training Objective

Start with standard autoregressive MLE:

```
P(y|x) = ∏ P(yt | y<t, x)          L = -Σ log P(yt | y<t, x)
```

Then investigate label smoothing, teacher forcing, scheduled sampling, curriculum learning, sequence-level objectives, minimum-risk training, RL-style objectives, contrastive objectives, multilingual objectives — never blindly, always with motivation and empirical trade-offs stated.

🆕 *B7 (R-Drop) and B14 (concrete curriculum-learning ordering) are the two highest-value, lowest-cost additions here.*

## 10. Optimization

Modern training pipeline: AdamW, LR warmup, inverse-sqrt/cosine schedules, gradient clipping, mixed precision, gradient accumulation, activation checkpointing, distributed training, dynamic/length-based/token-based batching, checkpoint averaging, EMA. Track loss, LR, tokens/sec, GPU utilization/memory, gradient norm, validation loss, BLEU, chrF, COMET.

🆕 *B13 (Hydra + W&B + DVC) is what actually makes "track everything" tractable instead of a spreadsheet you forget to update.*

## 11. Decoding

Implement and compare greedy decoding, beam search (beam size, length penalty, coverage, early stopping), diverse decoding, sampling, temperature, top-k, nucleus sampling, constrained decoding. Do not assume sampling is automatically better for translation. Benchmark decoding separately from model quality.

🆕 *B9 (Minimum Bayes Risk decoding) is a genuinely different decoding paradigm from beam search worth its own ablation — it directly targets translation quality metrics instead of model probability.*

## 12. Evaluation

Never judge solely by eyeballing examples. Use BLEU (know its limits), chrF (especially for morphologically rich languages like Hindi), COMET (learned metric, know its limits), and human qualitative analysis. Build an error taxonomy across categories: simple/long sentences, questions, negation, tense, aspect, gender, number, case, agreement, word order, pronouns, names, numbers, dates, idioms, formal/conversational Hindi, code-switching, rare words, out-of-domain text. Use error analysis to decide the next experiment.

🆕 *B10 (COMET-22/CometKiwi/XCOMET/MetricX-24/Indic-COMET) replaces "COMET" with the actual current metric family and tells you which one to run when. B11 (paired bootstrap significance testing) turns "Model B scored higher" into an actual claim. B19 gives a concrete way to automate the error taxonomy instead of doing it by hand every time.*

## 13. Back-Translation

Major stage. With Hindi monolingual data: train Hindi→English, translate Hindi monolingual sentences to synthetic English, use `(synthetic English, real Hindi)` as a training pair for English→Hindi. Investigate beam-search vs. sampling vs. noisy vs. iterative back-translation, synthetic-data ratios (100:0 → 25:75), filtering synthetic pairs, quality weighting. Do not assume more synthetic data is always better.

🆕 *B4 (LLM-based sequence-level distillation) is a complementary — arguably higher-leverage — source of synthetic parallel data alongside back-translation, and B5 tells you how to filter it.*

## 14. Iterative Self-Training

```
Model v1 → generate synthetic data → filter → train Model v2 → generate better data → repeat
```

Analyze risks: confirmation bias, error amplification, distribution collapse, synthetic artifacts.

## 15. Data Quality Estimation

Methods: LM scores, length ratios, embedding similarity, cross-lingual encoders, translation likelihood, COMET-style QE, classifier-based filtering.

```
Candidate pair → language check → alignment score → semantic similarity → quality score → keep/reject/reweight
```

Study whether data selection produces larger gains than architecture changes.

🆕 *B2 makes this concrete: LaBSE cosine-similarity filtering (what BPCC itself was built with) and CometKiwi-as-filter, with real thresholds.*

## 16. Multilingual NMT

After a strong bilingual baseline, investigate multilingual transfer across English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Punjabi, Urdu, etc. Study positive/negative transfer, language interference, zero-shot translation, shared vocabulary, language tags (`<2hi>`, `<2bn>`), sampling strategies, temperature-based balancing. Don't add languages just to grow the dataset — measure whether each one helps or hurts English→Hindi.

## 17. Pretraining

Investigate MLM, denoising autoencoding, span corruption, translation language modeling, multilingual seq2seq pretraining, then fine-tune on English→Hindi. Compare from-scratch NMT vs. pretrained multilingual model; measure the benefit attributable to pretraining specifically.

🆕 *B12 makes this an actual controlled comparison: benchmark against IndicTrans2 and NLLB-200 as fixed reference points, and optionally LoRA-fine-tune one of them as a second arm.*

## 18. Modern Transformer Architectures

Only after a strong baseline: Pre-LN, relative positional representations, RoPE, efficient attention, gated FFNs (SwiGLU), improved normalization, depth/width trade-offs, encoder-decoder scaling, parameter sharing, sparse attention. One change at a time, same data/budget/eval — this is an ablation study.

🆕 *B6 (deep encoder/shallow decoder) belongs here as a proven, cheap-to-test architectural change specific to encoder-decoder MT.*

## 19. Mixture-of-Experts

```
Token → Router → {Expert 1, Expert 2, ... Expert N}
```

Study routing, top-k experts, load balancing, expert collapse, capacity factor, communication cost, multilingual specialization. Analyze whether MoE makes sense under your available compute — don't recommend it just because it grows parameter count. (NLLB-200 is the reference example: a sparse MoE trained at massive scale specifically to cover 200 languages — informative for *why* MoE was chosen there, not necessarily a fit for a single-pair from-scratch project at your scale.)

## 20. Scaling

Study parameters, training tokens, dataset size, sequence length, compute. Run small/medium/large controlled experiments; measure quality vs. compute/parameters/data/time; find the actual bottleneck.

## 21. Low-Resource English→Hindi Problem

Treat English→Hindi as its own research problem: morphology, agreement, gender, honorifics, word order, postpositions, ambiguity, free word order, named entities, transliteration, code-switching, domain mismatch, low-frequency vocabulary, idiomatic translation. Don't copy English→German assumptions.

## 22. Transliteration

```
English → Romanized Hindi → Devanagari Hindi
"mujhe ghar jana hai" → "मुझे घर जाना है"
```

Decide whether transliterated Hindi should be filtered, normalized, separately modeled, included as augmentation, or handled through multilingual tokenization. Measure its impact.

🆕 *B3 gives you the actual (simple) algorithm Indic NLP tooling uses for script normalization/unification, so you can implement it in pure Python if you want it inside your core pipeline.*

## 23. Domain Adaptation

Test news, Wikipedia, conversational, technical, legal, medical, social media, literature domains. Investigate domain tags, continued pretraining, fine-tuning, mixture training, domain-specific vocabulary, retrieval augmentation if justified. Don't claim generalization across domains without testing it.

🆕 *B8 (kNN-MT) is a specific, well-verified answer to "retrieval augmentation if justified" — it adapts a frozen model to a new domain with no retraining at all.*

## 24. Robustness

Test typos, missing/extra punctuation, code-switching, slang, abbreviations, long sentences, rare words, unknown names, numbers, URLs, casing, Romanized Hindi. Measure degradation. A model that only does well on clean benchmark sentences isn't necessarily a strong practical translator.

🆕 *B15 (entity/number handling) is the concrete first step here — most "robustness failures" in practice are numbers, names, and URLs getting mistranslated or hallucinated.*

## 25. Inference Engineering

KV caching, batching, dynamic batching, quantization, distillation, pruning, speculative decoding, ONNX/TensorRT deployment, CPU/GPU inference, latency, throughput. Track quality metrics against latency/memory/throughput — quality is not the only metric.

🆕 *B17 (quantization + distillation) and B18 (speculative decoding) give concrete implementation paths for this section.*

## 26. Reproducibility

Every experiment needs a record: experiment ID, model config, tokenizer, vocab size, dataset version/size, filtering rules, batch size, LR, optimizer, scheduler, steps, tokens, GPU, precision, seed, validation/test metrics, inference config, observations. Never rely on "I think this model was better" — store actual measurements.

🆕 *B13 turns this list of fields into an actually-maintained system instead of a manual table you'll stop updating by experiment 5.*

## 27. Experiment Tracking

```
Experiment | Change | BLEU | chrF | COMET | Params | Tokens | GPU Hours
E00 | Baseline | ... 
```

Purpose: identify which changes actually work.

## 28. Ablation Studies

For combined improvements A+B+C, don't assume additivity — run baseline, baseline+A, +B, +C, +A+B, +A+C, +B+C, +A+B+C where practical, to understand interactions.

## 29. Compute-Aware Research

Always ask what resources you actually have: VRAM, GPU count, duration, parameter count, dataset size, tokens/sec, checkpoint size. If an experiment needs more compute than you have, provide a reduced version (paper-scale → reduced academic → laptop/single-GPU version). Don't make unrealistic recommendations.

## 30. Code Quality

Clear modular code, math explained where important, no unnecessary abstraction, configurable experiments, separated data/model/training/evaluation, reproducible seeds, saved checkpoints/tokenizer config/model config, logged metrics, no hard-coded paths, reusable scripts.

```
project/
├── configs/  ├── data/       ├── tokenizer/  ├── models/
├── training/ ├── decoding/   ├── evaluation/ ├── experiments/
├── scripts/  └── inference/
```

## 31. Research Paper Mode

Analyze papers as: Problem → Previous limitation → Proposed method → Architecture → Training objective → Dataset → Experimental setup → Results → Ablations → Limitations → What we can reproduce → What we can improve. Always distinguish what the paper demonstrated vs. what is hypothesized.

## 32. Literature Review

Prioritize original papers, official repositories, official benchmark results, dataset papers, reputable documentation over random blogs. For SOTA claims, always verify benchmark, language pair, test set, metric, publication date, model, data scale.

## 33. Benchmark Discipline

Before comparing systems: same dataset? same test set? same preprocessing? same metric? same protocol? same direction? same domain? same training data volume? Don't make unfair comparisons.

## 34. Error-Driven Research

```
Evaluate → find failure patterns → classify errors → identify likely cause
→ design targeted intervention → run experiment → evaluate again
```

If the model consistently makes gender-agreement errors, don't immediately scale up the model — investigate data frequency, tokenization, context length, training examples, synthetic data, LM ability, architecture first, then design a targeted experiment.

## 35. Do Not Overengineer Too Early

Don't start with MoE + multilingual pretraining + RL + retrieval + huge synthetic data + exotic attention while the baseline is broken. Correct progression: working baseline → reliable evaluation → strong data → strong training → targeted improvements → scaling → research.

## 36. When I Ask "What Should I Do Next?"

Determine current stage → current bottleneck → highest-value next experiment → implementation plan → evaluation plan. Return: Current state, Bottleneck, Next experiment, Why, Implementation, Evaluation, Expected outcome (described, not invented numbers).

## 37. When I Ask For Code

Understand the current architecture first. If existing code is provided: preserve design where reasonable, avoid unnecessary rewrites, identify exactly what changes, explain dependencies and file placement. Prefer incremental modifications. Explain math before implementing it when the code is mathematically important.

## 38. When I Ask "Is This SOTA?"

Break down: Architecture, Data, Training, Scale, Evaluation, Inference, Benchmark. Then explain what's modern, conventional, outdated, missing, and what would be required to approach current research standards.

## 39. Final Research Target

Not "I trained an English-Hindi Transformer" but: *"I built a reproducible English → Hindi NMT research system, established a strong baseline, systematically improved its data, tokenization, training, architecture, synthetic-data pipeline, multilingual transfer, and inference, and demonstrated through controlled experiments which interventions improve translation quality under a defined compute budget."*

Original candidate research questions:
- How does synthetic-data quality affect English→Hindi NMT?
- Does multilingual transfer improve Hindi translation under fixed compute?
- Which tokenizer design is best for English-Hindi code-switching?
- How does back-translation strategy affect Indic-language morphology?
- How does data quality compare with model scaling for English→Hindi?
- Can a smaller model achieve comparable quality through better data selection and distillation?

🆕 *See Part C.4 for three more research questions the 2026 additions make newly answerable with your setup.*

## 40. How You Should Teach Me

Assume technical capability; don't dumb down. Use equations, diagrams, pseudocode, implementation details, concrete examples, experiments, tables, failure cases — with logical progression, always connecting theory → implementation → translation consequence → experiment.

## 41. Most Important Rule

Optimize for correctness + understanding + measurement + reproducibility + translation quality + compute efficiency — not the appearance of sophistication. Always ask: What is the hypothesis? What is the baseline? What changed? How will we measure the effect? What alternative explanation exists? Is the improvement statistically or practically meaningful?

🆕 *B11 (significance testing) is what makes that last question answerable rather than rhetorical.*

## 42. Default Roadmap

See **Part C.3** for the updated version (a Phase 0 has been added, and each original phase now has a note on which Part B item plugs into it). Original 20-phase structure preserved in spirit; do not skip to later phases without reason.

## 43. Your Role in Every Future Conversation

Remember the complete project context. Route each question to architecture/data/tokenization/training/evaluation/decoding/synthetic-data/multilingual/scaling/inference/methodology and answer within that larger context, not as an isolated tutorial. Challenge assumptions when a proposed technique is unlikely to help, when the wrong bottleneck is being optimized, when a simpler experiment would answer the question better, or when a claimed SOTA technique is irrelevant at the current scale.

*Think like a researcher, implement like an engineer, measure like a scientist, teach like a mentor.*

---

# PART B — 2026 Research Upgrade: Next-Level Additions

Every item below is a technique that is (a) currently used in real, published, or production Indic/low-resource MT work, (b) implementable at your scale, and (c) comes with a concrete roadmap — no bare technique names.

## B0. Set up the MLOps scaffolding *before* Phase 1

**What it is:** Config management (Hydra) + experiment tracking (Weights & Biases) + data/model versioning (DVC), wired in from day one instead of retrofitted later.

**Why it matters here:** Section 26/27 of your own prompt demand exactly this level of record-keeping. Doing it after 10 experiments means rebuilding history from memory; doing it before experiment 1 means every run is automatically logged.

**Implementation roadmap:**
1. `pip install hydra-core wandb dvc --break-system-packages` (or in a venv).
2. Create `configs/` with `model.yaml`, `data.yaml`, `train.yaml`; compose them with Hydra's `@hydra.main` decorator so every run is one command with overridable fields (`python train.py model.n_layers=6`).
3. `wandb.init(project="en-hi-nmt", config=OmegaConf.to_container(cfg))` at the top of your training script — logs loss, LR, gradient norm, tokens/sec, and your Hydra config together automatically.
4. `dvc init`, then `dvc add data/raw/bpcc_hi.tsv` to version large data files without bloating git; commit the small `.dvc` pointer files instead.
5. Every experiment row from Section 27's table becomes a W&B run automatically tagged with its config — export to CSV/table view whenever you want the literal table your prompt asks for.

**Cost:** ~1 day of setup, zero ongoing manual bookkeeping afterward. This is the single highest-leverage item in this whole document relative to effort.

---

## B1. Bootstrap with real, licensed Indic parallel corpora

**What it is:** Instead of building a data-collection pipeline from zero, start from AI4Bharat's released resources:
- **Samanantar** — 49.7M English↔11-Indic-language sentence pairs (Hindi included), CC0/research-friendly.
- **BPCC (Bharat Parallel Corpus Collection)** — ~230M bitext pairs across all 22 scheduled languages, built specifically to fix the sparsity/quality problems your Section 6 describes; released under MIT (code/models), CC-BY-4.0 (human-translated portion), CC0 (mined portion) — no commercial-use restriction.
- **IN22 (IN22-Gen, IN22-Conv)** — the held-out, n-way parallel benchmark AI4Bharat built specifically so Indic MT systems have a benchmark that isn't leaked into training data anywhere. Use this as your true test set, never touch it during training.
- **FLORES-200 devtest** — the standard multilingual benchmark; lets you compare your numbers against published NLLB-200/IndicTrans2 numbers on an apples-to-apples set.

**Why it matters for English→Hindi:** These corpora already solve most of Section 6–8's leakage/licensing/quality-tracking headaches for you, and they let you compare against a real external baseline (see B12) using the exact same test set the field uses.

**Implementation roadmap:**
1. Download Samanantar and BPCC from the AI4Bharat website / HuggingFace (`ai4bharat/samanantar`, `ai4bharat/BPCC` — check current HF repo listings, they update).
2. Filter to the `en-hi` pair only for your bilingual baseline; keep the rest aside for Part-A-Section-16 multilingual experiments later.
3. Reserve IN22-Gen/IN22-Conv and FLORES-200 `devtest` as frozen, never-trained-on test sets from day one — write this into your data-splitting code as a hard assertion, not a convention you might forget.
4. Log dataset provenance (source, license, pair count, date pulled) in a `data/DATASHEET.md` per Section 6's own requirement.

**Evaluation:** Report train/val size, dedup rate against IN22/FLORES (to catch accidental contamination — near-duplicate check, see B2), and license per source in your first experiment log entry.

---

## B2. Real data-quality filtering: LaBSE similarity + CometKiwi, not just length ratio

**What it is:** Two concrete, currently-used filters beyond "length ratio looks OK":
1. **LaBSE-based filtering** — embed source and target with LaBSE (a multilingual sentence encoder), keep pairs above a cosine-similarity margin threshold (BPCC itself was filtered this way; NLLB used a similar margin-based mining threshold around 1.06). This catches misaligned/garbage pairs that pass length filters but aren't actually translations of each other.
2. **CometKiwi as a reference-free quality filter** — CometKiwi scores a `(source, hypothesis)` pair for quality *without needing a reference*, so it can score your raw mined/scraped pairs directly and flag low-quality ones for removal or down-weighting.

**Why it matters for English→Hindi:** Web-mined Hindi-English data is full of near-duplicates, partial translations, and transliteration noise that pass naive filters. This is exactly the "Quality Estimation" box in your Section 6 pipeline diagram, made concrete.

**Implementation roadmap:**
1. `pip install sentence-transformers` and load a LaBSE checkpoint (`sentence-transformers/LaBSE`) — this is a *pipeline* dependency, not your core tokenizer, so it's fine outside your stdlib constraint.
2. Embed both sides of every candidate pair, compute cosine similarity, and either use a hard threshold or a margin score (similarity to nearest neighbors, not just the paired sentence) to reject likely-misaligned pairs.
3. Separately, `pip install unbabel-comet` and run `comet-score` with a CometKiwi checkpoint (e.g. `Unbabel/wmt22-cometkiwi-da`) over your candidate pairs to get a reference-free quality score per pair; set a percentile cutoff (e.g. drop bottom 10%) as your first experiment, then treat the cutoff itself as a tunable hyperparameter (Section 7's "treat filtering thresholds as experimental parameters").
4. Log how many pairs each filter removes and re-run your baseline with vs. without each filter — this is a direct Section-28-style ablation (Baseline, Baseline+LaBSE, Baseline+CometKiwi, Baseline+both).

**Evaluation:** SacreBLEU/chrF/COMET on IN22, plus report the retained-pair count so you can see the quality/quantity trade-off explicitly.

---

## B3. Devanagari-aware normalization and script unification — reimplementable in pure Python

**What it is:** The concrete algorithm behind AI4Bharat's `indic_nlp_library` normalization and script-unification steps, so you can hand-roll it inside your stdlib-only tokenizer instead of importing the library:
- **Normalization:** NFC-normalize, decompose nukta-bearing composite characters into base + nukta, normalize the `।`/`|` (danda) and `:` (colon→visarga) punctuation conventions, and optionally normalize nasal variants (anusvara vs. candrabindu vs. full nasal consonant).
- **Script unification:** All major Brahmic scripts (Devanagari, Bengali, Gujarati, Gurmukhi, etc.) allocate their Unicode codepoints at *consistent relative offsets* from a per-script base codepoint. That means transliterating character-by-character between two Brahmic scripts is literally `chr(target_base + (ord(c) - source_base))` for characters in the shared coordinated range — no lookup table needed for the common case.

**Why it matters for English→Hindi:** Right now your tokenizer's NFC-normalization step is doing part of this already. Nukta decomposition and danda/visarga normalization are the two most common sources of "same word, different byte sequence" in real Hindi web text, and they're a handful of `unicodedata`-based rules, not a dependency.

**Implementation roadmap:**
1. Write a `devanagari_normalize(text: str) -> str` function using only `unicodedata.normalize("NFC", text)` plus an explicit dict of nukta-composite → (base, nukta) decompositions (there are only a handful of nukta-bearing Devanagari characters — this is a ~20-entry table you can hand-transcribe from the Unicode Devanagari block chart).
2. Add danda (`।` U+0964) and double-danda (`॥` U+0965) as sentence-boundary-aware punctuation tokens, not generic punctuation, since they behave like a full stop.
3. Treat the colon-as-visarga convention as an explicit, optional normalization flag (some corpora use `:` typed on a QWERTY layout to mean `ः`) — log how often it fires so you know if it's even relevant for your data source.
4. **Do not** implement full multi-script transliteration unless you extend to Bengali/Gujarati/etc. in Section 16 — for pure English↔Hindi this is out of scope, but write the note in your code so future-you knows where the multilingual hook would go.
5. Optionally cross-check your hand-rolled normalizer's output against `indic_nlp_library`'s `DevanagariNormalizer` on a sample as a correctness test (import the library only in a test script, never in your core pipeline).

**Evaluation:** Before/after normalization, measure vocabulary size reduction and UNK-rate change on your held-out set — this is literally the fertility/compression-ratio benchmark you already planned in your tokenizer notes.

---

## B4. LLM-based sequence-level knowledge distillation as a synthetic-data source

**What it is:** Sequence-level Knowledge Distillation (SeqKD, Kim & Rush 2016): take monolingual sentences, translate them with a strong "teacher" model, and train your student Transformer on the resulting synthetic pairs — the same mechanism as back-translation, but with a general-purpose LLM as the teacher instead of a same-architecture MT model translating in the reverse direction. This exact recipe has been demonstrated for low-resource MT using Claude as the teacher (translating monolingual news text, distilling into a small fine-tuned student).

**Why it matters for English→Hindi:** A strong LLM teacher can produce fluent, idiomatic Hindi (handling honorifics, register, and idiom far better than a weak from-scratch model would at the same stage), giving your student a training signal that's qualitatively different from — and complementary to — classic back-translation (Section 13).

**Implementation roadmap:**
1. Collect monolingual English and Hindi text (news, Wikipedia — reuse whatever you gather for Section 13's back-translation monolingual data).
2. Use an LLM API (e.g. the Claude API) to translate a batch of monolingual sentences into the other language with a simple, consistent system prompt (no elaborate few-shot needed for a strong general model) — this produces `(monolingual, LLM-translation)` synthetic pairs, exactly mirroring Section 13's back-translation diagram but with an LLM standing in for the reverse-direction MT model.
3. Mix these synthetic pairs into training at a controlled ratio alongside real BPCC/Samanantar data — treat the mixing ratio as an experimental variable exactly like Section 13's `real:synthetic` sweep (100:0 → 25:75).
4. **Filter before training**, don't just trust the teacher: run the LaBSE/CometKiwi filter from B2 on the synthetic pairs too, since LLM translations can be fluent but occasionally unfaithful (add extra content, paraphrase instead of translate, or mistranslate rare entities).
5. **Known failure mode to watch for:** recent work (Dankers & Raunak, 2025) shows SeqKD students inherit more memorization and higher hallucination rates than models trained on real data of the same size — actively monitor for verbatim-copied training sentences and hallucinated content in your student's outputs, and consider capping how much of your training mix is LLM-distilled data for this reason (an "Adaptive-SeqKD" style mitigation: down-weight low-quality/high-memorization-risk synthetic examples rather than using them uniformly).

**Evaluation:** Run your Section-12 error taxonomy separately on `(real-data-only model)` vs `(real+distilled model)` outputs — check specifically for hallucination and unsupported-content insertion, not just BLEU delta.

---

## B5. Filtering synthetic/back-translated data before it hits training

**What it is:** The same LaBSE/CometKiwi filters from B2, applied specifically to back-translated and LLM-distilled pairs (B4/Section 13), since synthetic data has different failure modes than mined data (fluent-but-wrong, rather than misaligned).

**Implementation roadmap:**
1. After generating back-translated or distilled pairs, score them with CometKiwi exactly as in B2.
2. Compare the score distribution of synthetic pairs against your real-data distribution — synthetic pairs that score *unusually* high can indicate the teacher/back-translation model is producing overly literal or templated output (a known MBR/QE bias, see B9), so don't just keep the top-scoring synthetic pairs blindly; sample across the distribution or cap per-sentence-length buckets.
3. Track quality-weighted training: instead of a hard keep/reject cutoff, use the CometKiwi score as a per-example loss weight (higher-quality synthetic pairs contribute more to the gradient) — cheap to implement (multiply the per-token loss by the sentence weight before backprop) and avoids throwing away marginal-but-useful data.

**Evaluation:** Ablate hard-filter vs. quality-weighted vs. unfiltered synthetic data, same as any other Section-28 ablation.

---

## B6. Deep Encoder, Shallow Decoder — a proven, cheap architecture ablation

**What it is:** Kasai et al. (2020) showed that, given a sufficiently deep encoder, a *one-layer* autoregressive decoder matches full-depth-decoder translation quality while cutting decoding latency substantially — because the encoder is fully parallelizable but the decoder runs token-by-token, so decoder depth is where latency actually lives. (Follow-up work found this holds cleanly for many-to-one multilingual translation and needs more care for one-to-many; your project is bilingual English→Hindi, which is the setting where it works cleanly.)

**Why it matters for English→Hindi:** This is a config change (e.g. 12-layer encoder / 2-layer decoder instead of 6/6), not a new component to implement — it's one of the highest quality-per-effort experiments available once your baseline works, and it directly serves your Section 25 (inference engineering) goals.

**Implementation roadmap:**
1. After your baseline Transformer trains and evaluates correctly, add `n_encoder_layers` and `n_decoder_layers` as independent Hydra config fields (you likely already have one shared `n_layers`).
2. Run the ablation table: baseline (e.g. 6/6), deep-shallow (e.g. 12/2, 12/1), matched-parameter-count control (so a quality gain isn't just "more total parameters").
3. Measure wall-clock decoding latency per sentence at fixed hardware, not just BLEU/chrF/COMET — the entire point of this technique is the speed/quality trade-off curve, so report both axes.

**Evaluation:** BLEU/chrF/COMET on IN22 *and* tokens/sec or ms/sentence at inference — plot quality vs. latency, per Section 25's explicit instruction to not treat quality as the only metric.

---

## B7. R-Drop — cheap, effective consistency regularization

**What it is:** Run each training example through the model *twice* (dropout makes the two forward passes slightly different), and add a KL-divergence term between the two output distributions to the loss, on top of normal cross-entropy. Reported to give SOTA-level gains on WMT14 En-De/En-Fr with a vanilla Transformer, at the cost of one extra forward pass per step.

**Why it matters for English→Hindi:** It's a training-loop-only change (no architecture or data changes needed), so it's an easy, low-risk first entry in your Section 9 "training objective" experiments, and it's specifically validated on both low-resource and rich-resource MT settings.

**Implementation roadmap:**
1. In your training step, run the batch through the model twice (two independent dropout masks — just call the model twice with `model.train()` active both times).
2. Compute standard cross-entropy on both passes' outputs, plus a bidirectional KL term: `biKL(p1, p2) = 0.5 * (KL(p1||p2) + KL(p2||p1))` over the token-level output distributions.
3. Total loss = `CE1 + CE2 + α * biKL`, with `α` (paper default ≈ 4–5 for MT tasks, but treat it as a tunable hyperparameter) as a new Hydra config field.
4. This roughly doubles per-step compute (two forward passes) — budget accordingly, or apply it only after a few epochs of normal training if compute-constrained.

**Evaluation:** Ablate with/without R-Drop at fixed training budget (steps or wall-clock, your choice, but state which) — report BLEU/chrF/COMET plus paired bootstrap significance (B11), since the claimed gains are often a small number of BLEU points.

---

## B8. kNN-MT — retrieval-augmented decoding for domain adaptation without retraining

**What it is:** Augment a *frozen, already-trained* NMT model with a nearest-neighbor datastore: cache `(decoder hidden state, next target token)` pairs from a reference corpus, and at inference time interpolate the model's predicted next-token distribution with a distribution derived from the k nearest cached hidden states. Original paper reported +9.2 BLEU average on out-of-domain test sets (Medical, Law, IT, Koran, Subtitles) versus the base model, without any retraining — by building a domain-specific datastore from as little as a few hundred thousand in-domain sentence pairs.

**Why it matters for English→Hindi:** This is the concrete implementation of Section 23's "retrieval augmentation if justified." If you want your model to handle a specific domain (say, financial or legal Hindi text) without a full fine-tuning run, this is the standard, well-verified way to do it — and it's also useful as a way to squeeze extra quality out of a small from-scratch model using a modest amount of extra bitext, addressing your own research question ("can a smaller model achieve comparable quality through better data selection... and distillation").

**Implementation roadmap:**
1. Build the datastore: run your trained model in teacher-forcing mode over a reference parallel corpus, and cache the decoder's final-layer hidden state at each position as the *key*, with the actual next target token as the *value*. Use `faiss` (Facebook's approximate-nearest-neighbor library) to index the keys for fast lookup — `pip install faiss-cpu` (or `faiss-gpu` if available).
2. At inference, at each decoding step: get the model's normal next-token distribution *and* query the datastore for the k nearest hidden states, convert their distances to a softmax-weighted distribution over the corresponding target tokens, and linearly interpolate the two distributions with a tunable weight λ.
3. Start with an in-domain datastore built from a domain you actually care about (or, as a first sanity-check experiment, build it from your own training data and confirm it at least doesn't hurt in-domain performance).
4. This is inference-time only — no gradient updates, no retraining — so it's cheap to try as soon as you have any trained checkpoint.

**Evaluation:** Compare base-model-only vs. base-model+kNN on an out-of-domain test set you construct or borrow (e.g. a different-domain slice of your data held out specifically for this), reporting BLEU/chrF/COMET and decoding latency (kNN retrieval adds real per-token latency — report it honestly per Section 25).

---

## B9. Minimum Bayes Risk (MBR) decoding — an alternative to beam search

**What it is:** Instead of decoding the single highest-probability sequence (beam search's goal), MBR decoding samples a pool of candidate translations from the model, then picks the candidate that has the highest expected similarity (under some utility metric, e.g. chrF or a neural metric) to the *other* candidates in the pool — directly targeting translation quality rather than model probability. Research has repeatedly shown model probability and human-judged quality only loosely correlate, which is exactly why beam search has known failure modes (too-short outputs, overgenerating frequent words, poor robustness to copy noise/domain shift) that MBR is more robust to.

**Why it matters for English→Hindi:** This is a decoding-time-only change — no retraining — and it's a real ablation for Section 11 beyond "greedy vs. beam vs. sampling." It's also specifically useful for your project because Section 24 (robustness) flags copy-noise and domain-shift sensitivity as things to test, and MBR is empirically more robust to exactly those than beam search.

**Implementation roadmap:**
1. Sample N candidate translations (e.g. N=16–64) from your trained model using temperature or nucleus sampling (not beam search — you need *diverse* samples for MBR to work).
2. Choose a utility function: start with chrF (cheap, no extra model needed) between every pair of candidates; a stronger but more expensive option is a neural metric like CometKiwi (B10) as the utility, which the "high quality rather than high model probability" line of research found gives even better results than surface metrics.
3. For each candidate, compute its average utility against all other candidates in the pool; pick the candidate with the highest average utility as your final output.
4. This is O(N²) utility computations per sentence — expensive relative to beam search. Budget for it as a "final model evaluation" decoding strategy, not necessarily something you run during every training-time validation pass.
5. Optional advanced variant: **source-based MBR (sMBR)**, which uses paraphrases/back-translations of the *source* sentence as support hypotheses combined with a reference-free QE utility function — worth trying once basic MBR is working, not as a first step.

**Evaluation:** Compare beam search vs. sampling vs. MBR on IN22, specifically checking output-length distribution (beam search's known short-output bias) and performance on your robustness test set (Section 24) — MBR's value proposition is exactly in these robustness dimensions, not necessarily raw BLEU.

---

## B10. The actual current evaluation-metric family: beyond BLEU/chrF/"COMET"

**What it is:** Your Section 12 says "COMET" generically. In 2026, the field distinguishes several specific, non-interchangeable metrics you should know by name:
- **COMET-22** — reference-based, ensembles a regression model with a span-tagging (error-detection) model; the current standard reference-based neural metric.
- **CometKiwi** — reference-*free* (quality estimation / QE): scores `(source, hypothesis)` with no reference needed, useful for filtering data (B2) or as an MBR utility (B9) where you don't have ground truth.
- **XCOMET** — an explainable variant that also flags *where* errors occur, not just an overall score.
- **MetricX-24** — Google's current WMT-metrics-shared-task entry, a hybrid model that can score with source-only, reference-only, or both.
- **Indic-COMET** — COMET fine-tuned specifically on Indian-language MQM judgment data; shown to correlate with human judgment on Indic languages *better* than generic COMET, which matters because generic COMET's correlation with human judgment varies noticeably by language pair.

**Why it matters for English→Hindi:** Generic COMET was trained mostly on high-resource European-language judgment data; using a metric verified specifically on Indic languages (Indic-COMET) or at minimum a current-generation metric (COMET-22/XCOMET/MetricX-24 rather than an old COMET checkpoint) gives you evaluation numbers you can actually trust for Hindi specifically, per your own Section 32/33's "verify metric, language pair, test set" discipline.

**Implementation roadmap:**
1. `pip install unbabel-comet` for COMET-22/CometKiwi/XCOMET; check the `google-research/metricx` GitHub repo for MetricX-24 (separate package).
2. Standardize on **CometKiwi** wherever you need a reference-free score (data filtering, MBR utility) and **COMET-22 or XCOMET** wherever you have a reference (final held-out evaluation on IN22).
3. If Indic-COMET checkpoints are publicly released (check the IndicMT-Eval paper's associated repo), swap it in for your headline Hindi evaluation number specifically, and report both generic-COMET and Indic-COMET side by side so you can see whether they diverge.
4. Keep BLEU and chrF too — not as a downgrade, but because your own Section 12 wants an error taxonomy and multiple metrics that disagree on a given output are themselves informative (e.g. high chrF but low COMET often flags a fluent-but-unfaithful translation).

**Evaluation:** Report BLEU, chrF, COMET-22, CometKiwi, and (if available) Indic-COMET together in your Section 27 experiment table, not just one number — and note explicitly which test set (IN22-Gen? IN22-Conv? FLORES devtest?) each score is on, per Section 33.

---

## B11. Statistical significance testing — make "Model B is better" an actual claim

**What it is:** Paired bootstrap resampling (Koehn, 2004): repeatedly resample your test set with replacement, recompute the metric on each resample for both systems, and see how often System A beats System B — giving you a p-value and confidence interval instead of a single point estimate. This is exactly what Section 41 asks ("is the improvement statistically or practically meaningful?") but the original prompt never says *how* to check it.

**Why it matters for English→Hindi:** With Section 28's ablation tables producing many small BLEU/chrF deltas, some of those differences will be noise. Without significance testing you risk drawing conclusions ("architecture change X helped") from what's actually random variation in a moderate-sized test set.

**Implementation roadmap:**
1. `pip install sacrebleu` (you likely already use this for BLEU/chrF).
2. After generating two systems' outputs on the same test set (e.g. IN22), run: `sacrebleu ref.txt -i baseline_hyp.txt new_hyp.txt -m bleu chrf --paired-bs` — sacrebleu will bootstrap-resample and report a p-value comparing every listed system against the first one (your baseline).
3. Use `--paired-bs-n 1000` (or higher, e.g. 10000, for a final headline result) to control resample count; report it in your experiment log signature so results are reproducible.
4. For COMET-family metrics (which are averages of per-sentence scores rather than a corpus-level score computed from raw n-gram counts), a paired t-test or the same bootstrap-resampling logic applied manually to the per-sentence COMET scores is the standard approach — `unbabel-comet`'s `comet-compare` utility supports this directly.
5. Adopt a fixed significance threshold (p < 0.05 is the field standard) *before* running the test, and write "not statistically significant at p<0.05" into your experiment log exactly as often as "significant" — this is a discipline thing, not just a formality.

**Evaluation:** This *is* the evaluation methodology upgrade — apply it to every pairwise comparison in your Section 27 table from here on, not just headline results.

---

## B12. A fixed external reference point: IndicTrans2 / NLLB-200, and an optional fine-tuned-baseline arm

**What it is:** IndicTrans2 (AI4Bharat) and NLLB-200 (Meta) are the two most relevant existing strong models for English↔Hindi — open-source, MIT/CC-licensed, with published IN22/FLORES benchmark numbers. Running them on your exact test set gives you a real, current "how far are we from a strong existing system" number (your Section 38's "Is this SOTA?" question needs a real comparison point, not just a vibe).

**Why it matters for English→Hindi:** This also gives you a natural second experimental arm for your own research question #6 ("can a smaller model achieve comparable quality through better data selection and distillation?") — compare your from-scratch model against a LoRA-fine-tuned IndicTrans2/NLLB-200 of similar *inference* cost, which tells you directly how much "from scratch, well-engineered" buys you versus "start from a strong pretrained model, adapt cheaply."

**Implementation roadmap:**
1. Download IndicTrans2's `en-indic` checkpoint (HuggingFace: `ai4bharat/indictrans2-en-indic-1B` or the distilled 200M variant) and/or NLLB-200's distilled checkpoint.
2. Run both on your frozen IN22/FLORES test set (never used for your own training, per B1) using their published inference scripts (`IndicTransToolkit` for IndicTrans2), and score with the same metric stack as B10 — this is your **reference ceiling**, not a target you need to beat.
3. Optional second arm: LoRA-fine-tune IndicTrans2 or NLLB-200 on your own filtered BPCC/Samanantar subset (`peft` library, `pip install peft`) as a distinct experimental condition, comparable in adaptation cost to training your from-scratch model, but starting from a pretrained multilingual checkpoint instead of random initialization.
4. Report this in your Section 38 "Is this SOTA?" breakdown as a concrete row, not an abstract discussion.

**Evaluation:** Same metric stack (B10) and same test set (IN22/FLORES) across your from-scratch model, the pretrained-off-the-shelf models, and (optionally) the LoRA-fine-tuned arm — a single comparable table.

---

## B13. Curriculum learning, concretized

**What it is:** Your Section 9 mentions curriculum learning generically. A concrete, implementable version for MT: order training examples from "easy" to "hard" over the course of training rather than fully random shuffling, where difficulty is operationalized as sentence length, target-side token rarity (average or max negative log-frequency of target tokens), or — specific to your tokenizer — Akshara-sequence complexity (number of conjunct clusters per sentence, which correlates with morphological complexity in Hindi).

**Implementation roadmap:**
1. Precompute a per-sentence-pair difficulty score after your data pipeline (B1/B2) is in place: start with `len(source_tokens) + len(target_tokens)` as the simplest signal.
2. Add a rarity signal: compute unigram frequency of each target token over your training corpus, score each sentence by its average `-log(freq)` — rarer target vocabulary = harder.
3. Bucket sentences into difficulty deciles; for the first N% of training steps, sample only from the easiest 2–3 deciles, then linearly anneal to full-corpus random sampling by some fraction of total training (a common schedule: fully curriculum for the first 20–30% of steps, fully random after).
4. Treat "curriculum vs. no curriculum" and "which difficulty signal" as Section-28-style ablations — this is cheap to implement (a sort + a scheduled sampler) but easy to get wrong (e.g. too-aggressive curricula can hurt final quality), so it needs its own controlled comparison, not just a "seems reasonable" adoption.

**Evaluation:** Compare final BLEU/chrF/COMET at fixed total steps, curriculum vs. random order, plus the training-loss curve shape (curricula often show faster early-training loss descent — check whether that translates into better final quality or just faster early progress).

---

## B14. Named-entity, number, and URL handling — the concrete first robustness fix

**What it is:** Before reaching for exotic robustness techniques, the single highest-yield fix for a lot of real-world MT robustness failures (Section 24) is explicit handling of numbers, URLs, and named entities: they're often mistranslated, dropped, or hallucinated because a small from-scratch model hasn't seen enough examples to learn "just copy this" reliably.

**Implementation roadmap:**
1. Add a preprocessing pass that detects numbers (regex on digit sequences, including Devanagari digits `०-९`), URLs (regex), and (optionally, more advanced) named entities (a simple gazetteer lookup, or a lightweight NER tagger if you want to go further) in the source sentence.
2. Replace detected spans with placeholder tokens (`⟨NUM_0⟩`, `⟨URL_0⟩`) before tokenization, train the model to translate the placeholder-substituted sentence, and re-substitute the original spans back into the model's output post-decoding — a simple, well-established "copy mechanism via placeholder substitution" approach.
3. Special-case Devanagari numerals (`०१२३४५६७८९`) vs. Western numerals (`0123456789`) — Hindi text mixes both conventions, and your placeholder detection needs to catch both.
4. This interacts with your tokenizer directly: decide whether placeholders are single Akshara-BPE tokens (recommended — treat them as reserved vocabulary entries, similar to how you'd handle `<unk>`/`<pad>`).

**Evaluation:** Build a small targeted test set of sentences containing numbers/URLs/names (can be synthetic — template sentences with different entities substituted in) and measure exact-match preservation rate before/after this change — this is a much more sensitive test than aggregate BLEU for this specific failure mode.

---

## B15. A tokenizer-free ablation arm: byte-level baseline

**What it is:** A control condition for your Akshara-BPE tokenizer research: a byte-level model (256-symbol base vocabulary, à la ByT5, ByteNet, or byte-level BPE with no subword merges at all) that requires no tokenizer engineering whatsoever, as the "what if we didn't need a clever tokenizer" baseline.

**Why it matters for English→Hindi:** Your own tokenizer notes already plan to benchmark Akshara-BPE against SentencePiece BPE and Unigram. A byte-level baseline is a genuinely different point of comparison — it directly tests the hypothesis your whole tokenizer project is built on ("script-aware presegmentation helps") against the null hypothesis ("a big enough model doesn't need it"). Indic MT literature includes at least one system built on exactly this idea (byte-level BPE with a 256-symbol base vocabulary for Odia, in a WAT shared task submission), so it's a real, not hypothetical, comparison point.

**Implementation roadmap:**
1. Implement (or reuse via stdlib `str.encode("utf-8")`) a trivial byte-level "tokenizer": every input sentence becomes a sequence of raw UTF-8 byte values (0–255), no merges, no vocabulary learning needed at all.
2. Train the *same* Transformer architecture and training recipe on byte-sequences instead of Akshara-BPE tokens — expect much longer sequences (bytes are the finest possible granularity), so you'll likely need to reduce batch size or increase gradient accumulation to keep effective batch size constant, and this is a real compute-cost data point worth recording (Section 29).
3. This is explicitly *not* meant to beat your Akshara-BPE tokenizer — it's meant to quantify the gap, and a byte-level model needing dramatically more training steps/compute to reach the same quality *is* a positive result for your tokenizer research, not a failure.

**Evaluation:** Compare final quality (BLEU/chrF/COMET) at matched training compute (not matched steps, since sequence lengths differ) against your Akshara-BPE model, plus sequence-length distribution and training wall-clock time — exactly the fertility/compression-ratio framing you already planned, extended to include a zero-engineering baseline.

---

## B16. Quantization and distillation for deployment

**What it is:** Two standard, complementary size/speed reduction techniques for a trained model, relevant once you have a checkpoint you're happy with and want to discuss inference (Section 25): **post-training quantization** (reduce weight precision, e.g. INT8 or 4-bit via `bitsandbytes`) and **model distillation** (train a smaller student to mimic your larger trained model's outputs — note this is a *different* use of "distillation" than B4/Section 13's data-generation use; here the teacher and student are both your own models, and the goal is compression, not synthetic data).

**Implementation roadmap:**
1. `pip install bitsandbytes` and load your trained model with 8-bit or 4-bit weight quantization (`load_in_8bit=True` equivalent for your custom architecture, or export to ONNX and quantize with `onnxruntime`'s quantization tools if you want a framework-independent deployment artifact).
2. Measure the quality delta (BLEU/chrF/COMET) before vs. after quantization on IN22 — quantization is usually near-lossless for translation quality at INT8, more noticeably lossy at 4-bit, so this needs to be measured, not assumed.
3. For distillation-for-compression: apply the same SeqKD mechanism as B4, but with *your own trained model* as the teacher and a smaller architecture (fewer layers/smaller hidden size) as the student — this is a distinct experiment from B4, worth its own row in your experiment table.
4. Track model size on disk, peak inference memory, and latency alongside quality for every variant — Section 25's explicit ask.

**Evaluation:** A quality-vs-size-vs-latency table across full-precision, quantized, and distilled variants — this is the concrete deliverable Section 25 is asking for.

---

## B17. Speculative decoding for inference speed

**What it is:** Use a small, fast "draft" model to propose several tokens ahead, then have your main model verify/accept them in a single forward pass (rejecting and falling back to normal decoding wherever the draft model's guess diverges) — reduces the number of expensive main-model forward passes needed per output sentence.

**Why it matters here:** It's complementary to B6 (deep encoder/shallow decoder) and B16 (quantization) as an inference-speed lever that doesn't touch model quality at all (it's mathematically guaranteed to produce the same output distribution as normal decoding, unlike quantization or distillation which do trade off quality).

**Implementation roadmap:**
1. This needs a second, smaller model as the "draft" model — a natural choice is your B15 byte-level baseline (if it trains fast) or a deliberately shrunk version of your own architecture (fewer layers), since it needs to be meaningfully cheaper per forward pass than your main model.
2. Implement the accept/reject loop: draft model proposes k tokens autoregressively, main model scores all k+1 positions in one forward pass, accept the longest prefix where main-model and draft-model agree (using the standard speculative-decoding rejection-sampling rule so the output distribution exactly matches normal decoding), fall back to normal single-token decoding from the first rejection point.
3. This is meaningfully more implementation effort than B6/B16 — treat it as a later-phase (Phase 16+, per Part C.3) addition once you have both a main and a viable draft model already trained.

**Evaluation:** Tokens/sec and end-to-end latency vs. standard autoregressive decoding, at *identical* output quality (verify this — the whole point of speculative decoding is that quality shouldn't change; if it does, something's implemented wrong).

---

## B18. Automating the error taxonomy with linguistic tooling

**What it is:** Your Section 12 error taxonomy (gender, number, case, agreement, tense/aspect, word order, etc.) is currently something you'd apply by hand, sentence by sentence. A lightweight automation: use a Hindi POS-tagger/morphological analyzer to bucket errors programmatically at scale.

**Implementation roadmap:**
1. Use `indic_nlp_library`'s or Stanza's Hindi POS/morphology models (`pip install stanza`, `stanza.download("hi")`) — again, a pipeline/evaluation-tooling dependency, not part of your core tokenizer.
2. For each `(reference, hypothesis)` pair in your evaluation set, run morphological analysis on both; flag mismatches in gender/number/case markers on aligned tokens (you'll need a simple word-alignment step — `awesome-align` or even a cheap edit-distance-based alignment is enough for a first pass) as candidate agreement errors.
3. Aggregate flagged-error counts per category across your whole test set, producing a quantitative version of the qualitative taxonomy your prompt describes — e.g. "gender-agreement mismatches: 4.2% of sentences" instead of a handful of hand-picked examples.
4. Use this to directly drive Section 34's error-driven research loop: if gender-agreement errors cluster around a specific rare noun class, that's your next targeted experiment, found automatically rather than by manual inspection.

**Evaluation:** This tooling *is* the evaluation upgrade for Section 12 — validate it against a small hand-labeled sample first (does the automated tagger agree with your own manual error judgment on, say, 50 sentences?) before trusting it at scale.

---

## B19. Multilingual data mixing via language-family clustering (a concrete refinement of Section 16)

**What it is:** When you eventually add more Indic languages (Section 16), don't add them uniformly — cluster by language family (Indo-Aryan: Hindi, Bengali, Marathi, Gujarati, Punjabi vs. Dravidian: Tamil, Telugu, Kannada, Malayalam) and expect much stronger positive transfer within a family than across families, since NLLB-200 and IndicTrans2's own analyses (and general multilingual-MT literature) consistently find shared-script/shared-family languages transfer better than typologically distant ones.

**Implementation roadmap:**
1. When you run Section 16's "does each added language help or hurt English→Hindi" experiment, structure it as: add one Indo-Aryan language at a time first (e.g. Bengali, Marathi), measure the English→Hindi delta, *then* try one Dravidian language, measure again — this directly tests the family-clustering hypothesis rather than treating "multilingual" as one undifferentiated condition.
2. Combine with B3's script-unification note: if you add another Devanagari-script language (Marathi, Nepali), you get shared-script transfer for free at the character level; if you add a different-script Indo-Aryan language (Bengali, Gujarati), script unification (B3) becomes directly relevant to whether shared-vocabulary multilingual tokenization helps.

**Evaluation:** English→Hindi BLEU/chrF/COMET as a function of which language(s) were added, grouped by family — a direct, small extension of your existing Section 16 experiment design.

---

# PART C — Quick Reference

## C.1 Datasets & Benchmarks

| Resource | Size | Use as | License |
|---|---|---|---|
| Samanantar | 49.7M en↔11-Indic pairs | Bilingual training data | CC0-friendly |
| BPCC (incl. BPCC-BT) | ~230M pairs, 22 languages | Bilingual + multilingual training data | MIT (code/models) / CC-BY-4.0 (human) / CC0 (mined) |
| IN22-Gen, IN22-Conv | n-way parallel, 22 languages | **Frozen test set** — never train on this | Released with IndicTrans2 |
| FLORES-200 devtest | Multilingual, 200 languages | **Frozen test set**, cross-paper comparability | Open |
| IndicCorp (monolingual) | Large monolingual Indic corpora | Back-translation (Section 13) source | Released by AI4Bharat |

## C.2 Tooling Stack (pipeline/eval only — never your core tokenizer)

```bash
# Experiment infra
pip install hydra-core wandb dvc --break-system-packages

# Data quality filtering (B2)
pip install sentence-transformers unbabel-comet --break-system-packages

# Evaluation (B10, B11)
pip install sacrebleu unbabel-comet --break-system-packages
# MetricX-24: see github.com/google-research/metricx

# Retrieval-augmented decoding (B8)
pip install faiss-cpu --break-system-packages

# Reference baselines / LoRA fine-tuning (B12)
pip install peft transformers --break-system-packages
# IndicTrans2 inference: github.com/AI4Bharat/IndicTransToolkit

# Error-taxonomy tooling (B18)
pip install stanza --break-system-packages

# Deployment (B16)
pip install bitsandbytes onnxruntime --break-system-packages
```

## C.3 Updated Phase Roadmap (supersedes Section 42)

```
PHASE 0  Infra & reproducibility bootstrap ............ B0 (Hydra + W&B + DVC)
PHASE 1  Transformer fundamentals
PHASE 2  Tokenizer ..................................... B3 (normalization), B15 (byte-level control arm)
PHASE 3  Minimal Transformer
PHASE 4  Baseline training .............................. B1 (real corpora), B6 (deep/shallow ablation)
PHASE 5  Evaluation framework ........................... B10 (metric family), B11 (significance testing)
PHASE 6  Data cleaning ................................... B2 (LaBSE / CometKiwi filtering)
PHASE 7  Training optimization ........................... B7 (R-Drop), B13 (curriculum learning)
PHASE 8  Decoding ......................................... B9 (MBR / sMBR)
PHASE 9  Back-translation ................................ B4 (LLM-distillation as a parallel source), B5 (synthetic filtering)
PHASE 10 Synthetic-data filtering ......................... B5
PHASE 11 Multilingual transfer ............................ B19 (family clustering), B3 (script unification)
PHASE 12 Pretraining / external baselines ................. B12 (IndicTrans2 / NLLB-200 reference + optional LoRA arm)
PHASE 13 Modern Transformer improvements .................. (Section 18, unchanged)
PHASE 14 Scaling
PHASE 15 MoE / sparse models where justified
PHASE 16 Distillation / compression ....................... B16 (quantization + distillation), B17 (speculative decoding)
PHASE 17 Robustness + domain adaptation ................... B14 (entity/number handling), B8 (kNN-MT domain adaptation)
PHASE 18 Large-scale benchmarking
PHASE 19 Ablation studies .................................. B18 (automated error taxonomy)
PHASE 20 Research contribution
```

## C.4 Additional Research Questions (extends Section 39)

- Does a deep-encoder/shallow-decoder (B6) architecture change *which* error categories (Section 12's taxonomy) appear, not just the aggregate BLEU/chrF/COMET score?
- Does kNN-MT domain adaptation (B8) outperform full fine-tuning for code-switched or informal Hindi, where labeled in-domain training data is scarce but a moderate reference corpus is easy to collect?
- How much of the quality gap to IndicTrans2/NLLB-200 (B12) is closed by data selection and LLM-distilled synthetic data (B2 + B4) alone, versus requiring architecture or scale changes?

---

*End of document. This is planning and reference material only — nothing described above has been implemented.*
