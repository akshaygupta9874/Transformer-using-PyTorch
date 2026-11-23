# Transformer using PyTorch

## Training performance

On the NVIDIA GeForce RTX 5060 Laptop GPU, an epoch improved from approximately **2 hours to 8 minutes** after the training path was optimized. This is about a **15x speedup**.

The main reason is that the original training loop was not making efficient use of the GPU. It used full FP32 activations, manually materialized the full attention score matrix, processed data with a single loader worker, and flushed TensorBoard logs on every batch.

## What changed

### 1. BF16 mixed precision

Training now uses CUDA BF16 autocasting:

- Activations and most matrix operations use BF16.
- The RTX 5060 supports BF16 natively.
- This reduces memory traffic and makes tensor operations much faster.
- The model and optimizer remain numerically stable without requiring a gradient scaler.

Configured in `config.py`:

```python
"use_amp": True,
"amp_dtype": "bfloat16",
```

### 2. Memory-efficient scaled dot-product attention

The original attention implementation explicitly created attention scores with matrix multiplication, masking, softmax, and another matrix multiplication. The model now uses PyTorch's optimized `scaled_dot_product_attention` implementation.

On CUDA, PyTorch can select a fused and memory-efficient attention kernel. This reduces both VRAM use and attention runtime. Integer masks from the dataset are converted to boolean masks before the attention call.

### 3. Smaller micro-batches with gradient accumulation

The per-step batch size is now `8`, with four steps accumulated before updating the weights:

```python
"batch_size": 8,
"gradient_accumulation_steps": 4,
```

This keeps an effective batch size of 32 while lowering peak VRAM usage and allowing the full sequence length to fit on the laptop GPU.

### 4. Faster input pipeline

The training DataLoader now uses:

```python
"num_workers": 2,
"prefetch_factor": 2,
```

It also uses pinned memory and persistent workers. Batches can be prepared while the GPU is processing the previous batch, reducing CPU-to-GPU idle time.

### 5. Fused Adam optimizer

CUDA uses PyTorch's fused Adam implementation when available. This combines optimizer operations into fewer GPU kernels and reduces optimizer overhead.

### 6. Reduced logging overhead

TensorBoard is no longer flushed to disk after every batch. It is flushed once per epoch instead. Frequent disk flushes were unnecessarily slowing the training loop.

### 7. CUDA initialization fix

The device is initialized as a `torch.device` before accessing its properties. This makes GPU name and memory reporting reliable on CUDA systems.

### 8. Correct target padding mask

The loss now ignores the target tokenizer's `[PAD]` token rather than the source tokenizer's padding token:

```python
nn.CrossEntropyLoss(
    ignore_index=tokenizer_tgt.token_to_id('[PAD]'),
    label_smoothing=0.1,
)
```


## Repository Structure

`
Transformer using PyTorch/
├── configs/               # Training and experiment YAML configurations
├── data/
│   ├── DATASHEET.md       # Benchmark and training dataset provenance
│   ├── benchmarks/        # Frozen anti-leak evaluation splits (FLORES, IN22)
│   └── tokenizers/        # Active English-Hindi Akshara tokenizers (v2)
├── docs/                  # Reference specifications and upgrade roadmaps
├── models/                # Modular Transformer implementations
│   ├── basic.py           # Vanilla Transformer baseline (Vaswani et al., 2017)
│   └── modern.py          # Modern SOTA Transformer (RMSNorm, RoPE, SwiGLU, KV Cache)
├── research/              # Advanced NMT research modules & unit tests
├── scripts/               # Data processing, benchmark & tokenizer CLI scripts
├── tokenizer/             # Akshara BPE tokenizer library & unit tests
├── config.py              # Configuration loader with smart tokenizer resolution
├── dataset.py             # Bilingual translation dataset
├── evaluate.py            # Multi-metric benchmark evaluation runner
├── model.py               # Backward-compatible shim for models.basic
├── modern_model.py        # Backward-compatible shim for models.modern
├── run_experiment.py      # Unified experiment runner
├── train.py               # Main training loop
└── translate.py           # Interactive & CLI translation
`

## Running training

## Building the Samanantar English-Hindi tokenizers

The project is configured for the Hindi Samanantar corpus on Hugging Face:
`ai4bharat/samanantar`, configuration `hi`. This is the approximately 10.1-million-pair
corpus intended for the larger vocabulary experiment.

First, make a quick 1,000-pair prototype:

```powershell
python scripts/build_tokenizers.py --max-samples 1000
```

For the full training split, remove the prototype files and run:

```powershell
Remove-Item data/tokenizers/tokenizer_en.akshara.v2.json, data/tokenizers/tokenizer_hi.akshara.v2.json
python scripts/build_tokenizers.py
```

This creates separate, versioned vocabularies:

- English: maximum 16,000 entries
- Hindi: maximum 40,000 entries, with Akshara pre-segmentation before BPE
- Each vocabulary budget includes the four special tokens and fallback units.
- A `▁` word-start marker lets BPE learn word-initial subwords without merging across words.
- Special IDs: `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`

The full corpus contains about 10.1 million training pairs. The current pure-Python BPE trainer is suitable for the prototype first; full-corpus training may require substantial RAM and take a long time because it repeatedly scans the corpus while learning merges.

The optimized trainer uses a lazy heap for pair selection and updates only sequences containing the selected pair. Encoding uses a merge-rank heap instead of scanning the sentence once per merge. This removes the old `O(merges * corpus_pairs)` rescans, but exact BPE still needs mutable token sequences and a pair-to-sequence index. On a typical laptop, the full 20-million-sentence run can still exceed available RAM; a disk-backed corpus/index implementation is required if memory pressure becomes critical.

After the tokenizer files are created, start model training with:

```powershell
python train.py
```

Do not reuse the 1,000-pair prototype tokenizers for final experiments. Delete them and rebuild from the full Samanantar training split first. The old unversioned/shared tokenizer files are intentionally not used by tokenizer v2.

## Research workflow

The project now includes a staged research layer matching the upgrade roadmap:

- `research/data.py` and `prepare_data.py` provide auditable cleaning, deduplication, contamination fingerprints, and per-dataset datasheets.
- `research/byte_tokenizer.py` is the UTF-8 byte-level control tokenizer.
- `research/evaluation.py` provides BLEU/chrF integration, paired bootstrap significance for sentence-level metrics, copy-preservation checks, and initial error-taxonomy flags.
- `research/adapters.py` contains explicit, optional adapters for LaBSE, COMET/CometKiwi, and Stanza; no model is downloaded unless you call an adapter.
- `research/decoding.py`, `research/synthetic.py`, `research/multilingual.py`, and `research/deployment.py` support MBR/speculative-decoding experiments, synthetic filtering, family-aware multilingual sampling, and INT8/latency measurements.

Install optional tools only for the experiment you are running:

```powershell
py -m pip install -r requirements-research.txt
```

Prepare an explicit local source TSV (English in column 1, Hindi in column 2) rather than silently downloading a corpus:

```powershell
python scripts/prepare_data.py raw.tsv data/processed/samanantar_en_hi.tsv --source-name Samanantar --source-url https://ai4bharat.iitm.ac.in/samanantar --license CC0
```

Frozen evaluation sets are documented in [data/DATASHEET.md](data/DATASHEET.md) and must never enter tokenizer or model training.

To cache the full Samanantar Hindi training configuration for the core experiment:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_samanantar.py
```

Use `--max-samples 1000` only to smoke-test the pipeline. The script writes a
local manifest and explicitly keeps IN22 and FLORES out of training.

Use the Python environment where CUDA-enabled PyTorch is installed:

```powershell
.\\.venv\\Scripts\\python.exe train.py
```

The current environment reports:

- PyTorch `2.11.0+cu128`
- CUDA `12.8`
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- GPU memory: approximately 8 GB
- BF16 support: enabled

## Tuning for this laptop

If CUDA out-of-memory occurs, reduce `batch_size` from `8` to `4` and increase `gradient_accumulation_steps` from `4` to `8` to preserve the same effective batch size.

If VRAM usage is low and training is stable, try increasing `batch_size` to `12` or `16`, then reduce gradient accumulation if desired. The largest batch size that fits without swapping or out-of-memory errors will generally provide the best throughput.

If GPU utilization is low, keep `num_workers` at `2` or try `4`. On Windows, too many workers can increase startup and memory overhead, so increase this gradually.
