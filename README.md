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

## Running training

Use the Python environment where CUDA-enabled PyTorch is installed:

```powershell
C:/Users/navin/AppData/Local/Programs/Python/Python312/python.exe train.py
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
