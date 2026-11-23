"""Deployment measurements, dynamic INT8 quantization, distillation loss, and ONNX export (B16)."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def checkpoint_size_mb(path: str | Path) -> float:
    """Return model checkpoint file size in megabytes."""
    return Path(path).stat().st_size / (1024 ** 2)


def dynamic_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to linear layers for CPU deployment."""
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    pad_id: int,
    alpha: float = 0.5,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Combined Cross-Entropy and KL distillation loss with padding mask exclusion.
    
    L = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(softmax(student/T) || softmax(teacher/T))
    """
    # Flatten shapes
    vocab_size = student_logits.size(-1)
    student_flat = student_logits.reshape(-1, vocab_size)
    teacher_flat = teacher_logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # 1. Hard label Cross Entropy
    ce_loss = F.cross_entropy(student_flat, labels_flat, ignore_index=pad_id)

    # 2. Soft distillation KL Divergence
    non_pad_mask = (labels_flat != pad_id).unsqueeze(-1)
    if not non_pad_mask.any():
        return ce_loss

    student_soft = F.log_softmax(student_flat / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_flat / temperature, dim=-1)

    kl = F.kl_div(student_soft, teacher_soft, reduction="none")
    kl_masked = (kl * non_pad_mask).sum() / non_pad_mask.sum().clamp(min=1)
    kd_loss = (temperature ** 2) * kl_masked

    return (1.0 - alpha) * ce_loss + alpha * kd_loss


def timed(callable_: Callable, repeats: int = 10) -> dict[str, float]:
    """Profile runtime durations for a callable."""
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        callable_()
        durations.append(time.perf_counter() - started)
    return {
        "mean_seconds": sum(durations) / repeats,
        "p50_seconds": sorted(durations)[repeats // 2],
        "min_seconds": min(durations),
    }


def benchmark_inference(
    model: nn.Module,
    sample_encoder_input: torch.Tensor,
    sample_encoder_mask: torch.Tensor,
    max_decode_len: int = 30,
    repeats: int = 20,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Measure inference latency (ms/sentence), throughput (tokens/sec), and peak VRAM."""
    from dataset import causal_mask

    device = device or sample_encoder_input.device
    model.eval()

    # Warmup
    with torch.no_grad():
        enc_out = model.encode(sample_encoder_input, sample_encoder_mask)
        dec_in = torch.tensor([[2]], dtype=torch.long, device=device)
        _ = model.decode(enc_out, sample_encoder_mask, dec_in, causal_mask(1).to(device))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies = []
    total_generated_tokens = 0

    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            enc_out = model.encode(sample_encoder_input, sample_encoder_mask)
            dec_in = torch.tensor([[2]], dtype=torch.long, device=device)
            for _ in range(max_decode_len):
                mask = causal_mask(dec_in.size(1)).to(device)
                out = model.decode(enc_out, sample_encoder_mask, dec_in, mask)
                next_tok = model.project(out[:, -1]).argmax(dim=-1, keepdim=True)
                dec_in = torch.cat([dec_in, next_tok], dim=1)
                if next_tok.item() == 3:  # EOS
                    break
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)
            total_generated_tokens += dec_in.size(1)

    mean_sec = sum(latencies) / len(latencies)
    total_time = sum(latencies)
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0

    return {
        "ms_per_sentence": mean_sec * 1000.0,
        "tokens_per_second": total_generated_tokens / max(1e-6, total_time),
        "peak_vram_mb": peak_vram_mb,
        "repeats": repeats,
    }


def export_to_onnx(model: nn.Module, dummy_encoder_in: torch.Tensor, dummy_mask: torch.Tensor, output_path: str | Path) -> None:
    """Export encoder to ONNX format for deployment."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model.encoder,
            (model.src_pos(model.src_embed(dummy_encoder_in)), dummy_mask),
            str(output_path),
            input_names=["src_emb", "src_mask"],
            output_names=["encoder_hidden"],
            dynamic_axes={"src_emb": {0: "batch", 1: "seq"}, "src_mask": {0: "batch", 3: "seq"}},
            opset_version=14,
        )
    print(f"Exported encoder to ONNX at {output_path}")
