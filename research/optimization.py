"""Modern Training Optimization: EMA, Checkpoint Averaging, Cosine Warmup, and MRT Loss (Part A, Sections 9 & 10)."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class ModelEMA:
    """Exponential Moving Average (EMA) of model parameters.
    
    Stabilizes training and consistently improves translation quality by +0.5 to +1.5 BLEU.
    theta_ema = beta * theta_ema + (1 - beta) * theta
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self) -> None:
        """Update shadow weights with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        """Backup model weights and load EMA shadow weights for validation/inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model weights after validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()


class CheckpointAverager:
    """Averages model parameters across the top-k or last-k checkpoints."""

    @staticmethod
    def average_checkpoints(checkpoint_paths: List[str | Path], output_path: str | Path) -> None:
        if not checkpoint_paths:
            raise ValueError("Need at least one checkpoint to average.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Averaging {len(checkpoint_paths)} checkpoints...")

        avg_state = {}
        num_checkpoints = len(checkpoint_paths)

        for i, cp_path in enumerate(checkpoint_paths):
            cp = torch.load(cp_path, map_location="cpu", weights_only=False)
            state_dict = cp.get("model_state_dict", cp)

            for key, val in state_dict.items():
                if val.dtype.is_floating_point:
                    if key not in avg_state:
                        avg_state[key] = val.clone().float() / num_checkpoints
                    else:
                        avg_state[key] += val.float() / num_checkpoints
                else:
                    avg_state[key] = val.clone()

        # Save averaged checkpoint
        base_cp = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=False)
        base_cp["model_state_dict"] = avg_state
        base_cp["averaged_from"] = [str(p) for p in checkpoint_paths]
        torch.save(base_cp, output_path)
        print(f"Saved averaged checkpoint to {output_path}")


class MinimumRiskTrainingLoss(nn.Module):
    """Minimum Risk Training (MRT) sequence-level loss (Shen et al., 2016).
    
    Directly optimizes expected metric risk (e.g., 100 - chrF) across sampled hypotheses:
    L_MRT = sum_i P_alpha(y_i | x) * Risk(y_i, y^*)
    """

    def __init__(self, alpha: float = 0.005, utility_fn: Optional[Callable[[str, str], float]] = None):
        super().__init__()
        self.alpha = alpha
        from research.decoding import pure_python_chrf
        self.utility_fn = utility_fn or (lambda hyp, ref: pure_python_chrf(hyp, ref) * 100.0)

    def forward(
        self,
        candidate_log_probs: torch.Tensor,
        candidate_texts: List[List[str]],
        reference_texts: List[str],
    ) -> torch.Tensor:
        """Compute MRT loss.
        
        Args:
            candidate_log_probs: Tensor of shape (batch_size, num_samples) - sequence log probabilities
            candidate_texts: List of lists containing decoded candidate translations
            reference_texts: List of reference translations
        """
        batch_size, num_samples = candidate_log_probs.shape
        scaled_log_probs = candidate_log_probs * self.alpha
        # Renormalized distribution over samples: P_alpha
        sample_probs = torch.softmax(scaled_log_probs, dim=-1)

        losses = []
        for b in range(batch_size):
            ref = reference_texts[b]
            risks = []
            for s in range(num_samples):
                hyp = candidate_texts[b][s]
                utility = self.utility_fn(hyp, ref)
                # Risk is negative utility
                risk = 100.0 - utility
                risks.append(risk)

            risk_tensor = torch.tensor(risks, device=candidate_log_probs.device, dtype=torch.float32)
            # Expected risk
            expected_risk = (sample_probs[b] * risk_tensor).sum()
            losses.append(expected_risk)

        return torch.stack(losses).mean()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Create a schedule with a linear warmup and cosine decay to min_lr_ratio."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class TokenFrequencyLossAnalyzer:
    """
    Analyzes prediction loss, perplexity, and accuracy across token frequency tiers.
    
    Partitions the target vocabulary into:
    - Head: Top 5% most frequent tokens (frequent grammatical particles, common words)
    - Mid: Next 45% tokens (moderate frequency content words)
    - Rare: Bottom 50% tokens (rare inflected forms, domain terms, rare named entities)
    - Fallback: Tokens corresponding to base character/byte fallback units
    """

    def __init__(
        self,
        tokenizer: Any,
        token_frequencies: Optional[Dict[int, int]] = None,
        pad_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else len(tokenizer.tokens)
        self.vocab_size = vocab_size

        self.tier_map: Dict[int, str] = {}
        if token_frequencies:
            sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
            non_pad = [tok for tok, _ in sorted_tokens if tok != pad_id]
            n_tokens = len(non_pad)
            head_cutoff = int(n_tokens * 0.05)
            mid_cutoff = int(n_tokens * 0.50)
            for idx, tok in enumerate(non_pad):
                if idx < head_cutoff:
                    self.tier_map[tok] = "head"
                elif idx < mid_cutoff:
                    self.tier_map[tok] = "mid"
                else:
                    self.tier_map[tok] = "rare"
        else:
            for idx in range(vocab_size):
                if idx == pad_id:
                    continue
                if idx < int(vocab_size * 0.05):
                    self.tier_map[idx] = "head"
                elif idx < int(vocab_size * 0.50):
                    self.tier_map[idx] = "mid"
                else:
                    self.tier_map[idx] = "rare"

        byte_base = getattr(tokenizer, "byte_base", None)
        byte_end = getattr(tokenizer, "byte_end", None)
        if byte_base is not None and byte_end is not None:
            for b_idx in range(byte_base, byte_end):
                self.tier_map[b_idx] = "fallback"

    def analyze(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Computes per-tier loss, accuracy, and perplexity for a batch of predictions.
        
        Args:
            logits: (Batch, seq_len, vocab_size) or (N, vocab_size)
            labels: (Batch, seq_len) or (N,)
        """
        flat_logits = logits.reshape(-1, self.vocab_size).detach()
        flat_labels = labels.reshape(-1).detach()

        loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=self.pad_id)
        element_losses = loss_fn(flat_logits, flat_labels)
        preds = flat_logits.argmax(dim=-1)
        corrects = (preds == flat_labels)

        tier_stats: Dict[str, Dict[str, Any]] = {
            "head": {"losses": [], "correct": 0, "total": 0},
            "mid": {"losses": [], "correct": 0, "total": 0},
            "rare": {"losses": [], "correct": 0, "total": 0},
            "fallback": {"losses": [], "correct": 0, "total": 0},
            "overall": {"losses": [], "correct": 0, "total": 0},
        }

        valid_mask = (flat_labels != self.pad_id)
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        for idx in valid_indices:
            label_id = flat_labels[idx].item()
            loss_val = element_losses[idx].item()
            is_corr = corrects[idx].item()
            tier = self.tier_map.get(label_id, "rare")

            tier_stats[tier]["losses"].append(loss_val)
            tier_stats[tier]["total"] += 1
            if is_corr:
                tier_stats[tier]["correct"] += 1

            tier_stats["overall"]["losses"].append(loss_val)
            tier_stats["overall"]["total"] += 1
            if is_corr:
                tier_stats["overall"]["correct"] += 1

        results = {}
        for tier, data in tier_stats.items():
            cnt = data["total"]
            if cnt > 0:
                mean_loss = sum(data["losses"]) / cnt
                acc = (data["correct"] / cnt) * 100.0
                ppl = math.exp(min(20.0, mean_loss))
            else:
                mean_loss = 0.0
                acc = 0.0
                ppl = 1.0
            results[tier] = {
                "count": cnt,
                "loss": mean_loss,
                "accuracy": acc,
                "perplexity": ppl,
            }

        return results


class TargetAwareLoss(nn.Module):
    """
    Target-aware Cross-Entropy Loss with optional token-frequency smoothing,
    rare entity weighting, and target-side label smoothing.
    
    Formula:
      L_i = w_{y_i} * CE(logits_i, y_i; label_smoothing)
    where:
      w_v = (freq_v + eps)^(-gamma), normalized to mean 1.0, clamped to [min_weight, max_weight]
      Entity tokens (numbers, URLs, placeholders) can receive an additional multiplier.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        label_smoothing: float = 0.1,
        frequency_weights: Optional[torch.Tensor] = None,
        entity_token_ids: Optional[Sequence[int]] = None,
        entity_weight_multiplier: float = 1.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing

        weights = torch.ones(vocab_size, dtype=torch.float32)
        if frequency_weights is not None:
            weights = weights * frequency_weights.float()

        if entity_token_ids:
            for eid in entity_token_ids:
                if 0 <= eid < vocab_size:
                    weights[eid] *= entity_weight_multiplier

        weights[pad_id] = 0.0

        non_pad_mask = torch.ones(vocab_size, dtype=torch.bool)
        non_pad_mask[pad_id] = False
        if weights[non_pad_mask].sum() > 0:
            weights[non_pad_mask] = weights[non_pad_mask] / weights[non_pad_mask].mean()

        self.register_buffer("weight", weights)
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )


def compare_label_smoothing(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothings: Sequence[float] = (0.0, 0.05, 0.1),
    pad_id: int = 0,
) -> Dict[float, Dict[str, float]]:
    """
    Compares cross_entropy loss under different label smoothing settings (0.0, 0.05, 0.1).
    Evaluates loss value, effective target probability weight, and entropy spread.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    results = {}
    for eps in smoothings:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=eps)
        val = loss_fn(flat_logits, flat_targets).item()
        results[eps] = {
            "loss": val,
            "perplexity": math.exp(min(20.0, val)),
            "label_smoothing": eps,
            "target_confidence": 1.0 - eps,
        }
    return results
