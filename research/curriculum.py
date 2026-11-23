"""Curriculum Learning for English-to-Hindi NMT (B13).

Provides difficulty scoring across three concrete signals:
1. Sentence length (source + target token count)
2. Target token rarity (negative log unigram frequency)
3. Akshara morphological complexity (conjunct / virama count)

Includes a decile-bucketed CurriculumSampler that smoothly anneals from
easy examples to the full corpus distribution over training.
"""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler


def compute_length_difficulty(source_len: int, target_len: int, max_seq_len: int = 350) -> float:
    """Normalized sequence length difficulty in [0.0, 1.0]."""
    return min(1.0, (source_len + target_len) / (2.0 * max_seq_len))


def compute_akshara_complexity(hindi_text: str) -> float:
    """Measure morphological complexity by counting virama (U+094D) conjunct clusters."""
    if not hindi_text:
        return 0.0
    virama_count = hindi_text.count("\u094D")
    return min(1.0, virama_count / 10.0)


def compute_rarity_difficulty(token_ids: list[int], unigram_frequencies: dict[int, int], total_tokens: int) -> float:
    """Average negative log probability of target tokens."""
    if not token_ids or total_tokens <= 0:
        return 0.0
    nll_sum = 0.0
    for tok in token_ids:
        freq = unigram_frequencies.get(tok, 1)
        prob = max(1e-9, freq / total_tokens)
        nll_sum += -math.log(prob)
    return nll_sum / len(token_ids)


def score_dataset_difficulty(
    records: list[dict],
    tokenizer_src,
    tokenizer_tgt,
    difficulty_metric: str = "length",
    max_seq_len: int = 350,
) -> list[float]:
    """Calculate per-sentence difficulty scores across the dataset.
    
    Supported metrics: 'length', 'akshara', 'rarity', 'composite'.
    """
    scores = []
    
    # Precompute unigram frequencies if needed
    if difficulty_metric in ("rarity", "composite"):
        token_counter = Counter()
        total_tokens = 0
        for item in records:
            tgt_text = item["tgt"] if "tgt" in item else item.get("translation", {}).get("hi", "")
            tokens = tokenizer_tgt.encode(tgt_text).ids
            token_counter.update(tokens)
            total_tokens += len(tokens)
    else:
        token_counter = {}
        total_tokens = 0

    for item in records:
        src_text = item["src"] if "src" in item else item.get("translation", {}).get("en", "")
        tgt_text = item["tgt"] if "tgt" in item else item.get("translation", {}).get("hi", "")
        
        src_ids = tokenizer_src.encode(src_text).ids
        tgt_ids = tokenizer_tgt.encode(tgt_text).ids

        len_score = compute_length_difficulty(len(src_ids), len(tgt_ids), max_seq_len)
        akshara_score = compute_akshara_complexity(tgt_text)

        if difficulty_metric == "length":
            score = len_score
        elif difficulty_metric == "akshara":
            score = akshara_score
        elif difficulty_metric == "rarity":
            score = compute_rarity_difficulty(tgt_ids, token_counter, total_tokens)
        elif difficulty_metric == "composite":
            rarity_score = compute_rarity_difficulty(tgt_ids, token_counter, total_tokens)
            score = 0.4 * len_score + 0.3 * akshara_score + 0.3 * min(1.0, rarity_score / 15.0)
        else:
            raise ValueError(f"Unknown difficulty metric: {difficulty_metric}")
        
        scores.append(score)

    return scores


class CurriculumSampler(Sampler[int]):
    """Samples examples from difficulty deciles, annealing from easy to full dataset.
    
    During the warmup period (step/total_steps < warmup_fraction), sampling is
    concentrated in the easiest deciles. As training progresses, the active pool
    expands linearly until all deciles are sampled uniformly.
    """

    def __init__(
        self,
        difficulty_scores: list[float],
        warmup_fraction: float = 0.25,
        total_steps: int = 10000,
        seed: int = 13,
    ):
        super().__init__()
        self.num_samples = len(difficulty_scores)
        self.warmup_fraction = max(0.01, warmup_fraction)
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        self.rng = random.Random(seed)

        # Sort indices by difficulty
        sorted_indices = sorted(range(self.num_samples), key=lambda i: difficulty_scores[i])
        
        # Partition into 10 decile buckets
        self.deciles: list[list[int]] = []
        chunk_size = max(1, math.ceil(self.num_samples / 10))
        for d in range(10):
            start = d * chunk_size
            end = min(self.num_samples, (d + 1) * chunk_size)
            if start < self.num_samples:
                self.deciles.append(sorted_indices[start:end])

    def set_step(self, step: int) -> None:
        """Update current training step to adjust curriculum progression."""
        self.current_step = step

    def get_progress(self) -> float:
        """Curriculum progress in [0.0, 1.0]."""
        return min(1.0, self.current_step / (self.total_steps * self.warmup_fraction))

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        progress = self.get_progress()
        
        # If progress < 1.0, active deciles expand from top 3 to all 10
        # When progress == 1.0, all 10 deciles are active
        num_active_deciles = min(10, max(3, int(3 + progress * 7)))
        
        active_indices: list[int] = []
        for d in range(num_active_deciles):
            if d < len(self.deciles):
                active_indices.extend(self.deciles[d])

        # Sample with replacement or uniform permutation to length
        indices = self.rng.choices(active_indices, k=self.num_samples) if active_indices else list(range(self.num_samples))
        self.rng.shuffle(indices)
        return iter(indices)
