"""Back-Translation & Iterative Self-Training Pipeline (Part A, Sections 13 & 14).

Provides:
- Synthetic source generation from monolingual Hindi text via reverse translation
- Synthetic noise injections (word dropout, word blanking, word shuffling)
- Controlled bitext mixing (authentic : synthetic)
- Iterative self-training round management
"""

from __future__ import annotations

import csv
from pathlib import Path
import random
from typing import Callable, List, Tuple, Optional


class SyntheticNoiser:
    """Applies noise perturbations to synthetic source text (Sennrich et al., 2016)."""

    def __init__(self, drop_prob: float = 0.1, blank_prob: float = 0.1, shuffle_window: int = 3, seed: int = 13):
        self.drop_prob = drop_prob
        self.blank_prob = blank_prob
        self.shuffle_window = shuffle_window
        self.rng = random.Random(seed)

    def apply(self, text: str) -> str:
        words = text.split()
        if not words:
            return text

        # 1. Word Dropout
        if self.drop_prob > 0:
            words = [w for w in words if self.rng.random() > self.drop_prob] or words

        # 2. Word Blanking (replace with <UNK> or blank token)
        if self.blank_prob > 0:
            words = ["<UNK>" if self.rng.random() < self.blank_prob else w for w in words]

        # 3. Local Word Shuffling
        if self.shuffle_window > 1 and len(words) > 1:
            permuted = list(words)
            for i in range(len(permuted)):
                swap_with = min(len(permuted) - 1, i + self.rng.randint(0, self.shuffle_window))
                permuted[i], permuted[swap_with] = permuted[swap_with], permuted[i]
            words = permuted

        return " ".join(words)


class BackTranslationPipeline:
    """Generates synthetic training pairs and mixes them with genuine parallel data."""

    def __init__(self, reverse_translator: Callable[[str], str], noiser: Optional[SyntheticNoiser] = None):
        self.reverse_translator = reverse_translator
        self.noiser = noiser or SyntheticNoiser()

    def generate_synthetic_pairs(
        self,
        monolingual_target_sentences: List[str],
        apply_noise: bool = True,
    ) -> List[Tuple[str, str]]:
        """Translate monolingual Hindi target sentences into synthetic English source sentences."""
        synthetic_pairs = []
        for authentic_hi in monolingual_target_sentences:
            authentic_hi = authentic_hi.strip()
            if not authentic_hi:
                continue

            synthetic_en = self.reverse_translator(authentic_hi)
            if apply_noise:
                synthetic_en = self.noiser.apply(synthetic_en)

            synthetic_pairs.append((synthetic_en, authentic_hi))
        return synthetic_pairs

    @staticmethod
    def mix_datasets(
        authentic_pairs: List[Tuple[str, str]],
        synthetic_pairs: List[Tuple[str, str]],
        synthetic_ratio: float = 0.5,
        seed: int = 13,
    ) -> List[Tuple[str, str]]:
        """Mix authentic and synthetic parallel bitext with a target synthetic ratio.
        
        Args:
            authentic_pairs: List of (english, hindi) genuine pairs
            synthetic_pairs: List of (synthetic_english, genuine_hindi) pairs
            synthetic_ratio: Desired fraction of synthetic pairs in output (e.g. 0.5 = 50:50 mix)
        """
        rng = random.Random(seed)
        num_authentic = len(authentic_pairs)
        if synthetic_ratio <= 0.0 or not synthetic_pairs:
            return list(authentic_pairs)
        if synthetic_ratio >= 1.0:
            return list(synthetic_pairs)

        # Calculate required synthetic count to meet ratio: S / (A + S) = r  =>  S = r * A / (1 - r)
        target_synthetic = int(num_authentic * synthetic_ratio / (1.0 - synthetic_ratio))
        target_synthetic = min(target_synthetic, len(synthetic_pairs))

        sampled_synthetic = rng.sample(synthetic_pairs, k=target_synthetic)
        mixed = list(authentic_pairs) + sampled_synthetic
        rng.shuffle(mixed)
        return mixed

    @staticmethod
    def save_tsv(pairs: List[Tuple[str, str]], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for src, tgt in pairs:
                writer.writerow([src, tgt])
        print(f"Saved {len(pairs):,} parallel pairs to {path}")
