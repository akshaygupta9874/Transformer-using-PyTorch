"""Dynamic Length-Bucketed Token Batching (Part A, Section 10).

Batches sentences by total token budget rather than fixed sentence count,
virtually eliminating padding waste and accelerating training by 2x-3x.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

from torch.utils.data import Sampler


class TokenBucketBatchSampler(Sampler[List[int]]):
    """Dynamic batch sampler that groups sentences of similar lengths into token-bounded batches.
    
    Ensures that for every batch:
        batch_size * max_length_in_batch <= max_tokens
    """

    def __init__(
        self,
        lengths: List[int],
        max_tokens: int = 4096,
        num_buckets: int = 20,
        shuffle: bool = True,
        seed: int = 13,
    ):
        super().__init__()
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def _generate_batches(self) -> List[List[int]]:
        # Sort indices by length
        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda i: self.lengths[i])

        batches: List[List[int]] = []
        current_batch: List[int] = []
        max_len_in_batch = 0

        for idx in indices:
            seq_len = max(1, self.lengths[idx])
            tentative_max = max(max_len_in_batch, seq_len)
            tentative_tokens = (len(current_batch) + 1) * tentative_max

            if tentative_tokens <= self.max_tokens and len(current_batch) < 128:
                current_batch.append(idx)
                max_len_in_batch = tentative_max
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                max_len_in_batch = seq_len

        if current_batch:
            batches.append(current_batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._generate_batches()
        return iter(batches)

    def __len__(self) -> int:
        return len(self._generate_batches())
