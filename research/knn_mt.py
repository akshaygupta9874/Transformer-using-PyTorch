"""kNN-MT: Retrieval-augmented decoding for domain adaptation without retraining (B8).

Provides datastore construction from parallel references, vector indexing
(using FAISS if installed, with a pure-PyTorch vectorized fallback), and
next-token probability interpolation during decoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class KNNMTDatastore:
    """Stores (decoder_hidden_state, next_target_token) key-value pairs."""

    def __init__(self, d_model: int, device: Optional[torch.device] = None):
        self.d_model = d_model
        self.device = device or torch.device("cpu")
        self.keys: list[torch.Tensor] = []
        self.vals: list[int] = []
        self._keys_tensor: Optional[torch.Tensor] = None
        self._vals_tensor: Optional[torch.Tensor] = None
        self._faiss_index = None

    def add(self, key: torch.Tensor, val: int) -> None:
        """Add a single key vector of shape (d_model,) and integer target token val."""
        if key.dim() != 1 or key.shape[0] != self.d_model:
            raise ValueError(f"Expected key shape ({self.d_model},), got {key.shape}")
        self.keys.append(key.detach().cpu().float())
        self.vals.append(int(val))
        self._keys_tensor = None  # Invalidate cached tensors

    def __len__(self) -> int:
        return len(self.vals)

    def consolidate(self) -> None:
        """Stack keys and values into contiguous PyTorch tensors and build index."""
        if not self.keys:
            self._keys_tensor = torch.empty((0, self.d_model), dtype=torch.float32)
            self._vals_tensor = torch.empty((0,), dtype=torch.long)
            return

        self._keys_tensor = torch.stack(self.keys, dim=0)
        # Normalize keys for cosine similarity
        norms = self._keys_tensor.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._keys_tensor = self._keys_tensor / norms
        self._vals_tensor = torch.tensor(self.vals, dtype=torch.long)

        # Attempt to build FAISS index if available
        try:
            import faiss
            index = faiss.IndexFlatIP(self.d_model)
            keys_np = self._keys_tensor.numpy().astype("float32")
            index.add(keys_np)
            self._faiss_index = index
        except ImportError:
            self._faiss_index = None

    def save(self, path: str | Path) -> None:
        """Save datastore tensors and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._keys_tensor is None:
            self.consolidate()
        torch.save({
            "d_model": self.d_model,
            "keys": self._keys_tensor,
            "vals": self._vals_tensor,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: Optional[torch.device] = None) -> "KNNMTDatastore":
        """Load datastore from disk."""
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        ds = cls(d_model=data["d_model"], device=device)
        ds._keys_tensor = data["keys"]
        ds._vals_tensor = data["vals"]
        ds.vals = data["vals"].tolist()
        ds.keys = list(data["keys"])
        ds.consolidate()
        return ds

    def search(self, query: torch.Tensor, k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search top-k nearest keys.
        
        Args:
            query: Tensor of shape (batch_size, d_model) or (d_model,)
            k: number of nearest neighbors
            
        Returns:
            distances: Tensor of shape (batch_size, k) - cosine similarities
            indices: Tensor of shape (batch_size, k) - datastore value token IDs
        """
        if self._keys_tensor is None or len(self._keys_tensor) == 0:
            raise RuntimeError("Datastore is empty. Add entries and call consolidate().")

        if query.dim() == 1:
            query = query.unsqueeze(0)
        batch_size = query.size(0)
        k = min(k, len(self._keys_tensor))

        # Normalize query vector
        query_norm = query / query.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        if self._faiss_index is not None:
            query_np = query_norm.detach().cpu().numpy().astype("float32")
            similarities, neighbor_idxs = self._faiss_index.search(query_np, k)
            sim_tensor = torch.from_numpy(similarities).to(query.device)
            target_ids = self._vals_tensor[neighbor_idxs].to(query.device)
            return sim_tensor, target_ids
        else:
            # PyTorch fallback (runs on CPU or CUDA)
            keys = self._keys_tensor.to(query.device)
            vals = self._vals_tensor.to(query.device)
            # Similarities: (batch_size, num_entries)
            sims = torch.matmul(query_norm, keys.T)
            top_sims, top_idxs = torch.topk(sims, k=k, dim=-1)
            target_ids = vals[top_idxs]
            return top_sims, target_ids


class KNNMTDecoder:
    """Combines model logits with kNN retrieval probabilities."""

    def __init__(self, datastore: KNNMTDatastore, k: int = 8, temperature: float = 10.0, lambda_weight: float = 0.2):
        self.datastore = datastore
        self.k = k
        self.temperature = temperature
        self.lambda_weight = lambda_weight

    def compute_knn_distribution(self, query: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Compute kNN token probability distribution over the target vocabulary.
        
        Args:
            query: Final decoder hidden state of shape (batch_size, d_model)
            vocab_size: Target vocabulary size
            
        Returns:
            knn_probs: Tensor of shape (batch_size, vocab_size)
        """
        batch_size = query.size(0)
        sims, target_ids = self.datastore.search(query, k=self.k)
        
        # Convert similarities into distances and compute softmax weights
        # Higher similarity -> lower distance -> higher probability
        scaled_sims = sims / self.temperature
        weights = F.softmax(scaled_sims, dim=-1)  # (batch_size, k)

        knn_probs = torch.zeros((batch_size, vocab_size), device=query.device, dtype=torch.float32)
        # Scatter weights into vocabulary probability distribution
        for b in range(batch_size):
            knn_probs[b].scatter_add_(0, target_ids[b], weights[b])

        return knn_probs

    def interpolate(self, model_log_probs: torch.Tensor, knn_probs: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate model probability with kNN probability:
        p(y) = (1 - lambda) * p_model(y) + lambda * p_knn(y)
        """
        model_probs = model_log_probs.exp()
        interp_probs = (1.0 - self.lambda_weight) * model_probs + self.lambda_weight * knn_probs
        return interp_probs.clamp(min=1e-12).log()
