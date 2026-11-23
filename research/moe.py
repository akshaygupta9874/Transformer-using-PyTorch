"""Sparse Mixture-of-Experts (MoE) FFN with Top-k Gating and Load Balancing (Part A, Section 19)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_model import SwiGLU


class MoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts layer with Top-k routing and auxiliary load balancing loss."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k cannot exceed num_experts"
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router gating network
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Independent expert FFNs
        self.experts = nn.ModuleList([
            SwiGLU(d_model, d_ff, dropout=dropout) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MoE forward pass and auxiliary load-balancing loss.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, d_model)
            aux_loss: Scalar auxiliary loss encouraging balanced expert routing
        """
        batch_size, seq_len, d_model = x.shape
        flat_x = x.view(-1, d_model)  # (N, d_model)
        num_tokens = flat_x.size(0)

        # Compute router logits and routing probabilities
        router_logits = self.router(flat_x)  # (N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k expert selection
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (N, top_k)
        # Renormalize top-k weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Compute auxiliary load-balancing loss: num_experts * sum_i(f_i * P_i)
        # f_i: fraction of tokens dispatched to expert i
        # P_i: average routing probability of expert i
        expert_mask = torch.zeros((num_tokens, self.num_experts), device=x.device)
        expert_mask.scatter_add_(1, topk_indices, torch.ones_like(topk_weights))
        tokens_per_expert = expert_mask.mean(dim=0)  # f_i
        mean_prob_per_expert = router_probs.mean(dim=0)  # P_i
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * mean_prob_per_expert)

        # Dispatch tokens to selected experts
        final_output = torch.zeros_like(flat_x)
        for e_idx, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            idx_in_topk = (topk_indices == e_idx).nonzero(as_tuple=False)
            if idx_in_topk.numel() == 0:
                continue

            token_indices = idx_in_topk[:, 0]
            k_pos = idx_in_topk[:, 1]
            expert_tokens = flat_x[token_indices]
            expert_out = expert(expert_tokens)
            weights = topk_weights[token_indices, k_pos].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_out * weights)

        return final_output.view(batch_size, seq_len, d_model), aux_loss
