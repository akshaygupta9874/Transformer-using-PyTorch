"""Decoding algorithms: Beam search utilities, Minimum Bayes Risk (MBR, B9), and Speculative Decoding (B17)."""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn.functional as F


def length_penalty(length: int, alpha: float) -> float:
    """Standard GNMT length penalty: ((5 + length) / 6) ** alpha."""
    return ((5 + length) / 6) ** alpha


def beam_search_decode(
    model,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    beam_size: int = 4,
    max_len: int = 100,
    alpha: float = 0.6,
    device: Optional[torch.device] = None,
) -> list[int]:
    """Beam search decoding with length normalization."""
    from dataset import causal_mask

    device = device or source.device
    sos_id = tokenizer_tgt.token_to_id("<BOS>")
    eos_id = tokenizer_tgt.token_to_id("<EOS>")

    encoder_output = model.encode(source, source_mask)
    beams: list[tuple[float, list[int]]] = [(0.0, [sos_id])]
    completed_beams: list[tuple[float, list[int]]] = []

    for _ in range(max_len):
        candidates = []
        for log_prob, seq in beams:
            if seq[-1] == eos_id:
                completed_beams.append((log_prob, seq))
                continue

            dec_in = torch.tensor([seq], dtype=torch.long, device=device)
            dec_mask = causal_mask(dec_in.size(1)).to(device)
            out = model.decode(encoder_output, source_mask, dec_in, dec_mask)
            logits = model.project(out[:, -1]).squeeze(0)
            log_probs = F.log_softmax(logits, dim=-1)

            top_log_probs, top_indices = torch.topk(log_probs, k=beam_size)
            for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                candidates.append((log_prob + lp, seq + [idx]))

        if not candidates:
            break

        candidates.sort(key=lambda item: item[0] / length_penalty(len(item[1]), alpha), reverse=True)
        beams = candidates[:beam_size]

        if len(completed_beams) >= beam_size:
            break

    all_completed = completed_beams or beams
    all_completed.sort(key=lambda item: item[0] / length_penalty(len(item[1]), alpha), reverse=True)
    return all_completed[0][1]


def pure_python_chrf(hypothesis: str, reference: str, n_order: int = 6, beta: float = 2.0) -> float:
    """Pure-Python character n-gram F-score (chrF).
    
    Operates without external dependencies for robust MBR decoding.
    """
    hyp_chars = hypothesis.replace(" ", "")
    ref_chars = reference.replace(" ", "")
    if not hyp_chars and not ref_chars:
        return 1.0
    if not hyp_chars or not ref_chars:
        return 0.0

    f_scores = []
    for n in range(1, n_order + 1):
        if len(hyp_chars) < n or len(ref_chars) < n:
            continue
        hyp_ngrams = Counter(hyp_chars[i:i + n] for i in range(len(hyp_chars) - n + 1))
        ref_ngrams = Counter(ref_chars[i:i + n] for i in range(len(ref_chars) - n + 1))
        
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total_hyp = sum(hyp_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        prec = matches / total_hyp if total_hyp > 0 else 0.0
        rec = matches / total_ref if total_ref > 0 else 0.0

        if prec + rec == 0.0:
            f_scores.append(0.0)
        else:
            beta2 = beta ** 2
            f = (1 + beta2) * (prec * rec) / (beta2 * prec + rec)
            f_scores.append(f)

    return (sum(f_scores) / len(f_scores)) if f_scores else 0.0


def mbr_rerank(candidates: list[str], utility: Optional[Callable[[str, str], float]] = None) -> str:
    """Select the candidate with highest expected utility under all peer hypotheses.
    
    Args:
        candidates: List of sampled candidate translations.
        utility: Symmetric utility function (cand, ref) -> score. Defaults to pure_python_chrf.
    """
    if not candidates:
        raise ValueError("MBR needs at least one candidate.")
    if len(candidates) == 1:
        return candidates[0]

    metric = utility or pure_python_chrf
    scores = []
    n = len(candidates)
    for i, candidate in enumerate(candidates):
        expected_utility = sum(metric(candidate, peer) for j, peer in enumerate(candidates) if i != j) / (n - 1)
        scores.append(expected_utility)
    
    best_idx = int(torch.argmax(torch.tensor(scores)).item())
    return candidates[best_idx]


def smbr_rerank(candidates: list[str], source: str, qe_utility: Callable[[str, str], float]) -> str:
    """Source-conditioned MBR (sMBR) using a reference-free Quality Estimation utility."""
    if not candidates:
        raise ValueError("sMBR needs at least one candidate.")
    return max(candidates, key=lambda candidate: qe_utility(source, candidate))


def top_p_sampling(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> int:
    """Apply temperature and nucleus (top-p) sampling to logits vector."""
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    
    scaled_logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to the right to keep also the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_logits[sorted_indices_to_remove] = -float("Inf")
    probs = F.softmax(sorted_logits, dim=-1)
    prev_idx = torch.multinomial(probs, num_samples=1)
    return int(sorted_indices[prev_idx].item())


def sample_candidate(
    model,
    encoder_output: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    max_len: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> list[int]:
    """Sample a single translation sequence using top-p nucleus sampling."""
    from dataset import causal_mask

    sos_id = tokenizer_tgt.token_to_id("<BOS>")
    eos_id = tokenizer_tgt.token_to_id("<EOS>")
    device = device or encoder_output.device

    decoder_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    while decoder_input.size(1) < max_len:
        dec_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, dec_mask)
        logits = model.project(out[:, -1]).squeeze(0)
        next_token = top_p_sampling(logits, temperature=temperature, top_p=top_p)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == eos_id:
            break

    return decoder_input[0].tolist()


def mbr_decode(
    model,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_tgt,
    num_samples: int = 16,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_len: int = 100,
    utility: Optional[Callable[[str, str], float]] = None,
) -> str:
    """Generate N candidate translations and select the MBR consensus translation."""
    encoder_output = model.encode(source, source_mask)
    candidates = []
    for _ in range(num_samples):
        token_ids = sample_candidate(
            model,
            encoder_output,
            source_mask,
            tokenizer_tgt,
            max_len=max_len,
            temperature=temperature,
            top_p=top_p,
        )
        text = tokenizer_tgt.decode(token_ids)
        candidates.append(text)

    return mbr_rerank(candidates, utility=utility)


def greedy_speculative_accept(
    draft_ids: list[int],
    verifier_next: Callable[[list[int]], int],
    prefix: list[int],
) -> tuple[list[int], int]:
    """Deterministic greedy verification step of speculative decoding."""
    accepted = list(prefix)
    for proposed in draft_ids:
        verified = verifier_next(accepted)
        if verified != proposed:
            accepted.append(verified)
            return accepted, len(accepted) - len(prefix) - 1
        accepted.append(proposed)
    return accepted, len(draft_ids)


def speculative_decode_step(
    draft_model,
    target_model,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    current_tokens: list[int],
    gamma: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
) -> tuple[list[int], int]:
    """Single speculative decoding step: proposes gamma tokens with draft model and validates with target model."""
    from dataset import causal_mask

    # Step 1: Draft model generates gamma tokens greedily
    draft_seq = list(current_tokens)
    draft_tokens = []
    for _ in range(gamma):
        if len(draft_seq) >= max_len:
            break
        draft_in = torch.tensor([draft_seq], dtype=torch.long, device=device)
        draft_mask = causal_mask(draft_in.size(1)).to(device)
        draft_enc = draft_model.encode(source, source_mask)
        draft_out = draft_model.decode(draft_enc, source_mask, draft_in, draft_mask)
        next_tok = int(draft_model.project(draft_out[:, -1]).argmax(dim=-1).item())
        draft_tokens.append(next_tok)
        draft_seq.append(next_tok)
        if next_tok == eos_id:
            break

    if not draft_tokens:
        return current_tokens, 0

    # Step 2: Target model evaluates all proposed positions in ONE forward pass
    eval_seq = current_tokens + draft_tokens
    target_in = torch.tensor([eval_seq], dtype=torch.long, device=device)
    target_mask = causal_mask(target_in.size(1)).to(device)
    target_enc = target_model.encode(source, source_mask)
    target_out = target_model.decode(target_enc, source_mask, target_in, target_mask)
    target_logits = target_model.project(target_out[0])  # (seq_len, vocab_size)

    # Compare draft tokens with target model argmax at each position
    accepted = list(current_tokens)
    num_accepted = 0
    start_pos = len(current_tokens) - 1

    for i, proposed in enumerate(draft_tokens):
        pos = start_pos + i
        target_choice = int(target_logits[pos].argmax().item())
        if target_choice == proposed:
            accepted.append(proposed)
            num_accepted += 1
            if proposed == eos_id:
                return accepted, num_accepted
        else:
            # Reject: take target model's choice and stop speculative window
            accepted.append(target_choice)
            return accepted, num_accepted

    # If all gamma accepted, take bonus token from target model
    last_pos = start_pos + len(draft_tokens)
    if last_pos < len(target_logits):
        bonus_token = int(target_logits[last_pos].argmax().item())
        accepted.append(bonus_token)

    return accepted, num_accepted
