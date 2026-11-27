"""research/packed_ref.py -- REFERENCE implementation of packed-sequence masks/positions + an equivalence
harness.  Use it to audit your own ``packed_masks`` / ``pack_pairs`` / ``PackedMemmapCollate``:

    packed output for every document  ==  that document run alone      (to ~1e-5 in fp32)

Conventions (True = may attend, matching F.scaled_dot_product_attention boolean masks):
    doc ids   (B, T) long : 1..k for the k documents packed in a row, 0 = padding
    positions (B, T) long : restart at 0 for every document (RoPE sees document-local positions)
    self mask (B,1,T,T)   : same document (& causal for the decoder).  Padding rows attend to padding keys
                            (>=1 allowed key per row), so no row is ever fully masked.
    cross mask(B,1,T,S)   : tgt doc == src doc.  Rows left with no allowed key (padding queries over a
                            full source row) are given key 0, so the softmax is always defined.

Why "never fully masked": F.scaled_dot_product_attention on recent PyTorch returns 0 for an all-False row on
CPU (I checked 2.15), but older versions/backends returned NaN, and NaN in a padding row poisons the
backward pass through 0*NaN.  Finite garbage in padding rows is harmless: real queries never see pad keys,
and PAD targets are ignored by the loss.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch


def doc_ids_from_lengths(lengths: Sequence[Sequence[int]], T: int) -> torch.Tensor:
    """lengths[b] = token lengths of the documents packed in row b -> (B, T) doc-id tensor."""
    out = torch.zeros(len(lengths), T, dtype=torch.long)
    for b, row in enumerate(lengths):
        s = 0
        for d, n in enumerate(row, start=1):
            out[b, s:s + n] = d
            s += n
        assert s <= T, f"row {b} needs {s} > T={T}"
    return out


def positions_from_doc_ids(doc: torch.Tensor) -> torch.Tensor:
    """Per-document position restarting at 0; padding gets 0."""
    B, T = doc.shape
    idx = torch.arange(T, device=doc.device).expand(B, T)
    start = torch.zeros_like(doc)                                # first index of each token's document
    newdoc = torch.ones(B, T, dtype=torch.bool, device=doc.device)
    newdoc[:, 1:] = doc[:, 1:] != doc[:, :-1]
    first = torch.where(newdoc, idx, torch.zeros_like(idx))
    start = torch.cummax(first, dim=1).values
    pos = idx - start
    return torch.where(doc > 0, pos, torch.zeros_like(pos))


def packed_self_mask(doc: torch.Tensor, causal: bool) -> torch.Tensor:
    B, T = doc.shape
    m = doc[:, :, None] == doc[:, None, :]                       # pad<->pad allowed, real<->pad never
    if causal:
        m = m & torch.ones(T, T, dtype=torch.bool, device=doc.device).tril()[None]
    return m[:, None]                                            # diag is always True -> no empty row


def packed_cross_mask(tgt_doc: torch.Tensor, src_doc: torch.Tensor) -> torch.Tensor:
    m = tgt_doc[:, :, None] == src_doc[:, None, :]               # (B, T, S)
    empty = ~m.any(dim=-1)                                       # pad queries with no pad key to look at
    m[..., 0] |= empty
    return m[:, None]


def build_packed_batch(src_docs: List[List[torch.Tensor]], tgt_docs: List[List[torch.Tensor]], S: int, T: int):
    """src_docs[b] / tgt_docs[b]: lists of 1-D id tensors for the docs of row b.  Returns a dict with ids,
    doc ids, positions and the three masks."""
    sl = [[len(x) for x in row] for row in src_docs]
    tl = [[len(x) for x in row] for row in tgt_docs]
    src_doc, tgt_doc = doc_ids_from_lengths(sl, S), doc_ids_from_lengths(tl, T)
    src = torch.zeros(len(src_docs), S, dtype=torch.long)
    tgt = torch.zeros(len(tgt_docs), T, dtype=torch.long)
    for b, (rs, rt) in enumerate(zip(src_docs, tgt_docs)):
        if rs: src[b, :sum(sl[b])] = torch.cat(rs)
        if rt: tgt[b, :sum(tl[b])] = torch.cat(rt)
    return dict(src=src, tgt=tgt, src_doc=src_doc, tgt_doc=tgt_doc,
                src_pos=positions_from_doc_ids(src_doc), tgt_pos=positions_from_doc_ids(tgt_doc),
                enc_mask=packed_self_mask(src_doc, causal=False),
                dec_mask=packed_self_mask(tgt_doc, causal=True),
                cross_mask=packed_cross_mask(tgt_doc, src_doc))


def expand_doc_weights(doc: torch.Tensor, doc_weights: torch.Tensor) -> torch.Tensor:
    """Per-document quality weights -> per-token weights (B, T).  doc_weights: (B, K+1), column 0 unused.
    Packed rows hold several documents, so a per-ROW sample_weight is wrong; pass this (B,T) tensor as
    ``sample_weight`` to chunked_cross_entropy / window_normalizer instead."""
    return torch.gather(doc_weights, 1, doc)


def check_packed_equivalence(
    run_single: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    run_packed: Callable[[dict], torch.Tensor],
    docs: List[Tuple[torch.Tensor, torch.Tensor]],
    pack_rows: List[List[int]],
    atol: float = 1e-5,
) -> float:
    """run_single(src_ids (S,), tgt_in_ids (T,)) -> decoder hidden (T, d)           [one document alone]
       run_packed(batch_dict) -> decoder hidden (B, T, d)  using batch['src'], ['tgt'], ['src_pos'], ['tgt_pos'],
                                 ['enc_mask'], ['dec_mask'], ['cross_mask']             [adapter for YOUR model]
       docs: list of (src_ids, tgt_in_ids); pack_rows: which doc indices go in each row (in order).
    Returns the max abs difference (and asserts it is <= atol)."""
    S = max(sum(len(docs[i][0]) for i in row) for row in pack_rows) + 3
    T = max(sum(len(docs[i][1]) for i in row) for row in pack_rows) + 3
    batch = build_packed_batch([[docs[i][0] for i in r] for r in pack_rows],
                               [[docs[i][1] for i in r] for r in pack_rows], S, T)
    out = run_packed(batch)
    worst = 0.0
    for b, row in enumerate(pack_rows):
        s = 0
        for i in row:
            n = len(docs[i][1])
            ref = run_single(*docs[i])
            worst = max(worst, (out[b, s:s + n] - ref).abs().max().item())
            s += n
    assert worst <= atol, f"packed != unpacked (max abs diff {worst:.3e})"
    return worst


# ------------------------------------------------------------------------------------------------
# Example: ONE forward function used for BOTH R-Drop passes (this is the structural fix for the
# train.py:847 bug).  The second pass must receive exactly the same masks/positions as the first;
# only the dropout realisation differs.  ADAPT kwarg names to your ModernTransformer.
# ------------------------------------------------------------------------------------------------
def packed_forward_hidden(model, batch):
    enc = model.encode(batch["src"], batch["enc_mask"], pos=batch["src_pos"])
    h = model.decode(enc, batch["cross_mask"], batch["tgt_in"], batch["dec_mask"], pos=batch["tgt_pos"])
    return h, batch["tgt_out"], batch.get("weight_tok")          # weight_tok: (B,T) from expand_doc_weights
