"""Reproducible research utilities for the English-to-Hindi NMT project.

All modules use the standard library unless an optional integration is invoked.
"""

from research.chunked_losses import (
    _ChunkedLinearCE,
    _ChunkedRDrop,
    chunked_cross_entropy,
    chunked_rdrop_loss,
    chunked_nll_sum,
    chunked_token_logprobs,
)
from research.dynamic_batching import TokenBucketBatchSampler
from research.accum import train_window, windows, window_normalizer, default_forward_hidden
from research.optim_groups import build_param_groups, make_optimizer
from research.logits_utils import (
    step_log_probs,
    greedy_next,
    top_p_top_k_filter,
    sample_from_logits,
    interpolate_knn_log_probs,
)
from research.tie_diagnostics import tied_grad_split
from research.dataset_cache import CachedBilingualDataset
from research.optimization import ModelEMA, CheckpointAverager, MinimumRiskTrainingLoss, mrt_loss
from research.memmap_data import MemmapPairs, MemmapCollate, pack_pairs, packed_masks, PackedMemmapCollate
from research.fast_decoding import (
    KVCache,
    fast_greedy_decode,
    fast_beam_search,
    fast_nucleus_sample,
    fast_mbr_decode,
    sample_batched,
    mbr_select,
)

from research.shared_epoch import SharedEpoch
from research.eval_tools import (
    canon_digits,
    copy_preservation,
    corpus_copy_preservation,
    hash_split,
    evaluate_benchmark,
    write_samples_tsv,
)
from research.profile_sweep import sweep, run_once, synthetic_batch, find_max_tokens
from research.packed_ref import (
    doc_ids_from_lengths,
    positions_from_doc_ids,
    packed_self_mask,
    packed_cross_mask,
    build_packed_batch,
    expand_doc_weights,
    check_packed_equivalence,
    packed_forward_hidden,
)

__all__ = [
    "_ChunkedLinearCE",
    "_ChunkedRDrop",
    "chunked_cross_entropy",
    "chunked_rdrop_loss",
    "chunked_nll_sum",
    "chunked_token_logprobs",
    "TokenBucketBatchSampler",
    "train_window",
    "windows",
    "window_normalizer",
    "default_forward_hidden",
    "build_param_groups",
    "make_optimizer",
    "step_log_probs",
    "greedy_next",
    "top_p_top_k_filter",
    "sample_from_logits",
    "interpolate_knn_log_probs",
    "tied_grad_split",
    "CachedBilingualDataset",
    "ModelEMA",
    "CheckpointAverager",
    "MinimumRiskTrainingLoss",
    "mrt_loss",
    "MemmapPairs",
    "MemmapCollate",
    "pack_pairs",
    "packed_masks",
    "PackedMemmapCollate",
    "KVCache",
    "fast_greedy_decode",
    "fast_beam_search",
    "fast_nucleus_sample",
    "fast_mbr_decode",
    "sample_batched",
    "mbr_select",
    "SharedEpoch",
    "canon_digits",
    "copy_preservation",
    "corpus_copy_preservation",
    "hash_split",
    "evaluate_benchmark",
    "write_samples_tsv",
    "sweep",
    "run_once",
    "synthetic_batch",
    "find_max_tokens",
    "doc_ids_from_lengths",
    "positions_from_doc_ids",
    "packed_self_mask",
    "packed_cross_mask",
    "build_packed_batch",
    "expand_doc_weights",
    "check_packed_equivalence",
    "packed_forward_hidden",
]

