from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from .akshara import (
    C_CONS, C_INDV, C_MATRA, C_MOD, C_NUKTA, C_VIRAMA, C_ZW, C_OTHER, C_START,
    char_class,
)
from .normalizer import WORD_MARK

if TYPE_CHECKING:
    from .tokenizer import Tokenizer

_ALLOWED_NEXT = {
    C_CONS:   {C_CONS, C_INDV, C_MATRA, C_MOD, C_NUKTA, C_VIRAMA, C_OTHER, C_START},
    C_INDV:   {C_CONS, C_INDV, C_MOD, C_OTHER, C_START},
    C_MATRA:  {C_CONS, C_INDV, C_MOD, C_OTHER, C_START},
    C_MOD:    {C_CONS, C_INDV, C_MOD, C_OTHER, C_START},
    C_NUKTA:  {C_CONS, C_INDV, C_MATRA, C_MOD, C_VIRAMA, C_OTHER, C_START},
    C_VIRAMA: {C_CONS, C_ZW, C_OTHER, C_START},  # no matra / double virama after virama
    C_ZW:     {C_CONS, C_OTHER, C_START},
    C_OTHER:  {C_CONS, C_INDV, C_OTHER, C_START},
    C_START:  {C_CONS, C_INDV, C_OTHER},         # no orphan matra / virama after whitespace
}
ALLOW: Tuple[Tuple[bool, ...], ...] = tuple(
    tuple(nxt in _ALLOWED_NEXT[prv] for nxt in range(9)) for prv in range(9))


class DevanagariGrammarMask:
    """
    ALLOW[prev_class, next_class] over the class of the last char of the previous
    token and the first char of the candidate. Tokens whose internal char
    sequence violates ALLOW (and <UNK>) are always masked. ▁ counts as START.
    Specials / placeholders / bytes are class OTHER.
    """

    def __init__(self, tok: "Tokenizer"):
        self.tok = tok
        V = len(tok.tokens)
        self.first_cls = [C_OTHER] * V
        self.last_cls = [C_OTHER] * V
        self.valid = [True] * V
        self.valid[tok.token_to_id.get("<UNK>", 1)] = False

        byte_end = getattr(tok, "byte_end", 4)
        for i in range(byte_end, V):
            seq = [C_START if ch == WORD_MARK else char_class(ch) for ch in tok.tokens[i]]
            if seq:
                self.first_cls[i], self.last_cls[i] = seq[0], seq[-1]
                self.valid[i] = all(ALLOW[a][b] for a, b in zip(seq, seq[1:]))

        self._tensors: Dict[str, tuple] = {}

    @staticmethod
    def classify(token: str) -> Tuple[int, int]:
        seq = [C_START if ch == WORD_MARK else char_class(ch) for ch in token]
        if not seq:
            return C_OTHER, C_OTHER
        return seq[0], seq[-1]

    # pure-Python API ------------------------------------------------------------
    def allowed(self, prev_id: Optional[int], next_id: int) -> bool:
        bos_id = self.tok.token_to_id.get("<BOS>", 2)
        prev = C_START if prev_id is None or prev_id == bos_id else self.last_cls[prev_id]
        return self.valid[next_id] and ALLOW[prev][self.first_cls[next_id]]

    def allowed_ids(self, prev_id: Optional[int]) -> List[int]:
        return [i for i in range(len(self.valid)) if self.allowed(prev_id, i)]

    # torch API --------------------------------------------------------------------
    def _get_tensors(self, device):
        import torch  # lazy: tokenizer itself has no dependency on torch
        key = str(device)
        if key not in self._tensors:
            self._tensors[key] = (
                torch.tensor(ALLOW, dtype=torch.bool, device=device),
                torch.tensor(self.first_cls, dtype=torch.long, device=device),
                torch.tensor(self.last_cls, dtype=torch.long, device=device),
                torch.tensor(self.valid, dtype=torch.bool, device=device))
        return self._tensors[key]

    def step_mask(self, prev_ids):
        """prev_ids: (B,) LongTensor of last emitted ids -> (B, V) bool, True = allowed."""
        bos_id = self.tok.token_to_id.get("<BOS>", 2)
        allow, first_cls, last_cls, valid = self._get_tensors(prev_ids.device)
        prev_cls = last_cls[prev_ids]
        prev_cls = prev_cls.masked_fill(prev_ids == bos_id, C_START)
        mask = allow[prev_cls][:, first_cls] & valid.unsqueeze(0)
        return mask | ~mask.any(dim=-1, keepdim=True)  # never mask an entire row

    def apply(self, logits, prev_ids):
        """logits: (B, V). Returns logits with illegal continuations set to -inf."""
        return logits.masked_fill(~self.step_mask(prev_ids), float("-inf"))
