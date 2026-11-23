"""Tokenizer-free UTF-8 control arm for matched-compute ablations."""

from dataclasses import dataclass


@dataclass
class Encoding:
    ids: list[int]


class ByteTokenizer:
    PAD, UNK, BOS, EOS = 256, 257, 258, 259
    VOCAB_SIZE = 260

    def encode(self, text: str) -> Encoding:
        return Encoding(list(text.encode("utf-8")))

    def decode(self, ids) -> str:
        values = bytes(value for value in ids if 0 <= int(value) <= 255)
        return values.decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> int:
        return {"<PAD>": self.PAD, "<UNK>": self.UNK, "<BOS>": self.BOS, "<EOS>": self.EOS}[token]

    def get_vocab_size(self) -> int:
        return self.VOCAB_SIZE
