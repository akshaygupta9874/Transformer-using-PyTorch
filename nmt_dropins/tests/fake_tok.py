"""Fake tokenizer + substitution used to test the builder with spawn workers."""
import random

class Enc:
    def __init__(self, ids): self.ids = ids

class FakeTok:
    def __init__(self, path): self.path = path
    def encode(self, text, add_bos=False, add_eos=False, dropout_p=0.0, rng=None):
        ids, i = [], 0
        while i < len(text):
            if i + 1 < len(text) and not (dropout_p > 0 and rng.random() < dropout_p):
                ids.append(1000 + (ord(text[i]) * 31 + ord(text[i + 1])) % 30000); i += 2   # "merge"
            else:
                ids.append(4 + ord(text[i]) % 900); i += 1
        if add_bos: ids = [2] + ids
        if add_eos: ids = ids + [3]
        return Enc(ids)

def load(path): return FakeTok(path)
def subst(s, t): return s.replace("1", "#"), t.replace("1", "#")
