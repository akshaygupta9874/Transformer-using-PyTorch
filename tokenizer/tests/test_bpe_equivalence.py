import unittest
from collections import Counter

from tokenizer import BPETrainer


def reference_merges(sequences, vocab_size, min_frequency):
    corpus = [list(sequence) for sequence in sequences]
    vocabulary = set(unit for sequence in corpus for unit in sequence)
    merges = []
    while len(vocabulary) < vocab_size:
        counts = Counter()
        for sequence in corpus:
            for left, right in zip(sequence, sequence[1:]):
                if not BPETrainer._boundary(left) and not BPETrainer._boundary(right):
                    counts[(left, right)] += 1
        candidates = [
            (count, pair)
            for pair, count in counts.items()
            if count >= min_frequency and pair[0] + pair[1] not in vocabulary
        ]
        if not candidates:
            break
        pair = max(candidates, key=lambda item: (item[0], item[1][0], item[1][1]))[1]
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocabulary.add(merged)
        for sequence_index, sequence in enumerate(corpus):
            updated = []
            index = 0
            while index < len(sequence):
                if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                    updated.append(merged)
                    index += 2
                else:
                    updated.append(sequence[index])
                    index += 1
            corpus[sequence_index] = updated
    return merges


class BPEEquivalenceTests(unittest.TestCase):
    def test_merge_order_matches_reference(self):
        sequences = [
            ["a", "b", "a", "b"],
            ["a", "b", "c"],
            ["x", "y", "x", "y"],
            ["कि", "ता", "ब"],
        ]
        expected = reference_merges(sequences, vocab_size=20, min_frequency=1)
        trainer = BPETrainer(vocab_size=20, min_frequency=1).train(sequences)
        actual = trainer.merges
        self.assertEqual(actual, expected)
        for sequence in sequences:
            expected_tokens = list(sequence)
            for pair in expected:
                merged = pair[0] + pair[1]
                updated = []
                index = 0
                while index < len(expected_tokens):
                    if index + 1 < len(expected_tokens) and tuple(expected_tokens[index:index + 2]) == pair:
                        updated.append(merged)
                        index += 2
                    else:
                        updated.append(expected_tokens[index])
                        index += 1
                expected_tokens = updated
            self.assertEqual(trainer.apply(sequence), expected_tokens)


if __name__ == "__main__":
    unittest.main()