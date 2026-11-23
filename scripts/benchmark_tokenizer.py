import argparse
import time

from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenizer training and encoding")
    parser.add_argument("--samples", type=int, default=10000)
    arguments = parser.parse_args()
    source = "This is a small English sentence."
    target = "यह एक छोटा हिंदी वाक्य है।"
    texts = (text for _ in range(arguments.samples) for text in (source, target))
    started = time.perf_counter()
    tokenizer = Tokenizer.train(texts, language="en-hi", vocab_size=50000, min_frequency=2)
    train_seconds = time.perf_counter() - started
    sample = source + " " + target
    started = time.perf_counter()
    encoded = tokenizer.encode(sample)
    encode_seconds = time.perf_counter() - started
    print(f"samples={arguments.samples:,} pairs")
    print(f"vocabulary={tokenizer.get_vocab_size():,}")
    print(f"merges={len(tokenizer.merges):,}")
    print(f"training_seconds={train_seconds:.3f}")
    print(f"encode_tokens={len(encoded.ids):,}")
    print(f"encode_seconds={encode_seconds:.6f}")


if __name__ == "__main__":
    main()