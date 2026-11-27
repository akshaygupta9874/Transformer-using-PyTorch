from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path, get_tokenizer_path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import shutil
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizer import Tokenizer, TokenizerValidationError, validate_tokenizer_metadata

import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from research.reproducibility import OptionalWandb, seed_everything, write_run_manifest

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('<BOS>')
    eos_idx = tokenizer_tgt.token_to_id('<EOS>')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, loss_fn, use_amp, amp_dtype, num_examples=2, use_grammar_mask=True, use_chunked_ce=False, ce_chunk_size=1024, pad_id=0, label_smoothing=0.1):
    model.eval()
    print_msg('Starting validation...')

    grammar_mask = None
    if use_grammar_mask and hasattr(tokenizer_tgt, "tokens"):
        try:
            from tokenizer.grammar_mask import DevanagariGrammarMask
            grammar_mask = DevanagariGrammarMask(tokenizer_tgt)
        except Exception:
            grammar_mask = None

    use_fast_kv = hasattr(model, "rope")
    if use_fast_kv:
        from research.fast_decoding import fast_greedy_decode
    if use_chunked_ce:
        from research.optimization import chunked_cross_entropy

    validation_loss_total = 0.0
    validation_batches = 0
    with torch.no_grad():
        for batch in validation_ds:
            encoder_input = batch['encoder_input'].to(device, non_blocking=True)
            decoder_input = batch['decoder_input'].to(device, non_blocking=True)
            encoder_mask = batch['encoder_mask'].to(device, non_blocking=True)
            decoder_mask = batch['decoder_mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                if use_chunked_ce and hasattr(model, "proj") and getattr(model.proj, "bias", None) is None:
                    batch_loss = chunked_cross_entropy(
                        decoder_output,
                        model.proj.weight,
                        label,
                        label_smoothing=label_smoothing,
                        ignore_index=pad_id,
                        chunk_size=ce_chunk_size,
                    )
                else:
                    proj_output = model.project_logits(decoder_output) if hasattr(model, "project_logits") else model.project(decoder_output)
                    batch_loss = loss_fn(
                        proj_output.reshape(-1, tokenizer_tgt.get_vocab_size()),
                        label.reshape(-1),
                    )

            validation_loss_total += batch_loss.item()
            validation_batches += 1

    validation_loss = validation_loss_total / max(validation_batches, 1)
    writer.add_scalar('validation loss', validation_loss, global_step)
    print_msg(f'Validation loss: {validation_loss:.4f}')

    count = 0

    source_texts = []
    expected = []
    predicted = []

    console_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    sos_id = tokenizer_tgt.token_to_id('<BOS>')
    eos_id = tokenizer_tgt.token_to_id('<EOS>')

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            if use_fast_kv:
                model_out = fast_greedy_decode(
                    model,
                    encoder_input,
                    encoder_mask,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    max_len=max_len,
                    device=device,
                    grammar_mask=grammar_mask,
                )
            else:
                model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device).detach().cpu().numpy()

            source_text = batch["src_text"][0] if "src_text" in batch else tokenizer_src.decode(encoder_input[0].detach().cpu().tolist())
            target_text = batch["tgt_text"][0] if "tgt_text" in batch else tokenizer_tgt.decode(batch["label"][0].detach().cpu().tolist())
            model_out_text = tokenizer_tgt.decode(model_out)

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        # Evaluate the character error rate
        # Compute the char error rate 
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer.item(), global_step)

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer.item(), global_step)

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu.item(), global_step)

        try:
            from research.eval_tools import corpus_copy_preservation
            cp_stats = corpus_copy_preservation(source_texts, predicted)
            if not math.isnan(cp_stats.get("copy_rate", float("nan"))):
                writer.add_scalar('validation copy_rate', cp_stats["copy_rate"], global_step)
                print_msg(f"Validation copy preservation rate: {cp_stats['copy_rate']*100:.1f}% ({cp_stats['items']} items tracked)")
        except Exception:
            pass
        writer.flush()
        print_msg('Validation metrics logged.')

def get_all_sentences(ds, lang):
    for item in ds:
        if 'translation' in item:
            yield item['translation'][lang]
        elif lang == 'en':
            yield item['src']
        elif lang == 'hi':
            yield item['tgt']
        else:
            raise KeyError(f"Unsupported dataset columns: {list(item)}")

def get_parallel_texts(ds, config):
    for item in ds:
        if 'translation' in item:
            yield item['translation'][config['lang_src']]
            yield item['translation'][config['lang_tgt']]
        else:
            yield item['src']
            yield item['tgt']

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = get_tokenizer_path(config, lang)
    if Path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
        is_valid, reasons = validate_tokenizer_metadata(
            tokenizer,
            config=config,
            expected_lang=lang,
            tokenizer_path=tokenizer_path,
            strict=config.get("strict_tokenizer_validation", False),
            raise_on_error=False,
        )
        if is_valid:
            print(f"Validated existing tokenizer at {tokenizer_path} for {lang} ({tokenizer.get_vocab_size():,} tokens).")
            return tokenizer

        # If the only reason is a vocabulary size difference from a validly trained tokenizer on the matching corpus
        if (
            not config.get("strict_tokenizer_validation", False)
            and all("Vocabulary size mismatch" in r for r in reasons)
            and getattr(tokenizer, "training_corpus_identity", None) in (config.get("datasource", "samanantar"), "ai4bharat/samanantar", "samanantar")
        ):
            print(f"Adopting existing {lang} tokenizer at {tokenizer_path} ({tokenizer.get_vocab_size():,} tokens, trained on {getattr(tokenizer, 'training_pair_count', 'N/A'):,} pairs).")
            if "tokenizer_vocab_sizes" in config:
                config["tokenizer_vocab_sizes"][lang] = tokenizer.get_vocab_size()
            return tokenizer

        warning_lines = "\n".join(f"  - {r}" for r in reasons)
        if config.get("retrain_invalid_tokenizer", False):
            print(f"Warning: Stale/invalid tokenizer at {tokenizer_path} will be rebuilt:\n{warning_lines}")
        else:
            raise TokenizerValidationError(
                f"Existing tokenizer at '{tokenizer_path}' failed validation against experiment configuration:\n"
                f"{warning_lines}\n\n"
                f"A stale tokenizer cannot be silently reused. To automatically retrain, set config['retrain_invalid_tokenizer'] = True."
            )

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    ds_len = len(ds) if hasattr(ds, "__len__") else 0
    if config.get('shared_vocabulary'):
        tokenizer = Tokenizer.train(
            get_parallel_texts(ds, config),
            language='en-hi',
            vocab_size=config.get('tokenizer_vocab_size', 50000),
            min_frequency=2,
            training_pair_count=ds_len,
            corpus_identity=config.get("datasource", "samanantar"),
            corpus_revision=config.get("dataset_config", "v0.2"),
        )
    else:
        vocab_size = config.get('tokenizer_vocab_sizes', {}).get(lang, 16000 if lang == 'en' else 40000)
        tokenizer = Tokenizer.train(
            get_all_sentences(ds, lang),
            language=lang,
            vocab_size=vocab_size,
            min_frequency=2,
            training_pair_count=ds_len,
            corpus_identity=config.get("datasource", "samanantar"),
            corpus_revision=config.get("dataset_config", "v0.2"),
        )
    tokenizer.save(tokenizer_path)
    return tokenizer

def _load_parallel_records(config):
    """Load training pairs from a cleaned TSV, local HF dataset dir, or HuggingFace hub."""
    train_tsv = config.get("train_tsv")
    datasource = config.get("datasource", "ai4bharat/samanantar")
    if not train_tsv and isinstance(datasource, str) and datasource.lower().endswith(".tsv"):
        train_tsv = datasource

    if train_tsv:
        import csv
        tsv_path = Path(train_tsv)
        if not tsv_path.exists():
            raise FileNotFoundError(f"Configured train_tsv not found: {tsv_path}")
        records = []
        with tsv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    records.append({"src": row[0], "tgt": row[1]})
        print(f"Loaded {len(records):,} pre-filtered parallel pairs from {tsv_path}")
        return records

    ds_path = Path(str(datasource))
    if ds_path.exists() and ds_path.is_dir():
        from datasets import load_from_disk
        print(f"Loading cached HuggingFace dataset from {ds_path}...")
        return load_from_disk(str(ds_path))

    dataset_kwargs = {"split": "train"}
    if config.get("dataset_config"):
        dataset_kwargs["name"] = config["dataset_config"]
    return load_dataset(datasource, **dataset_kwargs)


def _estimate_pair_lengths(ds_raw, tokenizer_src, tokenizer_tgt, lang_src: str, lang_tgt: str, seq_len: int) -> list[int]:
    """Compute exact token lengths from dataset metadata/memmap or exact tokenizer encoding."""
    if hasattr(ds_raw, "lengths") and callable(getattr(ds_raw, "lengths")):
        return ds_raw.lengths()
    if hasattr(ds_raw, "lengths") and not callable(getattr(ds_raw, "lengths")):
        return list(ds_raw.lengths)
    if hasattr(ds_raw, "pair_lengths") and callable(getattr(ds_raw, "pair_lengths")):
        sl, tl = ds_raw.pair_lengths()
        return np.maximum(sl + 2, tl + 1).tolist()

    n = len(ds_raw)
    lengths: list[int] = []
    for i in range(n):
        item = ds_raw[i]
        if "translation" in item:
            s_txt = item["translation"][lang_src]
            t_txt = item["translation"][lang_tgt]
        else:
            s_txt = item["src"]
            t_txt = item["tgt"]
        s_len = len(tokenizer_src.encode(s_txt).ids) + 2
        t_len = len(tokenizer_tgt.encode(t_txt).ids) + 1
        lengths.append(min(seq_len, max(s_len, t_len)))
    return lengths


def _get_memmap_ds(config):
    """Build DataLoaders backed by pre-tokenized uint16 MemmapPairs (zero epoch tokenization)."""
    from research.memmap_data import MemmapPairs, MemmapCollate, PackedMemmapCollate

    tokenizer_src = Tokenizer.load(get_tokenizer_path(config, config['lang_src']))
    tokenizer_tgt = tokenizer_src if config.get('shared_vocabulary') else Tokenizer.load(get_tokenizer_path(config, config['lang_tgt']))

    memmap_dir = Path(config["memmap_dir"])
    train_split = config.get("memmap_train_split", "train")
    full_train = MemmapPairs(
        memmap_dir,
        split=train_split,
        n_src_variants=int(config.get("memmap_src_variants", 0)),
        dropout_frac=float(config.get("memmap_dropout_frac", 0.3)),
        seed=int(config.get("seed", 13)),
    )

    if config.get("max_train_samples"):
        max_s = min(int(config["max_train_samples"]), len(full_train))
        if max_s < len(full_train):
            print(f"Limiting memmap dataset to {max_s:,} samples (config max_train_samples)...")
            from torch.utils.data import Subset
            sliced_lengths = full_train.lengths()[:max_s]
            full_train = Subset(full_train, range(max_s))
            full_train.lengths = lambda: sliced_lengths

    val_off = memmap_dir / "val.src.off.npy"
    if val_off.exists():
        train_ds = full_train
        val_ds = MemmapPairs(memmap_dir, split="val", n_src_variants=0, seed=int(config.get("seed", 13)))
        all_lengths = full_train.lengths()
    else:
        val_size_limit = int(config.get("val_size_limit", 2000))
        val_size = min(int(0.1 * len(full_train)), val_size_limit)
        val_size = max(1, val_size) if len(full_train) > 1 else 0
        train_size = len(full_train) - val_size
        train_ds, val_ds = random_split(full_train, [train_size, val_size])
        raw_lengths = full_train.lengths()
        all_lengths = [raw_lengths[i] for i in train_ds.indices]

    if not config.get("skip_benchmark_isolation", False):
        try:
            from scripts.fetch_benchmarks import assert_no_benchmark_leakage
        except ImportError:
            from fetch_benchmarks import assert_no_benchmark_leakage
        check_limit = min(int(config.get("leakage_check_samples", 5000)), len(train_ds))
        sample_check = [
            (tokenizer_src.decode(train_ds[i][0].tolist()), tokenizer_tgt.decode(train_ds[i][1].tolist()))
            for i in range(check_limit)
        ]
        assert_no_benchmark_leakage(sample_check, benchmark_dir=Path(config.get("benchmark_dir", "data/benchmarks")))

    pad_id_src = tokenizer_src.token_to_id("<PAD>")
    pad_id_tgt = tokenizer_tgt.token_to_id("<PAD>")
    bos_id_tgt = tokenizer_tgt.token_to_id("<BOS>")
    eos_id_tgt = tokenizer_tgt.token_to_id("<EOS>")
    val_collate = MemmapCollate(pad_id=pad_id_tgt, bos_id=bos_id_tgt, eos_id=eos_id_tgt)

    if config.get("use_packed_batches", False):
        train_collate = PackedMemmapCollate(
            src_cap=int(config.get("pack_src_cap", config.get("seq_len", 350))),
            tgt_cap=int(config.get("pack_tgt_cap", config.get("seq_len", 350))),
        )
    else:
        train_collate = val_collate

    loader_options = {
        "batch_size": config['batch_size'],
        "num_workers": config['num_workers'],
        "pin_memory": True,
        "persistent_workers": config['num_workers'] > 0,
        "collate_fn": train_collate,
    }
    if config['num_workers'] > 0:
        loader_options['prefetch_factor'] = config['prefetch_factor']

    if config.get("use_token_bucket_batching", False):
        from research.dynamic_batching import TokenBucketBatchSampler
        max_tokens = int(config.get("max_tokens_per_batch", 4096))
        batch_sampler = TokenBucketBatchSampler(
            all_lengths,
            max_tokens=max_tokens,
            shuffle=True,
            seed=config.get('seed', 13),
        )
        loader_options.pop("batch_size", None)
        loader_options["batch_sampler"] = batch_sampler
        print(f"Memmap + TokenBucketBatchSampler ready (pairs={len(train_ds):,}, batches/epoch={len(batch_sampler):,}).")
    else:
        loader_options["shuffle"] = True

    train_dataloader = DataLoader(train_ds, **loader_options)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=val_collate)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, None


def get_ds(config):
    if config.get("memmap_dir"):
        memmap_path = Path(config["memmap_dir"])
        if memmap_path.exists() and (memmap_path / "src.bin").exists():
            return _get_memmap_ds(config)
        else:
            print(f"Notice: memmap_dir '{config['memmap_dir']}' not found on disk. Falling back to standard dataset loading.")
    ds_raw = _load_parallel_records(config)

    if config.get("max_train_samples"):
        max_samples = min(int(config["max_train_samples"]), len(ds_raw))
        print(f"Limiting dataset to {max_samples:,} samples (config max_train_samples)...")
        if hasattr(ds_raw, "select"):
            ds_raw = ds_raw.select(range(max_samples))
        else:
            ds_raw = ds_raw[:max_samples]

    if config.get("clean_training_data", False):
        from research.data import ParallelCleaner, pair_fingerprint
        cleaner = ParallelCleaner(
            min_chars=config.get("clean_min_chars", 2),
            max_chars=config.get("clean_max_chars", 1000),
            max_length_ratio=config.get("clean_max_length_ratio", 3.0),
        )
        cleaned = []
        seen = set()
        for i in range(len(ds_raw)):
            item = ds_raw[i]
            s_raw = item["translation"][config["lang_src"]] if "translation" in item else item["src"]
            t_raw = item["translation"][config["lang_tgt"]] if "translation" in item else item["tgt"]
            s_norm, t_norm = cleaner.normalize(s_raw), cleaner.normalize(t_raw)
            if cleaner.reason_to_reject(s_norm, t_norm) is None:
                fp = pair_fingerprint(s_norm, t_norm)
                if fp not in seen:
                    seen.add(fp)
                    cleaned.append({"src": s_norm, "tgt": t_norm})
        print(f"Inline data cleaning retained {len(cleaned):,} / {len(ds_raw):,} pairs.")
        ds_raw = cleaned

    # Split: content-hash split based on source text prevents validation drift across corpus sizes
    raw_val_limit = config.get("val_size_limit", 2000)
    val_size_limit = min(raw_val_limit, max(1, int(0.2 * len(ds_raw)))) if len(ds_raw) > 1 else 0
    if config.get("use_hash_split", True) and len(ds_raw) > 1:
        from research.eval_tools import hash_split
        train_ds_raw, val_ds_raw = [], []
        permille = max(1, min(100, int(1000 * val_size_limit / max(1, len(ds_raw)))))
        for item in ds_raw:
            s_raw = item["translation"][config["lang_src"]] if "translation" in item else item["src"]
            if len(val_ds_raw) < val_size_limit and hash_split(s_raw, val_permille=permille):
                val_ds_raw.append(item)
            else:
                train_ds_raw.append(item)
        if not val_ds_raw and len(train_ds_raw) > 1:
            val_ds_raw.append(train_ds_raw.pop())
        elif not train_ds_raw and len(val_ds_raw) > 0:
            train_ds_raw.append(val_ds_raw.pop())
    else:
        val_ds_size = min(int(0.1 * len(ds_raw)), val_size_limit)
        val_ds_size = max(1, val_ds_size) if len(ds_raw) > 1 else 0
        train_ds_size = len(ds_raw) - val_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # Build vocabularies from training text only; validation remains held out.
    tokenizer_src = get_or_build_tokenizer(config, train_ds_raw, config['lang_src'])
    tokenizer_tgt = tokenizer_src if config.get('shared_vocabulary') else get_or_build_tokenizer(config, train_ds_raw, config['lang_tgt'])

    dynamic_padding = config.get("dynamic_padding", True)
    pad_to_max = not dynamic_padding

    train_ds = BilingualDataset(
        train_ds_raw, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len'],
        config.get('use_placeholders', False),
        pad_to_max=pad_to_max,
    )
    val_ds = BilingualDataset(
        val_ds_raw, tokenizer_src, tokenizer_tgt,
        config['lang_src'], config['lang_tgt'], config['seq_len'],
        config.get('use_placeholders', False),
        pad_to_max=pad_to_max,
    )

    # Sample up to 1,000 sentences to check length statistics quickly without stalling on 10M datasets
    sample_len_count = min(1000, len(ds_raw))
    max_len_src = 0
    max_len_tgt = 0

    for i in range(sample_len_count):
        item = ds_raw[i]
        if 'translation' in item:
            source_text = item['translation'][config['lang_src']]
            target_text = item['translation'][config['lang_tgt']]
        else:
            source_text = item['src']
            target_text = item['tgt']
        src_ids = tokenizer_src.encode(source_text).ids
        tgt_ids = tokenizer_tgt.encode(target_text).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length (sample of {sample_len_count:,}): source={max_len_src}, target={max_len_tgt}')

    loader_options = {
        "batch_size": config['batch_size'],
        "num_workers": config['num_workers'],
        "pin_memory": True,
        "persistent_workers": config['num_workers'] > 0,
    }
    if config['num_workers'] > 0:
        loader_options['prefetch_factor'] = config['prefetch_factor']

    if dynamic_padding:
        from dataset import PadCollate
        pad_collate = PadCollate(
            pad_token_src=tokenizer_src.token_to_id('<PAD>'),
            pad_token_tgt=tokenizer_tgt.token_to_id('<PAD>'),
        )
        loader_options["collate_fn"] = pad_collate

    # Enforce hard benchmark isolation check: contamination must halt training immediately
    if not config.get("skip_benchmark_isolation", False):
        try:
            from scripts.fetch_benchmarks import assert_no_benchmark_leakage
        except ImportError:
            try:
                from fetch_benchmarks import assert_no_benchmark_leakage
            except ImportError as err:
                raise RuntimeError(
                    "CRITICAL: Failed to import assert_no_benchmark_leakage to verify benchmark isolation."
                ) from err

        check_limit = min(int(config.get("leakage_check_samples", 5000)), len(train_ds))

        sample_check = [
            (train_ds[i]["src_text"], train_ds[i]["tgt_text"])
            for i in range(check_limit)
        ]
        benchmark_dir = Path(config.get("benchmark_dir", "data/benchmarks"))
        print(f"Enforcing benchmark isolation check across {len(sample_check):,} training pairs against {benchmark_dir}...")
        assert_no_benchmark_leakage(sample_check, benchmark_dir=benchmark_dir)
        print("✓ Benchmark isolation verified: zero contamination detected against frozen evaluation sets.")

    # B13: Curriculum learning sampler
    curriculum_config = config.get('curriculum') or {}
    curriculum_enabled = curriculum_config.get('enabled', False) if isinstance(curriculum_config, dict) else config.get('curriculum_enabled', False)
    curriculum_sampler = None

    if curriculum_enabled:
        from research.curriculum import CurriculumSampler, score_dataset_difficulty
        print("Curriculum learning enabled: scoring dataset difficulty...")
        warmup_frac = curriculum_config.get('warmup_fraction', 0.25) if isinstance(curriculum_config, dict) else 0.25
        diff_metric = curriculum_config.get('metric', 'length') if isinstance(curriculum_config, dict) else 'length'
        scores = score_dataset_difficulty(train_ds_raw, tokenizer_src, tokenizer_tgt, difficulty_metric=diff_metric, max_seq_len=config['seq_len'])
        total_steps = config.get('num_epochs', 20) * math.ceil(max(1, len(train_ds)) / config['batch_size'])
        curriculum_sampler = CurriculumSampler(scores, warmup_fraction=warmup_frac, total_steps=total_steps, seed=config.get('seed', 13))
        loader_options["sampler"] = curriculum_sampler
        loader_options["shuffle"] = False
    elif dynamic_padding and config.get("use_token_bucket_batching", False):
        from research.dynamic_batching import TokenBucketBatchSampler
        max_tokens = int(config.get("max_tokens_per_batch", 4096))
        lengths = _estimate_pair_lengths(
            train_ds_raw, tokenizer_src, tokenizer_tgt,
            config['lang_src'], config['lang_tgt'], config['seq_len'],
        )
        batch_sampler = TokenBucketBatchSampler(
            lengths,
            max_tokens=max_tokens,
            shuffle=True,
            seed=config.get('seed', 13),
        )
        loader_options.pop("batch_size", None)
        loader_options.pop("shuffle", None)
        loader_options.pop("sampler", None)
        loader_options["batch_sampler"] = batch_sampler
        print(f"TokenBucketBatchSampler enabled (max_tokens={max_tokens}, batches/epoch={len(batch_sampler):,}).")
    else:
        loader_options["shuffle"] = True

    train_dataloader = DataLoader(train_ds, **loader_options)
    val_loader_options = {"batch_size": 1, "shuffle": False}
    if dynamic_padding:
        val_loader_options["collate_fn"] = pad_collate
    val_dataloader = DataLoader(val_ds, **val_loader_options)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, curriculum_sampler

def get_model(config, vocab_src_len, vocab_tgt_len):
    if config.get("architecture") == "modern_transformer":
        from modern_model import build_modern_transformer
        return build_modern_transformer(
            vocab_src_len, vocab_tgt_len,
            d_model=config['d_model'],
            encoder_layers=config.get('encoder_layers', 6),
            decoder_layers=config.get('decoder_layers', 4),
            heads=config.get('heads', 8),
            d_ff=config.get('d_ff', 2048),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 2048),
            use_gradient_checkpointing=config.get('use_gradient_checkpointing', False),
            use_qk_norm=config.get('use_qk_norm', True),
            tie_embeddings=config.get('tie_embeddings', False),
            kv_heads=config.get('kv_heads', None),
            encoder_d_ff=config.get('encoder_d_ff', None),
        )
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
        d_model=config['d_model'], h=config.get('heads', 8), dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 2048), encoder_layers=config.get('encoder_layers', 6),
        decoder_layers=config.get('decoder_layers', 4),
    )
    return model


def compute_effective_warmup_steps(configured_warmup: int, total_steps: int, ratio: float = 0.06) -> int:
    """Scale warmup_steps relative to total_steps so LR actually completes warmup and decays."""
    if total_steps <= 1:
        return 1
    target = max(1, int(round(ratio * total_steps)))
    if total_steps >= 1700:
        target = max(100, target)
    return max(1, min(int(configured_warmup), target, total_steps - 1))


def build_optimizer_param_groups(model: nn.Module, weight_decay: float, decay_embeddings: bool = False) -> list[dict]:
    """Separate trainable parameters into decay (>=2D weights, non-embeddings) and no-decay (norms, biases, 1D, and embeddings unless decay_embeddings)."""
    try:
        from research.optim_groups import build_param_groups
        return build_param_groups(model, weight_decay=weight_decay, decay_embeddings=decay_embeddings)
    except Exception:
        emb_ids = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
        decay_params, no_decay_params, seen_ids = [], [], set()
        for name, param in model.named_parameters():
            if not param.requires_grad or id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            if param.ndim < 2 or (id(param) in emb_ids and not decay_embeddings):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [{"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0}]

def validate_checkpoint_compatibility(
    state: dict,
    config: dict,
    vocab_src_size: int,
    vocab_tgt_size: int,
    checkpoint_path: str | Path = "",
) -> None:
    """Verify saved checkpoint architecture and vocabulary sizes match the current run."""
    ckpt_cfg = state.get("config") or {}
    ckpt_arch = ckpt_cfg.get("architecture")
    ckpt_tok = state.get("tokenizer_metadata") or {}
    ckpt_src_vocab = ckpt_tok.get("vocab_src_size")
    ckpt_tgt_vocab = ckpt_tok.get("vocab_tgt_size")

    # Infer metadata from weight shapes/keys when loading legacy checkpoints without metadata
    model_sd = state.get("model_state_dict") or {}
    if ckpt_src_vocab is None:
        src_emb = model_sd.get("src_embed.embedding.weight") if "src_embed.embedding.weight" in model_sd else model_sd.get("src_embed.weight")
        if src_emb is not None:
            ckpt_src_vocab = int(src_emb.shape[0])
    if ckpt_tgt_vocab is None:
        tgt_emb = model_sd.get("tgt_embed.embedding.weight") if "tgt_embed.embedding.weight" in model_sd else model_sd.get("tgt_embed.weight")
        if tgt_emb is not None:
            ckpt_tgt_vocab = int(tgt_emb.shape[0])
    if ckpt_arch is None and model_sd:
        ckpt_arch = "modern_transformer" if any(k.startswith("rope.") or k.startswith("encoder_blocks.") for k in model_sd) else "vanilla"

    cur_arch = config.get("architecture", "vanilla")
    arch_mismatch = ckpt_arch is not None and ckpt_arch != cur_arch
    src_mismatch = ckpt_src_vocab is not None and int(ckpt_src_vocab) != int(vocab_src_size)
    tgt_mismatch = ckpt_tgt_vocab is not None and int(ckpt_tgt_vocab) != int(vocab_tgt_size)

    if arch_mismatch or src_mismatch or tgt_mismatch:
        ckpt_dir = Path(checkpoint_path).parent if checkpoint_path else "weights directory"
        raise RuntimeError(
            f"Incompatible checkpoint '{checkpoint_path}' for preload:\n"
            f"  - Saved checkpoint: architecture={ckpt_arch!r}, vocab=({ckpt_src_vocab}, {ckpt_tgt_vocab}), epoch={state.get('epoch')}\n"
            f"  - Current run:      architecture={cur_arch!r}, vocab=({vocab_src_size}, {vocab_tgt_size})\n\n"
            f"If you rebuilt the tokenizer or changed model architecture, either:\n"
            f"  1) Use a separate weights directory: add `model_folder=modern_weights` to your command, OR\n"
            f"  2) Remove or move old .pt files from '{ckpt_dir}' before training."
        )


def train_model(config):
    seed_everything(config.get('seed', 13))
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(device)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 3} GB")
        torch.set_float32_matmul_precision('high')
    elif device.type == 'mps':
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    use_amp = config["use_amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if config["amp_dtype"] == "bfloat16" else torch.float16

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, curriculum_sampler = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    if config.get("compile_model", False) and hasattr(torch, "compile") and device.type == "cuda":
        print("Compiling model with torch.compile for maximum GPU throughput...")
        model = torch.compile(model)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    write_run_manifest(config['experiment_name'], "from_scratch", config, config.get('seed', 13))
    tracker = OptionalWandb(config.get('wandb_project', 'en-hi-nmt'), config, config.get('wandb_enabled', False))

    weight_decay = float(config.get('weight_decay', 0.01))
    decay_embeddings = bool(config.get('decay_embeddings', False))
    param_groups = build_optimizer_param_groups(model, weight_decay, decay_embeddings)
    optimizer_kwargs = {"lr": config['lr'], "eps": 1e-9}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(param_groups, **optimizer_kwargs)
    configured_warmup = max(1, int(config.get('warmup_steps', 4000)))
    accum_steps = max(1, config.get('gradient_accumulation_steps', 1))
    batches_per_epoch = len(train_dataloader)
    if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "expected_len"):
        batches_per_epoch = train_dataloader.batch_sampler.expected_len
    total_steps = config['num_epochs'] * math.ceil(max(1, batches_per_epoch) / accum_steps)
    if config.get('auto_scale_warmup', True):
        warmup_steps = compute_effective_warmup_steps(configured_warmup, total_steps)
        if warmup_steps != configured_warmup:
            print(f"Auto-scaled warmup_steps: {warmup_steps:,} (from configured {configured_warmup:,} for {total_steps:,} total optimizer steps).")
    else:
        warmup_steps = configured_warmup
    if config.get('lr_schedule') == 'cosine':
        from research.optimization import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = LambdaLR(optimizer, lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5))

    ema = None
    if config.get('use_ema', False):
        from research.optimization import ModelEMA
        ema = ModelEMA(model, decay=config.get('ema_decay', 0.9999), warmup_ramp=True)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config.get('preload')
    if isinstance(preload, str) and preload.strip().lower() in ('', 'none', 'null', 'false'):
        preload = None
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=device, weights_only=False)
        validate_checkpoint_compatibility(
            state,
            config,
            vocab_src_size=tokenizer_src.get_vocab_size(),
            vocab_tgt_size=tokenizer_tgt.get_vocab_size(),
            checkpoint_path=model_filename,
        )
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if 'scheduler_state_dict' in state and state['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'ema_state_dict' in state and state['ema_state_dict'] is not None and ema is not None:
            ema_updates = state.get('ema_num_updates')
            if ema_updates is None:
                ema_updates = global_step // accum_steps
            ema.load_state_dict(state['ema_state_dict'], num_updates=ema_updates)
    else:
        print('No model to preload, starting from scratch')

    effective_batch_size = config['batch_size'] * config.get('gradient_accumulation_steps', 1)
    print(f"Batch size: {config['batch_size']}, Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}, Effective batch size: {effective_batch_size}")

    pad_id = tokenizer_tgt.token_to_id('<PAD>')
    if config.get('token_weighting') in ('entity', 'frequency'):
        from research.optimization import TargetAwareLoss
        entity_ids = [
            tokenizer_tgt.token_to_id(tok)
            for tok in getattr(tokenizer_tgt, "tokens", [])
            if tok.startswith("⟨") and tok.endswith("⟩")
        ]
        loss_fn = TargetAwareLoss(
            vocab_size=tokenizer_tgt.get_vocab_size(),
            pad_id=pad_id,
            label_smoothing=config.get('label_smoothing', 0.1),
            entity_token_ids=entity_ids if config.get('token_weighting') == 'entity' else None,
            entity_weight_multiplier=config.get('entity_weight_multiplier', 1.5),
        ).to(device)
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config.get('label_smoothing', 0.1)).to(device)

    use_chunked_ce = (
        config.get('use_chunked_ce', True)
        and hasattr(model, "proj")
        and getattr(model.proj, "bias", None) is None
        and config.get('token_weighting') not in ('entity', 'frequency')
    )
    ce_chunk_size = int(config.get('ce_chunk_size', 1024))
    z_loss = float(config.get('z_loss', 1e-4))
    rdrop_alpha = float(config.get('rdrop_alpha', 0.0))
    if use_chunked_ce:
        from research.chunked_losses import chunked_cross_entropy, chunked_rdrop_loss

    for epoch in range(initial_epoch, config['num_epochs']):
        ds_obj = train_dataloader.dataset
        base_ds = getattr(ds_obj, "dataset", ds_obj)
        if hasattr(base_ds, "set_epoch"):
            base_ds.set_epoch(epoch)
        if hasattr(train_dataloader, "batch_sampler") and hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch_index, batch in enumerate(batch_iterator):

            encoder_input = batch['encoder_input'].to(device,non_blocking=True) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device,non_blocking=True) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device,non_blocking=True) # (B, 1, 1, seq_len) or packed
            decoder_mask = batch['decoder_mask'].to(device,non_blocking=True) # (B, 1, seq_len, seq_len) or packed
            cross_mask = batch['cross_mask'].to(device, non_blocking=True) if 'cross_mask' in batch else encoder_mask
            src_pos = batch['src_pos'].to(device, non_blocking=True) if 'src_pos' in batch else None
            tgt_pos = batch['tgt_pos'].to(device, non_blocking=True) if 'tgt_pos' in batch else None
            sample_weight = batch['sample_weight'].to(device, non_blocking=True) if 'sample_weight' in batch else None

            label = batch['label'].to(device) # (B, seq_len)

            if global_step == 0 and batch_index == 0 and hasattr(model, "tgt_embed") and hasattr(model, "proj") and model.tgt_embed.weight is model.proj.weight:
                try:
                    from research.tie_diagnostics import tied_grad_split
                    diag = tied_grad_split(model, batch, device=device)
                    print(f"\n[Weight Tying Diagnostics] step {global_step}: ||g_emb||={diag['norm_emb_path']:.4e}, ||g_proj||={diag['norm_proj_path']:.4e}, ratio={diag['ratio_emb_over_proj']:.2f}, cosine={diag['cosine']:.4f}\n")
                except Exception as ex:
                    print(f"[Weight Tying Diagnostics] skipped: {ex}")

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                if src_pos is not None:
                    encoder_output = model.encode(encoder_input, encoder_mask, pos=src_pos)
                    decoder_output = model.decode(encoder_output, cross_mask, decoder_input, decoder_mask, pos=tgt_pos)
                else:
                    encoder_output = model.encode(encoder_input, encoder_mask)
                    decoder_output = model.decode(encoder_output, cross_mask, decoder_input, decoder_mask)

                if use_chunked_ce:
                    if rdrop_alpha > 0.0:
                        if src_pos is not None:
                            second_enc = model.encode(encoder_input, encoder_mask, pos=src_pos)
                            second_dec = model.decode(second_enc, cross_mask, decoder_input, decoder_mask, pos=tgt_pos)
                        else:
                            second_enc = model.encode(encoder_input, encoder_mask)
                            second_dec = model.decode(second_enc, cross_mask, decoder_input, decoder_mask)
                        loss = chunked_rdrop_loss(
                            decoder_output,
                            second_dec,
                            model.proj.weight,
                            label,
                            label_smoothing=config.get('label_smoothing', 0.1),
                            rdrop_alpha=rdrop_alpha,
                            ignore_index=pad_id,
                            sample_weight=sample_weight,
                            chunk_size=min(ce_chunk_size, 256),
                            z_loss=z_loss,
                        )
                    else:
                        loss = chunked_cross_entropy(
                            decoder_output,
                            model.proj.weight,
                            label,
                            label_smoothing=config.get('label_smoothing', 0.1),
                            ignore_index=pad_id,
                            sample_weight=sample_weight,
                            chunk_size=ce_chunk_size,
                            z_loss=z_loss,
                        )
                else:
                    use_raw_logits = hasattr(model, "project_logits")
                    proj_output = model.project_logits(decoder_output) if use_raw_logits else model.project(decoder_output)
                    loss = loss_fn(
                        proj_output.reshape(-1, tokenizer_tgt.get_vocab_size()),
                        label.reshape(-1),
                    )
                    if config.get('rdrop_alpha', 0.0) > 0:
                        # A second dropout realization should predict the same
                        # distribution under the same teacher-forced inputs.
                        second_dec = model.decode(model.encode(encoder_input, encoder_mask), encoder_mask, decoder_input, decoder_mask)
                        second_output = model.project_logits(second_dec) if use_raw_logits else model.project(second_dec)
                        first_flat = proj_output.reshape(-1, tokenizer_tgt.get_vocab_size())
                        second_flat = second_output.reshape(-1, tokenizer_tgt.get_vocab_size())
                        first_log_probs = F.log_softmax(first_flat, dim=-1) if use_raw_logits else first_flat
                        second_log_probs = F.log_softmax(second_flat, dim=-1) if use_raw_logits else second_flat
                        pad_id = tokenizer_tgt.token_to_id('<PAD>')
                        non_pad = (label.reshape(-1) != pad_id)
                        if non_pad.any():
                            kl1 = torch.nn.functional.kl_div(first_log_probs, second_log_probs.exp(), reduction='none', log_target=False).sum(dim=-1)
                            kl2 = torch.nn.functional.kl_div(second_log_probs, first_log_probs.exp(), reduction='none', log_target=False).sum(dim=-1)
                            kl = 0.5 * ((kl1 * non_pad).sum() + (kl2 * non_pad).sum()) / non_pad.sum().clamp(min=1)
                            loss = loss + config['rdrop_alpha'] * kl
                loss_for_backward = loss / config['gradient_accumulation_steps']

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.add_scalar('train lr', optimizer.param_groups[0]['lr'], global_step)
            tracker.log({'train/loss': loss.item(), 'train/lr': optimizer.param_groups[0]['lr']}, global_step)
            # Backpropagate the loss
            loss_for_backward.backward()

            if (batch_index + 1) % config['gradient_accumulation_steps'] == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_norm', 1.0))
                if torch.isfinite(gnorm):
                    optimizer.step()
                    scheduler.step()
                    if ema is not None:
                        ema.update()
                optimizer.zero_grad(set_to_none=True)

            if curriculum_sampler is not None:
                curriculum_sampler.set_step(global_step)
            global_step += 1

        if (batch_index + 1) % config['gradient_accumulation_steps'] != 0:
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_norm', 1.0))
            if torch.isfinite(gnorm):
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()
            optimizer.zero_grad(set_to_none=True)
        writer.flush()

        # Run validation at the end of every epoch
        if ema is not None:
            ema.apply_shadow()
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config['seq_len'],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
            loss_fn,
            use_amp,
            amp_dtype,
            use_grammar_mask=config.get('use_grammar_mask', True),
            use_chunked_ce=use_chunked_ce,
            ce_chunk_size=ce_chunk_size,
            pad_id=pad_id,
            label_smoothing=config.get('label_smoothing', 0.1),
        )
        if ema is not None:
            ema.restore()

        # Save the model at the end of every epoch with all metadata preserved
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        tokenizer_src_meta = getattr(tokenizer_src, "metadata", {})
        tokenizer_tgt_meta = getattr(tokenizer_tgt, "metadata", {})
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'global_step': global_step,
            'ema_state_dict': ema.state_dict() if ema is not None else None,
            'ema_num_updates': ema.num_updates if ema is not None else None,
            'config': dict(config),
            'tokenizer_metadata': {
                'lang_src': config.get('lang_src', 'en'),
                'lang_tgt': config.get('lang_tgt', 'hi'),
                'vocab_src_size': tokenizer_src.get_vocab_size(),
                'vocab_tgt_size': tokenizer_tgt.get_vocab_size(),
                'src_metadata': tokenizer_src_meta,
                'tgt_metadata': tokenizer_tgt_meta,
            },
            'data_metadata': {
                'datasource': config.get('datasource'),
                'dataset_config': config.get('dataset_config'),
                'seq_len': config.get('seq_len'),
                'train_samples': len(train_dataloader.dataset) if hasattr(train_dataloader, "dataset") else None,
            },
        }, model_filename)

    avg_last_k = int(config.get("average_last_k", 0) or 0)
    if avg_last_k > 1:
        from research.optimization import CheckpointAverager
        ckpt_dir = Path(f"{config['datasource']}_{config['model_folder']}")
        CheckpointAverager.average_last_k(
            ckpt_dir,
            k=avg_last_k,
            pattern=f"{config['model_basename']}*.pt",
        )
    tracker.finish()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    import sys
    try:
        from run_experiment import compose_config
        cli_args = [arg.lstrip("-") for arg in sys.argv[1:]]
        config = compose_config(cli_args if cli_args else ["modern"])
    except Exception:
        config = get_config()
    train_model(config)
