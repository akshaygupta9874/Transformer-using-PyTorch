from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path, get_tokenizer_path

import math
import torch
import torch.nn as nn
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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, loss_fn, use_amp, amp_dtype, num_examples=2):
    model.eval()
    print_msg('Starting validation...')

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
                proj_output = model.project(decoder_output)
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

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

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

def get_ds(config):
    # It only has the train split, so we divide it overselves
    dataset_kwargs = {"split": "train"}
    if config.get("dataset_config"):
        dataset_kwargs["name"] = config["dataset_config"]
    ds_raw = load_dataset(config["datasource"], **dataset_kwargs)

    if config.get("max_train_samples"):
        max_samples = min(int(config["max_train_samples"]), len(ds_raw))
        print(f"Limiting dataset to {max_samples:,} samples (config max_train_samples)...")
        ds_raw = ds_raw.select(range(max_samples))

    # Split: keep up to val_size_limit (default 2000) for validation, rest for training
    val_size_limit = config.get("val_size_limit", 2000)
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
            use_gradient_checkpointing=config.get('use_gradient_checkpointing', False),
        )
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'],
        d_model=config['d_model'], h=config.get('heads', 8), dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 2048), encoder_layers=config.get('encoder_layers', 6),
        decoder_layers=config.get('decoder_layers', 4),
    )
    return model

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

    optimizer_kwargs = {"lr": config['lr'], "eps": 1e-9, "weight_decay": config.get('weight_decay', 0.01)}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    warmup_steps = max(1, config.get('warmup_steps', 4000))
    accum_steps = max(1, config.get('gradient_accumulation_steps', 1))
    total_steps = config['num_epochs'] * math.ceil(max(1, len(train_dataloader)) / accum_steps)
    if config.get('lr_schedule') == 'cosine':
        from research.optimization import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = LambdaLR(optimizer, lambda step: min((step + 1) ** -0.5, (step + 1) * warmup_steps ** -1.5))

    ema = None
    if config.get('use_ema', False):
        from research.optimization import ModelEMA
        ema = ModelEMA(model, decay=config.get('ema_decay', 0.9999))

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        if 'scheduler_state_dict' in state and state['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(state['scheduler_state_dict'])
        if 'ema_state_dict' in state and state['ema_state_dict'] is not None and ema is not None:
            ema.load_state_dict(state['ema_state_dict'])
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

    for epoch in range(initial_epoch, config['num_epochs']):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch_index, batch in enumerate(batch_iterator):

            encoder_input = batch['encoder_input'].to(device,non_blocking=True) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device,non_blocking=True) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device,non_blocking=True) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device,non_blocking=True) # (B, 1, seq_len, seq_len)

            label = batch['label'].to(device) # (B, seq_len)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                loss = loss_fn(
                    proj_output.reshape(-1, tokenizer_tgt.get_vocab_size()),
                    label.reshape(-1),
                )
                if config.get('rdrop_alpha', 0.0) > 0:
                    # A second dropout realization should predict the same
                    # distribution under the same teacher-forced inputs.
                    second_output = model.project(model.decode(model.encode(encoder_input, encoder_mask), encoder_mask, decoder_input, decoder_mask))
                    first_log_probs = proj_output.reshape(-1, tokenizer_tgt.get_vocab_size())
                    second_log_probs = second_output.reshape(-1, tokenizer_tgt.get_vocab_size())
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_norm', 1.0))
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update()
                optimizer.zero_grad(set_to_none=True)

            if curriculum_sampler is not None:
                curriculum_sampler.set_step(global_step)
            global_step += 1

        if (batch_index + 1) % config['gradient_accumulation_steps'] != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_norm', 1.0))
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
    tracker.finish()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
