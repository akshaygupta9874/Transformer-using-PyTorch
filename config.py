from pathlib import Path

def get_config():
    return {
        "architecture": "vanilla",
        "norm_type": "layernorm",
        "pos_embedding": "sinusoidal",
        "ffn_type": "relu",
        "max_seq_len": 2048,
        "batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_epochs": 20,
        "lr": 10**-4,
        "lr_schedule": "cosine",
        "warmup_steps": 4000,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "use_ema": True,
        "ema_decay": 0.9999,
        "use_amp": True,
        "amp_dtype": "bfloat16",
        "dynamic_padding": True,
        "compile_model": False,
        "max_train_samples": None,
        "val_size_limit": 2000,
        "leakage_check_samples": 5000,
        "seq_len": 350,
        "d_model": 512,
        "encoder_layers": 6,
        "decoder_layers": 4,
        "heads": 8,
        "dropout": 0.1,
        "d_ff": 2048,
        "num_workers": 2,
        "prefetch_factor": 2,
        "datasource": "ai4bharat/samanantar",
        "dataset_config": "hi",
        "lang_src": "en",
        "lang_tgt": "hi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "data/tokenizers/tokenizer_{}.akshara.v2.json",
        "tokenizer_vocab_sizes": {"en": 16000, "hi": 40000},
        "shared_vocabulary": False,
        "seed": 13,
        "rdrop_alpha": 0.0,
        "wandb_enabled": False,
        "wandb_project": "en-hi-nmt",
        "use_placeholders": True,
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])


def get_tokenizer_path(config: dict, lang: str) -> Path:
    """Find tokenizer path with smart resolution across data/tokenizers and project root."""
    target = Path(config.get("tokenizer_file", "data/tokenizers/tokenizer_{}.akshara.v2.json").format(lang))
    if target.exists():
        return target
    # Fallback to data/tokenizers/
    candidate_dt = Path("data/tokenizers") / f"tokenizer_{lang}.akshara.v2.json"
    if candidate_dt.exists():
        return candidate_dt
    # Fallback to project root
    candidate_root = Path(f"tokenizer_{lang}.akshara.v2.json")
    if candidate_root.exists():
        return candidate_root
    return target

