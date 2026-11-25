"""Unified experiment runner with configuration composition and CLI overrides (B0).

Supports training, profiling, quantization, and evaluation workflows with
automatic run manifest logging and optional Weights & Biases tracking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch

from config import get_config
from research.reproducibility import seed_everything, write_run_manifest, OptionalWandb


def load_yaml(path: Path) -> dict:
    """Load yaml file using PyYAML if available, else lightweight parser."""
    if not path.exists():
        return {}
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Simple fallback parser for flat/nested 1-level YAML
        data: dict = {}
        curr_key = None
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if not v:
                    curr_key = k
                    data[curr_key] = {}
                elif curr_key and line.startswith(" "):
                    data[curr_key][k] = parse_scalar(v)
                else:
                    curr_key = None
                    data[k] = parse_scalar(v)
        return data


def parse_scalar(val: str) -> Any:
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "none"):
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val.strip("'\"")


def compose_config(cli_args: list[str]) -> dict:
    """Compose default config from config.py and configs/*.yaml with CLI overrides."""
    cli_args = [arg.lstrip("-") for arg in cli_args]
    config = get_config()
    configs_dir = Path("configs")

    # Merge YAML configs: model.yaml (baseline), train.yaml, data.yaml
    model_cfg = load_yaml(configs_dir / "model.yaml")
    train_cfg = load_yaml(configs_dir / "train.yaml")
    data_cfg = load_yaml(configs_dir / "data.yaml")

    for section in (model_cfg, train_cfg, data_cfg):
        for k, v in section.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    # If modern ablation is selected via CLI args or explicit config, merge modern.yaml
    is_modern = any("modern_transformer" in arg or "ablation=modern" in arg or arg == "modern" or "model=modern" in arg for arg in cli_args) or config.get("architecture") == "modern_transformer"
    if is_modern:
        modern_cfg = load_yaml(configs_dir / "modern.yaml")
        for k, v in modern_cfg.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    # Apply CLI dot-separated key=value overrides
    for raw_arg in cli_args:
        arg = raw_arg.lstrip("-")
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        key = key.strip()
        if key == "rdrop":
            key = "rdrop_alpha"
        parsed_val = parse_scalar(val)

        if "." in key:
            parts = key.split(".")
            curr = config
            for p in parts[:-1]:
                if p not in curr or not isinstance(curr[p], dict):
                    curr[p] = {}
                curr = curr[p]
            curr[parts[-1]] = parsed_val
            # Also mirror leaf property to root for flat lookup
            config[parts[-1]] = parsed_val
        else:
            config[key] = parsed_val

    return config


def run_profiling(config: dict) -> None:
    from research.deployment import benchmark_inference, checkpoint_size_mb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = config.get("architecture", "vanilla")
    print(f"Profiling architecture ({arch}) on device: {device}...")

    if arch == "modern_transformer":
        from modern_model import build_modern_transformer
        model = build_modern_transformer(
            src_vocab_size=16000,
            tgt_vocab_size=40000,
            d_model=config["d_model"],
            encoder_layers=config.get("encoder_layers", 6),
            decoder_layers=config.get("decoder_layers", 4),
            heads=config.get("heads", 8),
            d_ff=config.get("d_ff", 2048),
        ).to(device)
    else:
        from model import build_transformer
        model = build_transformer(
            src_vocab_size=16000,
            tgt_vocab_size=40000,
            src_seq_len=config["seq_len"],
            tgt_seq_len=config["seq_len"],
            d_model=config["d_model"],
            encoder_layers=config.get("encoder_layers", 6),
            decoder_layers=config.get("decoder_layers", 4),
        ).to(device)

    dummy_input = torch.randint(4, 16000, (1, 30), device=device)
    dummy_mask = torch.ones((1, 1, 1, 30), dtype=torch.bool, device=device)

    metrics = benchmark_inference(model, dummy_input, dummy_mask, max_decode_len=20, repeats=15, device=device)
    print("\n--- Latency and Throughput Profile ---")
    print(f"Architecture:   {arch}")
    print(f"Latency:        {metrics['ms_per_sentence']:.2f} ms/sentence")
    print(f"Throughput:     {metrics['tokens_per_second']:.2f} tokens/sec")
    print(f"Peak VRAM:      {metrics['peak_vram_mb']:.1f} MB")
    print("--------------------------------------\n")


def main() -> None:
    cli_args = [arg.lstrip("-") for arg in sys.argv[1:]]
    config = compose_config(cli_args)
    seed = config.get("seed", 13)
    seed_everything(seed)

    mode = config.get("mode", "train")
    exp_name = config.get("experiment_name", "runs/experiment")
    write_run_manifest(exp_name, mode, config, seed)

    print(f"Experiment initialized: mode='{mode}', seed={seed}")
    print(f"Config: encoder_layers={config.get('encoder_layers')}, decoder_layers={config.get('decoder_layers')}, rdrop={config.get('rdrop_alpha')}")

    if mode == "profile":
        run_profiling(config)
    elif mode == "train":
        from train import train_model
        train_model(config)
    else:
        print(f"Unknown mode: {mode}. Supported modes: train, profile.")


if __name__ == "__main__":
    main()
