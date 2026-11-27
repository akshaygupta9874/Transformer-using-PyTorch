from pathlib import Path
import argparse

import torch
from datasets import load_dataset
from tokenizer import Tokenizer

from config import get_config, latest_weights_file_path, get_tokenizer_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from research.placeholders import substitute
from research.decoding import beam_search_decode, mbr_decode
from research.knn_mt import KNNMTDatastore, KNNMTDecoder


def load_model(config, tokenizer_src, tokenizer_tgt, device, checkpoint_path=None):
    model_filename = checkpoint_path or latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(
            "No checkpoint found. Train the model before running translation."
        )

    state = torch.load(model_filename, map_location=device, weights_only=False)
    ckpt_cfg = state.get("config") or {}
    model_sd = state.get("model_state_dict") or {}
    ckpt_arch = ckpt_cfg.get("architecture")
    if ckpt_arch is None and model_sd:
        ckpt_arch = "modern_transformer" if any(k.startswith("rope.") or k.startswith("encoder_blocks.") for k in model_sd) else "vanilla"

    arch = ckpt_arch or config.get("architecture", "vanilla")
    effective_cfg = {**ckpt_cfg, **config, "architecture": arch}

    if arch == "modern_transformer":
        from modern_model import build_modern_transformer
        use_qk_norm = ckpt_cfg.get("use_qk_norm", any(".q_norm." in k for k in model_sd))
        tie_embeddings = ckpt_cfg.get("tie_embeddings", effective_cfg.get("tie_embeddings", False))
        kv_heads = ckpt_cfg.get("kv_heads", effective_cfg.get("kv_heads"))
        encoder_d_ff = ckpt_cfg.get("encoder_d_ff", effective_cfg.get("encoder_d_ff"))
        model = build_modern_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            d_model=effective_cfg["d_model"],
            encoder_layers=effective_cfg.get("encoder_layers", 6),
            decoder_layers=effective_cfg.get("decoder_layers", 4),
            heads=effective_cfg.get("heads", 8),
            d_ff=effective_cfg.get("d_ff", 2048),
            dropout=effective_cfg.get("dropout", 0.1),
            max_seq_len=effective_cfg.get("max_seq_len", 2048),
            use_qk_norm=use_qk_norm,
            tie_embeddings=tie_embeddings,
            kv_heads=kv_heads,
            encoder_d_ff=encoder_d_ff,
        ).to(device)
    else:
        model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            effective_cfg["seq_len"],
            effective_cfg["seq_len"],
            d_model=effective_cfg["d_model"],
            h=effective_cfg.get("heads", 8),
            dropout=effective_cfg.get("dropout", 0.1),
            d_ff=effective_cfg.get("d_ff", 2048),
            encoder_layers=effective_cfg.get("encoder_layers"),
            decoder_layers=effective_cfg.get("decoder_layers"),
        ).to(device)

    weights = state["model_state_dict"]
    ema_weights = state.get("ema_state_dict")
    if effective_cfg.get("use_ema") and ema_weights:
        weights = {**weights, **ema_weights}
    model.load_state_dict(weights)
    model.eval()
    print(f"Loaded checkpoint: {model_filename} (architecture={arch})")
    return model


def get_sentence_from_dataset(config, tokenizer_src, tokenizer_tgt, index):
    dataset_kwargs = {"split": "train"}
    if config.get("dataset_config"):
        dataset_kwargs["name"] = config["dataset_config"]
    dataset = load_dataset(config["datasource"], **dataset_kwargs)
    bilingual_dataset = BilingualDataset(
        dataset,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    if index < 0 or index >= len(bilingual_dataset):
        raise IndexError(f"Dataset index must be between 0 and {len(bilingual_dataset) - 1}.")
    item = bilingual_dataset[index]
    return item["src_text"], item["tgt_text"]


def translate(
    sentence,
    config=None,
    strategy=None,
    beam_size=None,
    mbr_samples=16,
    knn_datastore_path=None,
    knn_lambda=0.2,
    knn_k=8,
    checkpoint_path=None,
):
    config = config or get_config()
    if strategy is None:
        strategy = config.get("default_decoding_strategy", "greedy")
    if beam_size is None:
        beam_size = config.get("beam_size", 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Strategy: {strategy}")

    tokenizer_src = Tokenizer.load(get_tokenizer_path(config, config["lang_src"]))
    tokenizer_tgt = Tokenizer.load(get_tokenizer_path(config, config["lang_tgt"]))
    model = load_model(config, tokenizer_src, tokenizer_tgt, device, checkpoint_path=checkpoint_path)

    grammar_mask = None
    if config.get("use_grammar_mask", True) and hasattr(tokenizer_tgt, "tokens"):
        from tokenizer.grammar_mask import DevanagariGrammarMask
        grammar_mask = DevanagariGrammarMask(tokenizer_tgt)

    expected_translation = None
    if isinstance(sentence, int) or str(sentence).isdigit():
        sentence, expected_translation = get_sentence_from_dataset(
            config,
            tokenizer_src,
            tokenizer_tgt,
            int(sentence),
        )

    tokenization_text, placeholder_map = substitute(sentence) if config.get('use_placeholders', False) else (sentence, None)
    source_ids = tokenizer_src.encode(tokenization_text).ids
    max_source_tokens = config["seq_len"] - 2
    if len(source_ids) > max_source_tokens:
        raise ValueError(
            f"Sentence has {len(source_ids)} source tokens, but the model supports "
            f"at most {max_source_tokens}."
        )

    sos_id = tokenizer_src.token_to_id("<BOS>")
    eos_id = tokenizer_src.token_to_id("<EOS>")
    pad_id = tokenizer_src.token_to_id("<PAD>")
    source = torch.tensor(
        [sos_id, *source_ids, eos_id] + [pad_id] * (config["seq_len"] - len(source_ids) - 2),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    source_mask = (source != pad_id).unsqueeze(1).unsqueeze(1)

    target_sos_id = tokenizer_tgt.token_to_id("<BOS>")
    target_eos_id = tokenizer_tgt.token_to_id("<EOS>")
    use_amp = device.type == "cuda" and config.get("use_amp", False)
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        if strategy == "beam":
            output_tokens = beam_search_decode(
                model,
                source,
                source_mask,
                tokenizer_tgt,
                beam_size=beam_size,
                max_len=config["seq_len"],
                device=device,
            )
            predicted_translation = tokenizer_tgt.decode(output_tokens)

        elif strategy == "mbr":
            if hasattr(model, "rope"):
                from research.fast_decoding import fast_mbr_decode
                predicted_translation = fast_mbr_decode(
                    model,
                    source,
                    source_mask,
                    tokenizer_tgt,
                    num_samples=mbr_samples,
                    max_len=config["seq_len"],
                    grammar_mask=grammar_mask,
                )
            else:
                predicted_translation = mbr_decode(
                    model,
                    source,
                    source_mask,
                    tokenizer_tgt,
                    num_samples=mbr_samples,
                    max_len=config["seq_len"],
                )

        elif strategy == "knn":
            if not knn_datastore_path or not Path(knn_datastore_path).exists():
                raise FileNotFoundError(f"kNN datastore not found at {knn_datastore_path}")
            datastore = KNNMTDatastore.load(knn_datastore_path, device=device)
            knn_decoder = KNNMTDecoder(datastore, k=knn_k, lambda_weight=knn_lambda)
            
            encoder_output = model.encode(source, source_mask)
            decoder_input = torch.tensor([[target_sos_id]], dtype=torch.long, device=device)
            vocab_size = tokenizer_tgt.get_vocab_size()

            while decoder_input.size(1) < config["seq_len"]:
                decoder_mask = causal_mask(decoder_input.size(1)).to(device)
                dec_out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
                model_log_probs = model.project(dec_out[:, -1])  # (1, vocab_size)
                
                # Query kNN with the last decoder hidden state
                query_hidden = dec_out[:, -1]
                knn_probs = knn_decoder.compute_knn_distribution(query_hidden, vocab_size)
                interp_log_probs = knn_decoder.interpolate(model_log_probs, knn_probs)
                
                next_token = interp_log_probs.argmax(dim=-1).item()
                decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == target_eos_id:
                    break
            predicted_translation = tokenizer_tgt.decode(decoder_input[0].tolist())

        elif strategy == "fast_greedy":
            from research.fast_decoding import fast_greedy_decode
            output_tokens = fast_greedy_decode(
                model,
                source,
                source_mask,
                sos_id=target_sos_id,
                eos_id=target_eos_id,
                max_len=config["seq_len"],
                device=device,
                grammar_mask=grammar_mask,
            )
            predicted_translation = tokenizer_tgt.decode(output_tokens)

        elif strategy == "fast_beam":
            from research.fast_decoding import fast_beam_search
            output_tokens = fast_beam_search(
                model,
                source,
                source_mask,
                sos_id=target_sos_id,
                eos_id=target_eos_id,
                beam_size=beam_size,
                max_len=config["seq_len"],
                alpha=config.get("length_penalty_alpha", 0.6),
                coverage_beta=config.get("coverage_penalty_beta", 0.2),
                device=device,
                grammar_mask=grammar_mask,
            )
            predicted_translation = tokenizer_tgt.decode(output_tokens)

        else:  # default: greedy
            encoder_output = model.encode(source, source_mask)
            decoder_input = torch.tensor([[target_sos_id]], dtype=torch.long, device=device)
            while decoder_input.size(1) < config["seq_len"]:
                decoder_mask = causal_mask(decoder_input.size(1)).to(device)
                output = model.decode(
                    encoder_output,
                    source_mask,
                    decoder_input,
                    decoder_mask,
                )
                next_token = model.project(output[:, -1]).argmax(dim=1).item()
                decoder_input = torch.cat(
                    [decoder_input, torch.tensor([[next_token]], device=device)],
                    dim=1,
                )
                if next_token == target_eos_id:
                    break
            predicted_translation = tokenizer_tgt.decode(decoder_input[0].tolist())

    if placeholder_map is not None:
        predicted_translation = placeholder_map.restore(predicted_translation)

    print(f"SOURCE:    {sentence}")
    if expected_translation is not None:
        print(f"TARGET:    {expected_translation}")
    print(f"PREDICTED: {predicted_translation}")
    return predicted_translation


def main():
    parser = argparse.ArgumentParser(description="Translate English text to Hindi.")
    parser.add_argument(
        "sentence",
        nargs="?",
        default="I am not a very good student.",
        help="Text to translate, or a training-dataset index.",
    )
    parser.add_argument("--modern", action="store_true", help="Load modern transformer config (configs/modern.yaml)")
    parser.add_argument("--strategy", choices=["greedy", "beam", "mbr", "knn", "fast_greedy", "fast_beam"], default=None)
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--mbr-samples", type=int, default=16)
    parser.add_argument("--knn-datastore", type=Path, default=None)
    parser.add_argument("--knn-lambda", type=float, default=0.2)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = None
    if args.modern:
        from run_experiment import compose_config
        cfg = compose_config(["architecture=modern_transformer"])

    translate(
        args.sentence,
        config=cfg,
        strategy=args.strategy,
        beam_size=args.beam_size,
        mbr_samples=args.mbr_samples,
        knn_datastore_path=args.knn_datastore,
        knn_lambda=args.knn_lambda,
        knn_k=args.knn_k,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
