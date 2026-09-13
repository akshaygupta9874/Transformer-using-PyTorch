from pathlib import Path
import argparse

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

from config import get_config, latest_weights_file_path
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def load_model(config, tokenizer_src, tokenizer_tgt, device):
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"],
    ).to(device)

    model_filename = latest_weights_file_path(config)
    if model_filename is None:
        raise FileNotFoundError(
            "No checkpoint found. Train the model before running translation."
        )

    state = torch.load(model_filename, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {model_filename}")
    return model


def get_sentence_from_dataset(config, tokenizer_src, tokenizer_tgt, index):
    dataset = load_dataset(
        config["datasource"],
        name=f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
        trust_remote_code=True,
    )
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


def translate(sentence, config=None):
    config = config or get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer_src = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_src"]))
    tokenizer_tgt = Tokenizer.from_file(config["tokenizer_file"].format(config["lang_tgt"]))
    model = load_model(config, tokenizer_src, tokenizer_tgt, device)

    expected_translation = None
    if isinstance(sentence, int) or str(sentence).isdigit():
        sentence, expected_translation = get_sentence_from_dataset(
            config,
            tokenizer_src,
            tokenizer_tgt,
            int(sentence),
        )

    source_ids = tokenizer_src.encode(sentence).ids
    max_source_tokens = config["seq_len"] - 2
    if len(source_ids) > max_source_tokens:
        raise ValueError(
            f"Sentence has {len(source_ids)} source tokens, but the model supports "
            f"at most {max_source_tokens}."
        )

    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    pad_id = tokenizer_src.token_to_id("[PAD]")
    source = torch.tensor(
        [sos_id, *source_ids, eos_id] + [pad_id] * (config["seq_len"] - len(source_ids) - 2),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    source_mask = (source != pad_id).unsqueeze(1).unsqueeze(1)

    target_sos_id = tokenizer_tgt.token_to_id("[SOS]")
    target_eos_id = tokenizer_tgt.token_to_id("[EOS]")
    decoder_input = torch.tensor([[target_sos_id]], dtype=torch.long, device=device)
    use_amp = device.type == "cuda" and config.get("use_amp", False)
    amp_dtype = torch.bfloat16 if config.get("amp_dtype") == "bfloat16" else torch.float16

    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        encoder_output = model.encode(source, source_mask)
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
    print(f"SOURCE:    {sentence}")
    if expected_translation is not None:
        print(f"TARGET:    {expected_translation}")
    print(f"PREDICTED: {predicted_translation}")
    return predicted_translation


def main():
    parser = argparse.ArgumentParser(description="Translate English text to Italian.")
    parser.add_argument(
        "sentence",
        nargs="?",
        default="I am not a very good student.",
        help="Text to translate, or a training-dataset index.",
    )
    args = parser.parse_args()
    translate(args.sentence)


if __name__ == "__main__":
    main()
