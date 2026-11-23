import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(
        self,
        ds,
        tokenizer_src,
        tokenizer_tgt,
        src_lang,
        tgt_lang,
        seq_len,
        use_placeholders=False,
        pad_to_max=True,
    ) -> None:

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        self.use_placeholders = use_placeholders
        self.pad_to_max = pad_to_max

        # Source special tokens
        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id('<BOS>')],
            dtype=torch.int64
        )

        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id('<EOS>')],
            dtype=torch.int64
        )

        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id('<PAD>')],
            dtype=torch.int64
        )

        # Target special tokens
        self.sos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id('<BOS>')],
            dtype=torch.int64
        )

        self.eos_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id('<EOS>')],
            dtype=torch.int64
        )

        self.pad_token_tgt = torch.tensor(
            [tokenizer_tgt.token_to_id('<PAD>')],
            dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):

        src_target_pair = self.ds[index]

        if 'translation' in src_target_pair:
            src_text = src_target_pair['translation'][self.src_lang]
            tgt_text = src_target_pair['translation'][self.tgt_lang]
        else:
            src_text = src_target_pair['src']
            tgt_text = src_target_pair['tgt']
        if self.use_placeholders:
            from research.placeholders import substitute_parallel
            src_text, tgt_text, _ = substitute_parallel(src_text, tgt_text)

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        if not self.pad_to_max:
            # Dynamic collation path: truncate to maximum budget if necessary
            max_enc = self.seq_len - 2
            max_dec = self.seq_len - 1
            if len(enc_input_tokens) > max_enc:
                enc_input_tokens = enc_input_tokens[:max_enc]
            if len(dec_input_tokens) > max_dec:
                dec_input_tokens = dec_input_tokens[:max_dec]

            encoder_input = torch.cat([
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
            ])
            decoder_input = torch.cat([
                self.sos_token_tgt,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
            ])
            label = torch.cat([
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token_tgt,
            ])
            return {
                "encoder_input": encoder_input,
                "decoder_input": decoder_input,
                "label": label,
                "src_text": src_text,
                "tgt_text": tgt_text,
            }

        # Fixed padding path (seq_len)
        enc_num_padding_tokens = (
            self.seq_len - len(enc_input_tokens) - 2
        )
        dec_num_padding_tokens = (
            self.seq_len - len(dec_input_tokens) - 1
        )

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.full(
                (enc_num_padding_tokens,),
                self.pad_token.item(),
                dtype=torch.int64
            )
        ])

        decoder_input = torch.cat([
            self.sos_token_tgt,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.full(
                (dec_num_padding_tokens,),
                self.pad_token_tgt.item(),
                dtype=torch.int64
            )
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token_tgt,
            torch.full(
                (dec_num_padding_tokens,),
                self.pad_token_tgt.item(),
                dtype=torch.int64
            )
        ])

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask":
                (encoder_input != self.pad_token)
                .unsqueeze(0)
                .unsqueeze(0)
                .int(),
            "decoder_mask":
                (decoder_input != self.pad_token_tgt)
                .unsqueeze(0)
                .unsqueeze(0)
                .int()
                & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }


class PadCollate:
    """Collate function that dynamically pads each batch to its longest item."""

    def __init__(self, pad_token_src: int = 0, pad_token_tgt: int = 0):
        self.pad_token_src = pad_token_src
        self.pad_token_tgt = pad_token_tgt

    def __call__(self, batch: list[dict]) -> dict:
        max_enc = max(item["encoder_input"].size(0) for item in batch)
        max_dec = max(item["decoder_input"].size(0) for item in batch)

        b_enc_input = []
        b_dec_input = []
        b_label = []
        b_src_text = []
        b_tgt_text = []

        for item in batch:
            enc = item["encoder_input"]
            dec = item["decoder_input"]
            lbl = item["label"]

            pad_enc = max_enc - enc.size(0)
            pad_dec = max_dec - dec.size(0)

            if pad_enc > 0:
                enc = torch.cat([enc, torch.full((pad_enc,), self.pad_token_src, dtype=torch.int64)])
            if pad_dec > 0:
                dec = torch.cat([dec, torch.full((pad_dec,), self.pad_token_tgt, dtype=torch.int64)])
                lbl = torch.cat([lbl, torch.full((pad_dec,), self.pad_token_tgt, dtype=torch.int64)])

            b_enc_input.append(enc)
            b_dec_input.append(dec)
            b_label.append(lbl)
            b_src_text.append(item["src_text"])
            b_tgt_text.append(item["tgt_text"])

        encoder_input = torch.stack(b_enc_input, dim=0)
        decoder_input = torch.stack(b_dec_input, dim=0)
        label = torch.stack(b_label, dim=0)

        encoder_mask = (encoder_input != self.pad_token_src).unsqueeze(1).unsqueeze(1).int()
        decoder_mask = (decoder_input != self.pad_token_tgt).unsqueeze(1).unsqueeze(1).int() & causal_mask(decoder_input.size(1))

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": b_src_text,
            "tgt_text": b_tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(
        torch.ones(1, size, size),
        diagonal=1
    ).type(torch.int)

    return mask == 0
