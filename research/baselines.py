"""External reference points: IndicTrans2, NLLB-200, and PEFT/LoRA adaptation (B12).

Provides programmatic evaluation runners for pretrained benchmarks and a lightweight
LoRA fine-tuning setup for comparing from-scratch NMT against adapted multilingual baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict


class IndicTrans2Baseline:
    """Wrapper for AI4Bharat IndicTrans2 models."""

    def __init__(self, model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None

    def load(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as err:
            raise RuntimeError("Install transformers to run IndicTrans2 baseline.") from err

        print(f"Loading IndicTrans2 checkpoint: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self._model.eval()

    def translate_batch(self, sentences: list[str], max_length: int = 256) -> list[str]:
        if self._model is None:
            self.load()
        import torch

        # Format input with language token if required
        inputs = self._tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_length=max_length)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


class NLLB200Baseline:
    """Wrapper for Meta NLLB-200 models."""

    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._tokenizer = None
        self._model = None

    def load(self):
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as err:
            raise RuntimeError("Install transformers to run NLLB-200 baseline.") from err

        print(f"Loading NLLB-200 checkpoint: {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, src_lang="eng_Latn")
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def translate_batch(self, sentences: list[str], max_length: int = 256) -> list[str]:
        if self._model is None:
            self.load()
        import torch

        tgt_lang_id = self._tokenizer.convert_tokens_to_ids("hin_Deva")
        inputs = self._tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, forced_bos_token_id=tgt_lang_id, max_length=max_length)
        return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


def setup_lora_fine_tuning(base_model, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
    """Configure parameter-efficient LoRA adapters for comparative adaptation arm."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as err:
        raise RuntimeError("Install peft (pip install peft) for LoRA fine-tuning.") from err

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],
    )
    return get_peft_model(base_model, peft_config)
