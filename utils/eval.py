from __future__ import annotations
from typing import Optional
import re
import math

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig


class QuizEvaluator:
    """
    Lightweight generative evaluator to score a student answer against a gold solution.
    Returns a float score in [0, 1].

    Supported engines (suggested):
      - google/mt5-small (multilingual, CPU-friendly)
      - Shahm/t5-small-german (German T5)
    """

    def __init__(self, model_name: str = "google/mt5-small"):
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        self.is_enc_dec = bool(getattr(self.config, "is_encoder_decoder", False))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else None
        if self.is_enc_dec:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_prompt(self, student: str, gold: str) -> str:
        # Strict instruction: content-aligned scoring only, output only integer 0..100.
        return (
            "Bewerte, wie gut die Schüler-Antwort inhaltlich mit der Musterlösung übereinstimmt. "
            "Gib NUR eine ganze Zahl zwischen 0 und 100 aus (Prozent), ohne Text.\n\n"
            f"Musterlösung:\n{gold}\n\nSchüler-Antwort:\n{student}\n\nBewertung (0-100):"
        )

    @torch.no_grad()
    def score(self, student: str, gold: str) -> float:
        student = (student or "").strip()
        gold = (gold or "").strip()
        if not student or not gold:
            return 0.0

        prompt = self._build_prompt(student, gold)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=768)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        pad_id = self.tok.pad_token_id
        if pad_id is None and self.tok.eos_token_id is not None:
            pad_id = self.tok.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=8,
            pad_token_id=pad_id,
            no_repeat_ngram_size=3,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
            length_penalty=1.0,
            eos_token_id=self.tok.eos_token_id if self.tok.eos_token_id is not None else None,
        )
        if self.is_enc_dec and getattr(self.model.config, "decoder_start_token_id", None) is None:
            gen_kwargs["decoder_start_token_id"] = self.tok.pad_token_id

        # Some decoder-only models do not accept token_type_ids
        if not self.is_enc_dec and "token_type_ids" in enc:
            enc.pop("token_type_ids")

        out = self.model.generate(**enc, **gen_kwargs)
        if self.is_enc_dec:
            gen_ids = out[0]
        else:
            in_len = enc["input_ids"].shape[-1]
            gen_ids = out[0][in_len:]
        txt = self.tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Parse first integer 0..100
        m = re.search(r"\b(100|\d{1,2})\b", txt)
        if m:
            val = int(m.group(1))
            val = max(0, min(100, val))
            return val / 100.0

        # Fallback: lexical overlap as rough estimate
        def toks(s: str):
            return set(re.findall(r"\w+", s.lower()))

        a = toks(student); b = toks(gold)
        if not a or not b:
            return 0.0
        jac = len(a & b) / len(a | b)
        return float(jac)
