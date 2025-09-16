from typing import Optional, List
import math
import re
import difflib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig


class PostEditor:
    """
    Light-weight grammar/sentence-structure post editor for QA answers.
    Goal: polish German output without adding or removing facts.
    Supported:
    - Seq2Seq models (e.g., T5 family like Shahm/t5-small-german)
    - Decoder-only models (e.g., Qwen, Gemma, Phi, TinyLlama)
    """

    def __init__(self, model_name: str = "Shahm/t5-small-german"):
        self.model_name = model_name
        lower = model_name.lower()
        # Some models (e.g., sentence-transformers/...) are encoders only and cannot generate.
        # In that case, transparently proxy to a small seq2seq paraphraser (mt5-small) as engine.
        self._proxied = False
        engine_name = model_name
        if lower.startswith("sentence-transformers/") or "paraphrase-multilingual-minilm-l12-v2" in lower:
            engine_name = "google/mt5-small"
            self._proxied = True

        self.tok = AutoTokenizer.from_pretrained(engine_name)
        config = AutoConfig.from_pretrained(engine_name)
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        self.uses_grammar_prefix = ("aiassociates/t5-small-grammar-correction-german" in lower) or ("grammar-correction-german" in lower)
        # Mode selection: paraphrase vs grammar polishing
        self.mode = "paraphrase" if ("mt5" in engine_name.lower() or "flan-t5" in engine_name.lower()) else "grammar"
        dtype = torch.float16 if torch.cuda.is_available() else None
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(engine_name, torch_dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(engine_name, torch_dtype=dtype)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_prompt(self, text: str) -> str:
        if self.uses_grammar_prefix:
            # Model expects a simple prefix prompt
            return f"grammar: {text}"
        if self.mode == "paraphrase":
            keep_terms = self._extract_keep_terms(text)
            keep_hint = ", ".join(keep_terms) if keep_terms else "—"
            instr = (
                "Paraphrasiere den folgenden Text deutlich um, in einfaches, gut lesbares Deutsch. "
                "Nutze andere Satzstrukturen und Synonyme, kürzere klare Sätze. "
                "Ändere KEINE Inhalte oder Fakten. Behalte wichtige Fachwörter exakt bei (wenn vorhanden): "
                f"{keep_hint}.\n\n"
            )
            return f"{instr}Text:\n{text}\n\nParaphrase:"
        # Generic strict instruction for grammar polishing
        instr = (
            "Korrigiere Grammatik und Satzbau des folgenden deutschen Textes. "
            "Ändere keine Fakten, füge nichts hinzu und entferne keine relevanten Inhalte. "
            "Antworte NUR mit dem korrigierten Text, ohne Erklärungen.\n\n"
        )
        return f"{instr}Text: {text}\n\nKorrigierte Fassung:"

    def _extract_keep_terms(self, text: str, max_terms: int = 10) -> List[str]:
        # very light heuristic: keep numbers and capitalized tokens (likely proper nouns)
        terms = []
        seen = set()
        for m in re.finditer(r"\b([A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]{2,}|\d[\d\.,]*)\b", text):
            t = m.group(0)
            if t not in seen:
                terms.append(t)
                seen.add(t)
            if len(terms) >= max_terms:
                break
        return terms

    @torch.no_grad()
    def edit(self, text: str, target_chars: int = 800, context: Optional[str] = None) -> str:
        if not text or not text.strip():
            return text

        prompt = self._build_prompt(text.strip())
        # Tokenization/truncation limits
        if self.is_encoder_decoder:
            max_len = 512
        else:
            max_len = 1024
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        # Generation settings
        max_new_tokens = max(64, min(512, math.ceil(target_chars / 4)))
        pad_id = self.tok.pad_token_id
        if pad_id is None and self.tok.eos_token_id is not None:
            pad_id = self.tok.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            no_repeat_ngram_size=3,
            eos_token_id=self.tok.eos_token_id if self.tok.eos_token_id is not None else None,
        )

        if self.is_encoder_decoder:
            # Deterministic settings
            if self.uses_grammar_prefix:
                # Follow model's recommended settings (similar to HappyTransformer snippet)
                gen_kwargs.update(dict(
                    do_sample=False,
                    num_beams=5,
                    early_stopping=True,
                    length_penalty=1.0,
                    min_new_tokens=1,
                ))
            elif self.mode == "paraphrase":
                # Für mT5/T5: Sampling aktivieren, um tatsächliche Umformulierungen zu erzwingen
                gen_kwargs.update(dict(
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    num_beams=1,
                    early_stopping=True,
                    length_penalty=1.0,
                    min_new_tokens=16,
                    repetition_penalty=1.05,
                    no_repeat_ngram_size=4,
                    num_return_sequences=3,
                ))
            else:
                gen_kwargs.update(dict(
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.05,
                    min_new_tokens=16,
                ))
            if getattr(self.model.config, "decoder_start_token_id", None) is None:
                gen_kwargs["decoder_start_token_id"] = self.tok.pad_token_id
        else:
            # Low-variance sampling
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=0.15,
                top_p=0.9,
                repetition_penalty=1.15,
            ))

        # Some decoder-only models do not accept token_type_ids
        if not self.is_encoder_decoder and "token_type_ids" in enc:
            enc.pop("token_type_ids")

        out = self.model.generate(**enc, **gen_kwargs)
        # Decode helper
        def _decode_seq(ids):
            return self.tok.decode(ids, skip_special_tokens=True).strip()

        if self.is_encoder_decoder and self.mode == "paraphrase" and gen_kwargs.get("num_return_sequences", 1) > 1:
            # multiple candidates -> choose least similar to original
            cands = []
            for i in range(out.shape[0]):
                cands.append(_decode_seq(out[i]))
            base = re.sub(r"\s+", " ", text.strip().lower())
            def sim(a, b):
                return difflib.SequenceMatcher(None, a, b).ratio()
            picked = None
            best_score = 1.0
            for c in cands:
                if not c:
                    continue
                s = sim(base, re.sub(r"\s+", " ", c.strip().lower()))
                if s < best_score:
                    best_score = s
                    picked = c
            edited = (picked or cands[0] or "").strip()
        else:
            if self.is_encoder_decoder:
                gen_ids = out[0]
                edited = _decode_seq(gen_ids)
            else:
                in_len = enc["input_ids"].shape[-1]
                gen_ids = out[0][in_len:]
                edited = _decode_seq(gen_ids)

        # Falls das Modell nahezu identischen Text zurückgibt, stärke die Umformulierung und versuche es einmal erneut
        def _normalize(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())
        if _normalize(edited) == _normalize(text):
            # zweiter Versuch mit höherer Diversität
            if self.is_encoder_decoder and self.mode == "paraphrase":
                retry_kwargs = dict(
                    do_sample=True,
                    temperature=0.85,
                    top_p=0.95,
                    num_beams=1,
                    min_new_tokens=24,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=5,
                    pad_token_id=pad_id,
                    eos_token_id=self.tok.eos_token_id if self.tok.eos_token_id is not None else None,
                )
                if getattr(self.model.config, "decoder_start_token_id", None) is None:
                    retry_kwargs["decoder_start_token_id"] = self.tok.pad_token_id
                out2 = self.model.generate(**enc, **retry_kwargs)
                gen_ids2 = out2[0]
                edited2 = self.tok.decode(gen_ids2, skip_special_tokens=True).strip()
                if edited2 and _normalize(edited2) != _normalize(text):
                    edited = edited2

        # Post-trim without cutting sentences where possible
        if target_chars and len(edited) > target_chars:
            # try cut at last sentence boundary within target
            cut = edited[:target_chars]
            last_punct = max(cut.rfind("."), cut.rfind("!"), cut.rfind("?"))
            if last_punct > 0 and last_punct >= target_chars - 120:
                edited = cut[: last_punct + 1].strip()
            else:
                # fall back to word boundary
                edited = cut.rsplit(" ", 1)[0].strip() + "…"

        # Safety: wenn leer, dann Original zurückgeben; sonst paraphrase beibehalten
        if len(edited.strip()) == 0:
            return text.strip()
        return edited
