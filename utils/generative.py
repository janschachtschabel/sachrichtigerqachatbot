from typing import List, Dict
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig


class GenerativeAnswerer:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        dtype = torch.float16 if torch.cuda.is_available() else None
        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def build_prompt(self, question: str, context_items: List[Dict]) -> Dict[str, str]:
        """Return a dict with either {'text': prompt_text} or {'chat': chat_prompt} depending on tokenizer support.
        Uses only the answers of the top-2 QA pairs to keep the input compact for small models like FLAN-T5.
        """
        # Restrict to top-2 items for better grounding on small models
        context_items = context_items[:2]

        # Build context from QA pairs (include question+answer), truncated
        def _shorten(txt: str, max_chars: int = 600) -> str:
            txt = (txt or "").strip()
            return txt if len(txt) <= max_chars else (txt[: max_chars - 1].rstrip() + "…")

        ctx_blocks = []
        for i, it in enumerate(context_items, start=1):
            q = (it.get("question") or "").strip()
            a = (it.get("answer") or it.get("text") or it.get("content") or "").strip()
            if not a:
                continue
            q = _shorten(q, 200)
            a = _shorten(a)
            block = f"Kontext {i} – Frage: {q}\nKontext {i} – Antwort: {a}"
            ctx_blocks.append(block)
        context_text = "\n\n".join(ctx_blocks)

        sentinel = "KEINE_ANTWORT"
        strict_noinfo = (
            "Zu diesem Thema liegen leider keine ausreichenden Informationen im Datensatz vor. "
            "Bitte versuche es mit einer anderen Frage oder formuliere sie konkreter."
        )

        is_gemma = "gemma" in self.model_name.lower()

        if hasattr(self.tok, "apply_chat_template") and is_gemma and not self.is_encoder_decoder:
            # Prefer chat-style prompt for Gemma with very strict grounding
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein hilfreicher deutscher Assistent. Antworte NUR basierend auf dem bereitgestellten Kontext. "
                        f"Wenn die Antwort NICHT eindeutig im Kontext steht, antworte EXAKT: '{sentinel}'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Kontext:\n{context_text}\n\n"
                        f"Frage: {question}\n\n"
                        f"Antworte präzise, knapp, auf Deutsch und ohne Spekulationen."
                    ),
                },
            ]
            chat_prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"chat": chat_prompt, "sentinel": sentinel, "noinfo": strict_noinfo}

        # Fallback/plain prompts
        if self.is_encoder_decoder:
            # FLAN‑freundliche Formulierung: Frage vor den Kontext (wichtig wegen max_length/Trunkierung)
            # Bilinguale Kurz‑Instruktion hilft kleinen Modellen
            prompt = (
                "Aufgabe: Beantworte die Frage ausschließlich auf Basis des Kontexts. "
                "Falls möglich, gib eine kurze, präzise Antwort in 1–2 Sätzen. "
                "Wenn keine relevanten Informationen im Kontext stehen, schreibe: Keine ausreichenden Informationen.\n"
                "Instruction (en): Answer only using the context. If not possible, reply: 'Keine ausreichenden Informationen.'\n\n"
                f"Frage: {question}\n\n"
                f"Kontext:\n{context_text}\n\n"
                f"Antwort:"
            )
            return {"text": prompt}
        else:
            system = (
                "Du bist ein hilfreicher deutscher Assistent. Antworte präzise und NUR basierend auf dem bereitgestellten Kontext. "
                f"Wenn Informationen fehlen, antworte EXAKT: '{sentinel}'."
            )
            prompt = (
                f"{system}\n\n"
                f"Kontext:\n{context_text}\n\n"
                f"Frage: {question}\n\n"
                f"Antwort:"
            )
            return {"text": prompt, "sentinel": sentinel, "noinfo": strict_noinfo}

    @torch.no_grad()
    def generate(self, question: str, context_items: List[Dict], target_chars: int = 350) -> Dict:
        if not context_items:
            return {
                "answer": "Zu diesem Thema liegen leider keine ausreichenden Informationen im Datensatz vor. Bitte versuche es mit einer anderen Frage oder formuliere sie konkreter.",
                "confidence": 0.0,
                "source": None,
                "fallback": True,
            }
        # restrict to top-3 for generation as well
        context_items = context_items[:3]
        prompt_dict = self.build_prompt(question, context_items)
        # Approximate tokens from characters (German): ~4 chars/token
        max_new_tokens = max(64, min(512, math.ceil(target_chars / 4)))
        if "chat" in prompt_dict:
            # Truncate long chat prompts conservatively
            inputs = self.tok(prompt_dict["chat"], return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        else:
            # For seq2seq like FLAN-T5 keep encoder input within limit
            max_len = 512 if getattr(self, "is_encoder_decoder", False) else 2048
            inputs = self.tok(prompt_dict["text"], return_tensors="pt", truncation=True, max_length=max_len).to(self.device)
        # Some decoder-only models (e.g., Pleias-RAG) don't accept token_type_ids
        try:
            model_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
        except Exception:
            # BatchEncoding may support pop directly
            if "token_type_ids" in inputs:
                inputs.pop("token_type_ids")
            model_inputs = inputs

        pad_id = self.tok.pad_token_id
        if pad_id is None and self.tok.eos_token_id is not None:
            pad_id = self.tok.eos_token_id

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=not getattr(self, "is_encoder_decoder", False),  # deterministic for seq2seq
            temperature=0.7,
            top_p=0.9,
            eos_token_id=self.tok.eos_token_id if self.tok.eos_token_id is not None else None,
            pad_token_id=pad_id,
        )
        if getattr(self, "is_encoder_decoder", False):
            gen_kwargs.update(dict(num_beams=4, early_stopping=True, length_penalty=0.9, min_new_tokens=16))
        # Seq2Seq models (e.g., FLAN‑T5) may require decoder_start_token_id
        if getattr(self, "is_encoder_decoder", False):
            if getattr(self.model.config, "decoder_start_token_id", None) is None:
                gen_kwargs["decoder_start_token_id"] = self.tok.pad_token_id

        out = self.model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        full = self.tok.decode(out[0], skip_special_tokens=True)
        # Extract the answer after the [ANTWORT] tag if present, else take tail
        if "[ANTWORT]" in full:
            answer = full.split("[ANTWORT]")[-1].strip()
        else:
            answer = full.split("\n")[-1].strip()
        # Map sentinel to friendly message (only if it's essentially just the sentinel)
        sentinel = prompt_dict.get("sentinel")
        noinfo = prompt_dict.get("noinfo")
        if sentinel:
            norm = answer.strip().strip(" .!?")
            if norm == sentinel or (sentinel in norm and len(norm) <= len(sentinel) + 8):
                answer = noinfo
        if not answer:
            answer = "Zu diesem Thema liegen leider keine ausreichenden Informationen im Datensatz vor. Bitte versuche es mit einer anderen Frage oder formuliere sie konkreter."
        # Use the best (top-1) as source for display
        source = context_items[0]
        return {
            "answer": answer,
            "confidence": 0.0,  # not applicable for generative; could map from reranker later
            "source": source,
            "fallback": False,
        }
