from typing import List, Dict
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


def _window_slices(text: str, win: int = 448, stride: int = 128, limit: int = 20):
    if len(text) <= win:
        yield (0, len(text))
        return
    start = 0
    n = 0
    while start < len(text) and n < limit:
        end = min(len(text), start + win)
        yield (start, end)
        if end >= len(text):
            break
        start = end - stride
        n += 1


class QAAnswerer:
    def __init__(self, model_name: str = "deepset/xlm-roberta-base-squad2"):
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def answer(self, question: str, context_items: List[Dict], target_chars: int = 350):
        if not context_items:
            return {"answer": "Keine relevanten Informationen gefunden.", "confidence": 0.0, "fallback": True}
        # Build combined context from top-N items (already sliced by caller)
        segments = []  # list of (start, end, item)
        parts = []
        cursor = 0
        for it in context_items:
            txt = (it.get("answer") or it.get("text") or it.get("content") or "").strip()
            if not txt:
                continue
            parts.append(txt)
            start = cursor
            end = start + len(txt)
            segments.append((start, end, it))
            cursor = end + 2  # account for separator length below
        context = "\n\n".join(parts)
        if not context.strip():
            return {"answer": "Keine verwertbaren Informationen im Kontext.", "confidence": 0.0, "fallback": True}

        best = None
        for s, e in _window_slices(context, win=448, stride=128, limit=20):
            chunk = context[s:e]
            enc = self.tok(question, chunk, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            start_logits = out.start_logits.squeeze(0)
            end_logits = out.end_logits.squeeze(0)
            start_idx = int(torch.argmax(start_logits))
            end_idx = int(torch.argmax(end_logits))
            # ensure valid span
            if end_idx < start_idx:
                continue
            score = float(torch.softmax(start_logits, dim=-1)[start_idx] * torch.softmax(end_logits, dim=-1)[end_idx])
            ans_ids = enc["input_ids"][0][start_idx:end_idx+1]
            ans = self.tok.decode(ans_ids, skip_special_tokens=True)
            # derive char positions via substring search in this chunk
            abs_start = s
            abs_end = e
            if ans:
                local_idx = chunk.find(ans)
                if local_idx != -1:
                    abs_start = s + local_idx
                    abs_end = abs_start + len(ans)
            entry = {
                "score": score,
                "answer": ans,
                "start": abs_start,
                "end": abs_end,
                "context": context,
                "fallback": False,
            }
            if not best or entry["score"] > best["score"]:
                best = entry
        if not best or not best["answer"].strip():
            # heuristic fallback
            sentences = [sent.strip() for sent in context.replace("\n", " ").split(".") if len(sent.strip()) > 10]
            answer = ". ".join(sentences[:2]).strip()
            return {
                "answer": answer,
                "confidence": float((context_items[0] or {}).get("similarity", 0.5)),
                "source": context_items[0] if context_items else None,
                "context": context,
                "fallback": True,
            }
        # Expand around span to target length using simple sentence boundaries
        span_start = max(0, min(best["start"], len(context)))
        span_end = max(span_start, min(best["end"], len(context)))
        span_len = max(1, span_end - span_start)
        target = max(100, min(800, int(target_chars)))
        need = max(0, target - span_len)
        extra_left = need // 2
        extra_right = need - extra_left

        left = max(0, span_start - extra_left)
        right = min(len(context), span_end + extra_right)

        # snap to sentence boundaries where possible
        boundary_chars = ".!?\n"
        prev_cut = context.rfind(".", 0, span_start)
        for ch in boundary_chars:
            prev_cut = max(prev_cut, context.rfind(ch, 0, span_start))
        if prev_cut != -1 and prev_cut >= left - 50:
            left = max(0, prev_cut + 1)
        next_cut = len(context)
        for ch in boundary_chars:
            pos = context.find(ch, span_end)
            if pos != -1:
                next_cut = min(next_cut, pos + 1)
        if next_cut != len(context) and next_cut <= right + 50:
            right = min(len(context), next_cut)

        final_answer = context[left:right].strip()

        best["answer"] = final_answer
        best["confidence"] = best.pop("score", 0.0)
        # Determine source passage by segment boundaries
        src_item = context_items[0] if context_items else None
        for (s0, e0, it) in segments:
            if span_start >= s0 and span_start < e0:
                src_item = it
                break
        best["source"] = src_item
        # keep combined context so the app can show the excerpt block
        best["context"] = context
        return best


def highlight_span(context: str, start: int, end: int) -> str:
    start = max(0, min(start, len(context)))
    end = max(start, min(end, len(context)))
    pre = context[:start]
    mid = context[start:end]
    post = context[end:]
    # No special yellow highlight; use same style/size as QA pairs via 'small' class
    return f"<div class='context-excerpt small'>{pre}{mid}{post}</div>"
