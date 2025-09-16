from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        if not texts:
            return []
        scores = []
        batch = 8
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            enc = self.tokenizer([query]*len(chunk), chunk, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            logits = out.logits.squeeze(-1)
            # some models have shape (bs,1), others (bs,2) -> reduce to prob for positive/relevance
            if logits.dim() == 1:
                prob = torch.sigmoid(logits)
            else:
                prob = torch.softmax(logits, dim=-1)[..., -1]
            scores.extend(prob.detach().cpu().tolist())
        # return raw probabilities [0,1]
        return [float(x) for x in scores]
