#!/usr/bin/env python3
"""
Precompute SBERT embeddings (no external API) for QA JSON datasets, with optional PCA + Int8 quantization
and write compact binary assets compatible with the web app loader under public/quant.

- Loads input JSON array of QA items (expects objects with a 'question' field)
- Computes embeddings using Sentence-Transformers (default: paraphrase-multilingual-MiniLM-L12-v2, 384 dims)
- Applies optional PCA to target dimension (default: 256)
- L2-normalizes and optionally quantizes to Int8
- Writes files:
    public/quant/<dataset_id>.embeddings.bin
    public/quant/<dataset_id>.pca_components.bin
    public/quant/<dataset_id>.pca_mean.bin
    public/quant/<dataset_id>.meta.json
    public/quant/<dataset_id>.items.json (compact items without embeddings)

Usage (Windows PowerShell example):
  python scripts/precompute_sbert_embeddings.py \
    -i src/data/qa_Klexikon-Prod-180825.json \
    --out-dir public/quant \
    --dataset-id qa_Klexikon-Prod-180825_sbert \
    --pca-dim 256 --quantize

Requires:
  pip install --upgrade sentence-transformers numpy tqdm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import numpy as np
except Exception:
    print("Missing dependency: numpy. Install with: pip install --upgrade numpy", file=sys.stderr)
    raise

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    print("Missing dependency: sentence-transformers. Install with: pip install --upgrade sentence-transformers", file=sys.stderr)
    raise

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims
DEFAULT_PCA_DIM = 256


@dataclass
class Options:
    model_id: str = DEFAULT_MODEL
    batch_size: int = 64
    format: str = "bin"  # only 'bin' supported for this script
    out_dir: str = "public/quant"
    dataset_id: Optional[str] = None
    pca_dim: int = DEFAULT_PCA_DIM
    use_pca: bool = True
    quantize: bool = True
    question_keys: Optional[List[str]] = None
    answer_keys: Optional[List[str]] = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def derive_dataset_id(input_path: str, override: Optional[str]) -> str:
    if override:
        return override
    base = os.path.basename(input_path)
    if base.lower().endswith(".json"):
        base = base[:-5]
    return base + "_sbert"


def load_items(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of QA objects")
    return data


def _first_nonempty(obj: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in obj and isinstance(obj[k], str):
            val = obj[k].strip()
            if val:
                return val
    return None


def normalize_items(items: List[Dict[str, Any]], question_keys: Optional[List[str]] = None, answer_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    # Robust defaults for German/English datasets
    q_keys = (question_keys or [
        "question", "frage", "q", "title", "titel", "prompt", "text"  # try common variants
    ])
    a_keys = (answer_keys or [
        "answer", "antwort", "a", "text", "content", "body", "antwort_text"
    ])
    out: List[Dict[str, Any]] = []
    for obj in items:
        q = _first_nonempty(obj, q_keys) or ""
        a = _first_nonempty(obj, a_keys) or ""
        rec: Dict[str, Any] = {
            "question": q,
            "answer": a,
        }
        url = obj.get("url") or obj.get("wwwurl")
        if url:
            rec["url"] = url
        if obj.get("category") or obj.get("subject"):
            rec["category"] = obj.get("category") or obj.get("subject")
        if obj.get("type"):
            rec["type"] = obj.get("type")
        if obj.get("difficulty"):
            rec["difficulty"] = obj.get("difficulty")
        if obj.get("node_id") or obj.get("id"):
            rec["node_id"] = obj.get("node_id") or obj.get("id")
        if obj.get("level"):
            rec["level"] = obj.get("level")
        out.append(rec)
    return out


def fit_pca(X: np.ndarray, target_dim: int):
    # Center
    mu = X.mean(axis=0)
    Xc = X - mu
    # Compute PCA via SVD
    U, S, VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:target_dim].T.astype(np.float32)  # (D x d)
    Xp = Xc @ comps
    # L2 normalize rows
    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xp / norms
    return mu.astype(np.float32), comps, Xn.astype(np.float32)


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (X / norms).astype(np.float32)


def quantize_int8(X: np.ndarray) -> np.ndarray:
    Xq = np.clip(np.rint(127.0 * X), -127, 127).astype(np.int8)
    return Xq


def write_assets(base: str, embeddings_bytes: bytes, comps: np.ndarray, mu: np.ndarray, meta: Dict[str, Any], compact_items: List[Dict[str, Any]]):
    emb_path = base + ".embeddings.bin"
    comp_path = base + ".pca_components.bin"
    mean_path = base + ".pca_mean.bin"
    meta_path = base + ".meta.json"
    items_path = base + ".items.json"

    with open(emb_path, "wb") as f:
        f.write(embeddings_bytes)
    with open(comp_path, "wb") as f:
        f.write(comps.astype(np.float32).tobytes(order="C"))
    with open(mean_path, "wb") as f:
        f.write(mu.astype(np.float32).tobytes(order="C"))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    with open(items_path, "w", encoding="utf-8") as f:
        json.dump(compact_items, f, ensure_ascii=False, separators=(",", ":"))

    print("Wrote compact SBERT assets under", os.path.dirname(base))
    for p in [emb_path, comp_path, mean_path, meta_path, items_path]:
        print(" -", os.path.basename(p))


def process_dataset(input_path: str, opts: Options) -> str:
    items_raw = load_items(input_path)
    items = normalize_items(items_raw, opts.question_keys, opts.answer_keys)

    # Diagnostics: show counts early
    total_items = len(items)
    non_empty_q = sum(1 for obj in items if (obj.get("question") or "").strip())
    print(f"[SBERT] Loaded items: {total_items}; non-empty 'question': {non_empty_q}")

    texts = [(i, (obj.get("question") or "").strip()) for i, obj in enumerate(items)]
    texts = [(i, t) for (i, t) in texts if t]
    if not texts:
        raise ValueError("No non-empty question fields found. Try specifying --question-key (and optionally --answer-key) to match your dataset schema.")

    model = SentenceTransformer(opts.model_id)

    batch_size = max(1, opts.batch_size)
    vecs: List[np.ndarray] = []
    idxs: List[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        batch_texts = [t for (_, t) in batch]
        emb = model.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
        # Ensure float32
        emb = emb.astype(np.float32)
        vecs.append(emb)
        idxs.extend([j for (j, _) in batch])

    X = np.vstack(vecs)  # (N, D)
    N, D0 = X.shape

    # PCA and normalization
    if opts.use_pca:
        mu, comps, Xn = fit_pca(X, int(opts.pca_dim))
    else:
        mu = X.mean(axis=0).astype(np.float32)
        comps = np.eye(D0, int(opts.pca_dim), dtype=np.float32) if int(opts.pca_dim) < D0 else np.eye(D0, dtype=np.float32)
        Xc = X - mu
        Xn = l2_normalize_rows(Xc @ comps)

    # Quantize or keep float32
    if opts.quantize:
        Xq = quantize_int8(Xn)
        emb_bytes = Xq.tobytes(order="C")
        quant_kind = "int8"
    else:
        emb_bytes = Xn.astype(np.float32).tobytes(order="C")
        quant_kind = "float32"

    # Row mapping back to original order indices
    row_to_item = np.array(idxs, dtype=np.int32)

    # Prepare meta and paths
    dataset_id = derive_dataset_id(input_path, opts.dataset_id)
    ensure_dir(opts.out_dir)
    base = os.path.join(opts.out_dir, dataset_id)

    meta = {
        "version": 1,
        "providerId": "sbert",
        "model": opts.model_id,
        "dataset_id": dataset_id,
        "source_dim": int(D0),
        "pca_dim": int(Xn.shape[1]),
        "quant": quant_kind,
        "rows": int(N),
        "files": {
            "embeddings": os.path.basename(base + ".embeddings.bin"),
            "pca_components": os.path.basename(base + ".pca_components.bin"),
            "pca_mean": os.path.basename(base + ".pca_mean.bin"),
        },
        "row_to_item_index": [int(i) for i in row_to_item.tolist()],
    }

    # Write assets
    write_assets(base, emb_bytes, comps, mu, meta, items)

    return base + ".meta.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute SBERT embeddings (no-API) for QA dataset with optional PCA+quantization")
    p.add_argument("-i", "--input", required=True, help="Path to input JSON (array of QA items)")
    p.add_argument("-c", "--batch-size", type=int, default=64, help="Batch size for encoding (default: 64)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence-Transformers model id (default: {DEFAULT_MODEL})")
    p.add_argument("--format", choices=["bin"], default="bin", help="Export format (only 'bin' supported)")
    p.add_argument("--out-dir", default="public/quant", help="Output directory for compact assets (default: public/quant)")
    p.add_argument("--dataset-id", default=None, help="Optional dataset ID to use for output file names (default: derive from input + _sbert)")
    p.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM, help="Target PCA dimension (default: 256)")
    p.add_argument("--no-pca", dest="use_pca", action="store_false", help="Disable PCA (keeps original dim)")
    p.add_argument("--quantize", dest="quantize", action="store_true", help="Enable Int8 quantization (default)")
    p.add_argument("--no-quantize", dest="quantize", action="store_false", help="Disable quantization (keep float32)")
    p.add_argument("--question-key", dest="question_keys", action="append", help="Field name for question (can be specified multiple times). Defaults include: question, frage, q, title, titel, prompt, text")
    p.add_argument("--answer-key", dest="answer_keys", action="append", help="Field name for answer (can be specified multiple times). Defaults include: answer, antwort, a, text, content, body, antwort_text")
    p.set_defaults(use_pca=True, quantize=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    opts = Options(
        model_id=args.model,
        batch_size=args.batch_size,
        format=args.format,
        out_dir=args.out_dir,
        dataset_id=args.dataset_id,
        pca_dim=args.pca_dim,
        use_pca=args.use_pca,
        quantize=args.quantize,
        question_keys=args.question_keys,
        answer_keys=args.answer_keys,
    )
    try:
        meta_path = process_dataset(args.input, opts)
        print("Done. Wrote:", meta_path)
        return 0
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
