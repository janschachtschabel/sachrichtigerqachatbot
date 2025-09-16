import os
import json
import time
import io
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.embeddings import EmbeddingBackend, ensure_dir
from utils.search import vector_search
from utils.reranker import CrossEncoderReranker
from utils.qa import QAAnswerer, highlight_span
from utils.generative import GenerativeAnswerer
from utils.post_edit import PostEditor
from utils.eval import QuizEvaluator

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
EMB_DIR = os.path.join(APP_DIR, "embeddings")
ensure_dir(DATA_DIR)
ensure_dir(EMB_DIR)

# -----------------------------
# Helpers: list available embedding datasets
# -----------------------------
def list_embedding_sets() -> List[str]:
    if not os.path.isdir(EMB_DIR):
        return []
    names = []
    for fn in os.listdir(EMB_DIR):
        if not fn.endswith('.npz'):
            continue
        base = fn[:-4]
        items_path = os.path.join(EMB_DIR, base + '_items.json')
        meta_path = os.path.join(EMB_DIR, base + '.json')
        if os.path.exists(items_path) and os.path.exists(meta_path):
            names.append(base)
    names.sort()
    return names

# -----------------------------
# Streamlit Page Config & Theme
# -----------------------------
st.set_page_config(
    page_title="QA Bot (Streamlit)",
    page_icon="🤖",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.small { font-size: 0.85rem; color: #64748b; }
.badge { display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-green { background: #e8f5e9; color: #2e7d32; }
.badge-yellow { background: #fff8e1; color: #ef6c00; }
.badge-blue { background: #e3f2fd; color: #1565c0; }
.badge-purple { background: #f3e5f5; color: #6a1b9a; }
.card { border-left: 4px solid #3b82f6; background: #f8fafc; padding: 12px; border-radius: 8px; }
.card-success { border-left: 4px solid #22c55e; background: #ecfdf5; padding: 12px; border-radius: 8px; }
.card-info { border-left: 4px solid #3b82f6; background: #e3f2fd; padding: 12px; border-radius: 8px; }
.card-gray { border-left: 4px solid #6b7280; background: #f3f4f6; padding: 12px; border-radius: 8px; margin: 6px 0; }
.codebox { background: #0b1020; color: #e5e7eb; padding: 10px; border-radius: 6px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; }
.hl { background: #fde68a; color: #111827; padding: 0 2px; border-radius: 4px; }
.context-excerpt { border-left: 4px solid #3b82f6; background: #add8e6; padding: 12px; border-radius: 8px; margin: 6px 0; }
.qa-pair { background: #f3f4f6; padding: 12px; border-radius: 8px; margin: 6px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Session State Defaults
# -----------------------------
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
if "qa_model" not in st.session_state:
    st.session_state.qa_model = "deutsche-telekom/electra-base-de-squad2"
if "reranker_model" not in st.session_state:
    st.session_state.reranker_model = "cross-encoder/ms-marco-MiniLM-L-12-v2"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "answer_mode" not in st.session_state:
    st.session_state.answer_mode = "QA"  # or "Generativ"
if "gen_model" not in st.session_state:
    st.session_state.gen_model = "PleIAs/Pleias-RAG-350M"
if "embed_top_k" not in st.session_state:
    st.session_state.embed_top_k = 50  # Anzahl Embedding-Kandidaten
if "reranker_check_k" not in st.session_state:
    st.session_state.reranker_check_k = 20  # Wie viele prüft der Cross-Encoder
if "display_k" not in st.session_state:
    st.session_state.display_k = 5  # Wie viele Ergebnisse anzeigen
if "qa_top_n" not in st.session_state:
    st.session_state.qa_top_n = 5  # Wie viele Inhalte einzeln vom QA geprüft werden
if "qa_conf_threshold" not in st.session_state:
    st.session_state.qa_conf_threshold = 0.2  # Mindestvertrauen (QA) für Übernahme
if "quiz_chat" not in st.session_state:
    st.session_state.quiz_chat: List[Dict] = []
if "quiz_mode" not in st.session_state:
    st.session_state.quiz_mode = "await_topic"  # await_topic, await_answer, await_restart, done
if "quiz_candidates" not in st.session_state:
    st.session_state.quiz_candidates: List[Dict] = []
if "quiz_index" not in st.session_state:
    st.session_state.quiz_index = 0
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""
if "quiz_dataset" not in st.session_state:
    st.session_state.quiz_dataset = None
if "quiz_eval_model" not in st.session_state:
    st.session_state.quiz_eval_model = "Embedding (Standard)"
if "quiz_clear_input" not in st.session_state:
    st.session_state.quiz_clear_input = False

# Lazy loaders
@st.cache_resource(show_spinner=False)
def get_embedding_backend(model_name: str):
    return EmbeddingBackend(model_name)

@st.cache_resource(show_spinner=False)
def get_reranker(model_name: str):
    return CrossEncoderReranker(model_name)

@st.cache_resource(show_spinner=False)
def get_qa(model_name: str, api_version: str = "v2"):
    # api_version is a dummy argument to bust cache when we change QA API
    return QAAnswerer(model_name)

@st.cache_resource(show_spinner=False)
def get_generative(model_name: str, api_version: str = "v1"):
    # api_version is a dummy argument to bust cache when we change Generative API
    return GenerativeAnswerer(model_name)

@st.cache_resource(show_spinner=False)
def get_post_editor(model_name: str, api_version: str = "v1"):
    # api_version busts cache when PostEditor changes
    return PostEditor(model_name)

@st.cache_resource(show_spinner=False)
def get_quiz_evaluator(model_name: str):
    return QuizEvaluator(model_name)

# -----------------------------
# Helpers
# -----------------------------
def load_dataset(file_or_path) -> List[Dict]:
    if isinstance(file_or_path, str):
        path = file_or_path
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = file_or_path.getvalue().decode("utf-8")
    try:
        # JSON Lines
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                items = json.loads(text)
                break
        return items
    except Exception:
        # CSV fallback
        df = pd.read_csv(io.StringIO(text))
        return df.to_dict(orient="records")


def save_embeddings(name: str, items: List[Dict], embeddings: np.ndarray, meta: Dict):
    ensure_dir(EMB_DIR)
    base = os.path.join(EMB_DIR, name)
    np.savez_compressed(base + ".npz", embeddings=embeddings)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "count": len(items)}, f, ensure_ascii=False, indent=2)
    with open(base + "_items.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_embeddings(name: str) -> Tuple[List[Dict], np.ndarray, Dict]:
    base = os.path.join(EMB_DIR, name)
    items = json.load(open(base + "_items.json", "r", encoding="utf-8"))
    data = np.load(base + ".npz")
    meta = json.load(open(base + ".json", "r", encoding="utf-8"))
    return items, data["embeddings"], meta.get("meta", {})

# -----------------------------
# UI Tabs
# -----------------------------
tab_search, tab_quiz, tab_precompute, tab_settings, tab_about = st.tabs(["🔎 Suche", "🧠 Quizz", "⚙️ Precompute", "🛠️ Einstellungen", "ℹ️ Über"])

# -----------------------------
# Tab: Einstellungen
# -----------------------------
with tab_settings:
    st.subheader("Modelle und Parameter")
    st.session_state.embed_model = st.selectbox(
        "Embedding‑Modell",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
        ],
        index=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
        ].index(st.session_state.embed_model) if st.session_state.embed_model in [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small",
        ] else 0,
    )
    st.session_state.reranker_model = st.selectbox(
        "Cross‑Encoder (Re‑Ranker)",
        [
            "Deaktiviert (nur Embedding)",
            "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",  # stärker
            "cross-encoder/ms-marco-electra-base",     # stärker (größer)
        ],
        index=[
            "Deaktiviert (nur Embedding)",
            "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-electra-base",
        ].index(st.session_state.reranker_model) if st.session_state.reranker_model in [
            "Deaktiviert (nur Embedding)",
            "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-electra-base",
        ] else 0,
    )
    st.session_state.answer_mode = st.radio("Antwort‑Modus", ["QA", "Generativ"], horizontal=True)
    if st.session_state.answer_mode == "QA":
        st.session_state.qa_model = st.selectbox(
            "QA‑Modell (extraktiv)",
            [
                "deepset/xlm-roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "deepset/gelectra-base-germanquad-distilled",
                "deepset/gelectra-large-germanquad",
                "deutsche-telekom/electra-base-de-squad2",
                "deutsche-telekom/bert-multi-english-german-squad2",
                "svalabs/infoxlm-german-question-answering",
                "LLukas22/all-MiniLM-L12-v2-qa-all",
            ],
            index=[
                "deepset/xlm-roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "deepset/gelectra-base-germanquad-distilled",
                "deepset/gelectra-large-germanquad",
                "deutsche-telekom/electra-base-de-squad2",
                "deutsche-telekom/bert-multi-english-german-squad2",
                "svalabs/infoxlm-german-question-answering",
                "LLukas22/all-MiniLM-L12-v2-qa-all",
            ].index(st.session_state.qa_model) if st.session_state.qa_model in [
                "deepset/xlm-roberta-base-squad2",
                "distilbert-base-cased-distilled-squad",
                "deepset/gelectra-base-germanquad-distilled",
                "deepset/gelectra-large-germanquad",
                "deutsche-telekom/electra-base-de-squad2",
                "deutsche-telekom/bert-multi-english-german-squad2",
                "svalabs/infoxlm-german-question-answering",
                "LLukas22/all-MiniLM-L12-v2-qa-all",
            ] else 0,
        )
        _pe_options = [
            "Deaktiviert",
            "Shahm/t5-small-german",
            "aiassociates/t5-small-grammar-correction-german",
            "google/mt5-small",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "PleIAs/Pleias-RAG-350M",
            "PleIAs/Pleias-RAG-1B",
            "Qwen/Qwen2-0.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "google/gemma-3-1b-it",
            "google/gemma-2-2b-it",
            "microsoft/Phi-3-mini-4k-instruct",
            "utter-project/EuroLLM-1.7B",
        ]
        if "qa_post_edit" not in st.session_state:
            st.session_state.qa_post_edit = _pe_options[0]
        st.session_state.qa_post_edit = st.selectbox(
            "QA Nachbearbeitung (Grammatik/Satzbau)",
            _pe_options,
            index=_pe_options.index(st.session_state.qa_post_edit) if st.session_state.qa_post_edit in _pe_options else 0,
            help="Optional: Ein kleines LLM poliert Satzbau/Grammatik der QA‑Antwort, ohne neue Fakten hinzuzufügen."
        )
    else:
        st.session_state.gen_model = st.selectbox(
            "Generatives Modell (klein)",
            [
                "PleIAs/Pleias-RAG-350M",
                "PleIAs/Pleias-RAG-1B",
                "Shahm/t5-small-german",
                "google/gemma-3-1b-it",
                "utter-project/EuroLLM-1.7B",
                "Qwen/Qwen2-0.5B-Instruct",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "google/gemma-2-2b-it",
                "microsoft/Phi-3-mini-4k-instruct",
            ],
            index=0,
            help="Hinweis: Größere Modelle benötigen mehr RAM/VRAM; 0.5B/1.1B laufen i.d.R. auch auf CPU."
        )
    colA, colB = st.columns(2)
    with colA:
        st.session_state.threshold = st.slider("Schwelle (Embedding & Re‑Ranker, ≥)", 0.0, 1.0, st.session_state.threshold, 0.05)
    with colB:
        st.session_state.qa_conf_threshold = st.slider("QA Confidence‑Schwelle (≥)", 0.0, 1.0, st.session_state.qa_conf_threshold, 0.05)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.embed_top_k = st.number_input("Embedding Top‑K (Kandidaten)", min_value=1, max_value=1000, value=st.session_state.embed_top_k, step=1)
    with col2:
        st.session_state.reranker_check_k = st.number_input("Cross‑Encoder Prüf‑K", min_value=1, max_value=500, value=st.session_state.reranker_check_k, step=1)
    with col3:
        st.session_state.display_k = st.number_input("Anzeige Top‑K", min_value=1, max_value=100, value=st.session_state.display_k, step=1)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.session_state.qa_top_n = st.slider("QA Top‑N (einzeln prüfen)", 1, 10, st.session_state.qa_top_n, 1)
    with col_q2:
        st.caption("QA prüft jedes der Top‑N Dokumente einzeln; nur Antworten mit genügend Vertrauen werden übernommen.")

    # Answer length control
    if "answer_len_chars" not in st.session_state:
        st.session_state.answer_len_chars = 2000
    st.session_state.answer_len_chars = st.slider("Antwort‑Länge (Ziel, Zeichen)", 100, 2000, st.session_state.answer_len_chars, 50)

    st.info("Modelle werden on‑demand geladen. Falls Downloads lange dauern, prüfe Internet/Zeit‑outs.")

# -----------------------------
# Tab: Precompute
# -----------------------------
with tab_precompute:
    st.subheader("Embeddings vorab berechnen")
    st.markdown("Lade ein Dataset (JSONL/JSON/CSV) mit Feldern `question`, `answer`, `wwwurl`.")

    uploaded = st.file_uploader("Dataset hochladen", type=["jsonl", "json", "csv"])
    default_name = st.text_input("Name für Embeddings (Dateibasis)", "mein_dataset")
    text_mode = st.radio("Textbasis für Embedding", ["answer", "question", "question+answer"], index=2)

    if st.button("Precompute starten", type="primary", disabled=uploaded is None):
        if uploaded is None:
            st.warning("Bitte Dataset hochladen.")
        else:
            items = load_dataset(uploaded)
            st.write(f"Gefundene Einträge: {len(items)}")
            with st.spinner("Lade Embedding‑Modell…"):
                emb = get_embedding_backend(st.session_state.embed_model)
            texts = []
            for it in items:
                q = (it.get("question") or "").strip()
                a = (it.get("answer") or "").strip()
                if text_mode == "answer":
                    texts.append(a)
                elif text_mode == "question":
                    texts.append(q)
                else:
                    texts.append((q + "\n\n" + a).strip())

            progress = st.progress(0)
            batch = 64
            all_vecs = []
            for i in range(0, len(texts), batch):
                chunk = texts[i:i+batch]
                vecs = emb.encode(chunk)
                all_vecs.append(vecs)
                progress.progress(min(1.0, (i+batch)/max(1,len(texts))))
            embeddings = np.vstack(all_vecs) if all_vecs else np.zeros((0, emb.dim), dtype=np.float32)

            meta = {
                "model": st.session_state.embed_model,
                "text_mode": text_mode,
                "dim": emb.dim,
                "created": int(time.time()),
            }
            save_embeddings(default_name, items, embeddings, meta)
            st.success(f"Fertig. Gespeichert unter embeddings/{default_name}.npz + .json + _items.json")

# -----------------------------
# Tab: Suche
# -----------------------------
with tab_search:
    st.subheader("Suche & QA")

    left, right = st.columns([3, 2])
    with left:
        query = st.text_input("Deine Frage", "Warum gibt es viele Vulkane auf Island?")
        # Auto-detect available embedding datasets and allow selection
        col_ds, col_refresh = st.columns([4,1])
        with col_ds:
            available_sets = list_embedding_sets()
            if not available_sets:
                st.info("Kein Embedding‑Datensatz gefunden. Bitte zuerst im Tab Precompute erstellen.")
            dataset_name = st.selectbox(
                "Embeddings‑Datensatz",
                options=available_sets,
                index=0 if available_sets else 0,
                disabled=(len(available_sets) == 0),
            )
        with col_refresh:
            if st.button("🔄 Aktualisieren"):
                st.rerun()
        run = st.button("Suchen", type="primary")
    with right:
        st.markdown("**Status**")
        st.markdown(f"<span class='badge badge-blue'>Embedding: {st.session_state.embed_model}</span>", unsafe_allow_html=True)
        rr_label = (
            "nur Embedding" if st.session_state.reranker_model == "Deaktiviert (nur Embedding)" else st.session_state.reranker_model
        )
        st.markdown(f"<span class='badge badge-purple'>Re‑Ranker: {rr_label}</span>", unsafe_allow_html=True)
        if st.session_state.answer_mode == "QA":
            st.markdown(f"<span class='badge badge-green'>QA: {st.session_state.qa_model}</span>", unsafe_allow_html=True)
            # Show post-edit model if enabled
            if st.session_state.get("qa_post_edit") and st.session_state.qa_post_edit != "Deaktiviert":
                st.markdown(
                    f"<span class='badge badge-yellow'>Post‑Edit: {st.session_state.qa_post_edit}</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(f"<span class='badge badge-green'>Generativ: {st.session_state.gen_model}</span>", unsafe_allow_html=True)

    if run and query.strip():
        if not list_embedding_sets():
            st.warning("Bitte zuerst einen Embeddings‑Datensatz über Precompute erzeugen und auswählen.")
            st.stop()
        try:
            items, embeddings, meta = load_embeddings(dataset_name)
        except Exception as e:
            st.error(f"Konnte Embeddings '{dataset_name}' nicht laden: {e}")
            st.stop()

        with st.spinner("Lade Modelle…"):
            emb = get_embedding_backend(st.session_state.embed_model)
            reranker = None
            reranker_enabled = st.session_state.reranker_model != "Deaktiviert (nur Embedding)"
            if reranker_enabled:
                reranker = get_reranker(st.session_state.reranker_model)
            evaluator = None
            if st.session_state.quiz_eval_model.startswith("Generativ"):
                eval_model = "google/mt5-small" if "mt5" in st.session_state.quiz_eval_model else "Shahm/t5-small-german"
                evaluator = get_quiz_evaluator(eval_model)
            qa = None
            gen = None
            if st.session_state.answer_mode == "QA":
                qa = get_qa(st.session_state.qa_model, api_version="v3")
            else:
                gen = get_generative(st.session_state.gen_model, api_version="v2")

        # Search
        q_vec = emb.encode([query])[0]
        sims, idx = vector_search(q_vec, embeddings, top_k=min(int(st.session_state.embed_top_k), len(items)))
        results = []
        for s, i in zip(sims, idx):
            if s < st.session_state.threshold:
                continue
            doc = items[int(i)].copy()
            doc["embedding_sim"] = float(s)
            doc["text_for_rerank"] = (doc.get("question", "") + "\n\n" + doc.get("answer", "")).strip()
            results.append(doc)

        if not results:
            st.warning(f"Keine Treffer über der Embedding‑Schwelle (≥ {st.session_state.threshold*100:.0f}%).")
            st.stop()

        # Re-rank (optional) or keep embedding order
        if reranker is not None:
            to_rerank = results[: min(int(st.session_state.reranker_check_k), len(results))]
            rerank_texts = [r["text_for_rerank"] for r in to_rerank]
            scores = reranker.rerank(query, rerank_texts)
            for doc, sc in zip(to_rerank, scores):
                doc["reranker_score"] = float(sc)
            # Enforce reranker threshold (≥ threshold)
            results = [d for d in to_rerank if d.get("reranker_score", 0) >= st.session_state.threshold]
            if not results:
                st.warning(f"Keine Treffer nach Re‑Ranker‑Schwelle (≥ {st.session_state.threshold*100:.0f}%).")
                st.stop()
            results.sort(key=lambda d: d.get("reranker_score", 0), reverse=True)
        else:
            # Only embedding filter/sort
            results.sort(key=lambda d: d.get("embedding_sim", 0), reverse=True)
            # If desired, still apply a minimum embedding threshold (already applied above)

        # Prepare for answer generation
        best = results[0]
        top_n = 3 if len(results) >= 3 else (2 if len(results) >= 2 else 1)
        context_passages = results[:top_n]
        with st.spinner("Generiere Antwort…"):
            if st.session_state.answer_mode == "QA":
                # Aggregierte QA: jedes Dokument einzeln prüfen und Sätze zusammenfügen
                accepted_sentences: List[str] = []
                accepted_confs: List[float] = []
                accepted_sources: List[Dict] = []
                qa_checks = min(int(st.session_state.qa_top_n), len(results))
                per_target = max(120, min(300, st.session_state.answer_len_chars // max(1, qa_checks)))
                for doc in results[:qa_checks]:
                    try:
                        single = qa.answer(query, [doc], target_chars=per_target)
                    except TypeError:
                        single = qa.answer(query, [doc])
                    conf = float(single.get("confidence", 0.0))
                    if (not single.get("fallback")) and conf >= float(st.session_state.qa_conf_threshold):
                        text = (single.get("answer", "") or "").strip()
                        if text:
                            # Split into sentences
                            parts = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", text) if len(s.strip()) > 0]
                            accepted_sentences.extend(parts)
                            accepted_confs.append(conf)
                            accepted_sources.append(doc)
                if accepted_sentences:
                    # De-duplicate while preserving order
                    seen = set()
                    unique = []
                    for snt in accepted_sentences:
                        if snt not in seen:
                            unique.append(snt)
                            seen.add(snt)
                    # Build up to target length without cutting sentences
                    target = int(st.session_state.answer_len_chars)
                    parts = []
                    total_len = 0
                    for snt in unique:
                        add_len = len(snt) + (1 if parts else 0)
                        if total_len + add_len <= target:
                            parts.append(snt)
                            total_len += add_len
                        else:
                            break
                    if not parts and unique:
                        # If even first sentence doesn't fit fully, cut at word boundary
                        clipped = unique[0][:target]
                        clipped = clipped.rsplit(" ", 1)[0].strip() + "…"
                        combined = clipped
                    else:
                        combined = " ".join(parts).strip()
                    qa_res = {
                        "answer": combined,
                        "confidence": float(np.mean(accepted_confs)) if accepted_confs else 0.0,
                        "source": accepted_sources[0] if accepted_sources else best,
                        "fallback": False,
                        "aggregate": True,
                    }
                else:
                    qa_res = {"answer": "", "confidence": 0.0, "source": best, "fallback": True, "aggregate": True}

                # Optional: QA Nachbearbeitung (Grammatik/Satzbau) mit kleinem LLM
                if (
                    st.session_state.get("qa_post_edit", "Deaktiviert") != "Deaktiviert"
                    and not qa_res.get("fallback")
                    and (qa_res.get("answer") or "").strip()
                ):
                    try:
                        editor = get_post_editor(st.session_state.qa_post_edit, api_version="v1")
                        _orig_ans = qa_res.get("answer", "")
                        edited = editor.edit(_orig_ans, target_chars=int(st.session_state.answer_len_chars))
                        if isinstance(edited, str) and len(edited.strip()) > 0:
                            qa_res["answer"] = edited.strip()
                            qa_res["post_edited"] = (qa_res["answer"].strip() != (_orig_ans or "").strip())
                    except Exception as _e:
                        # Silently ignore post-edit errors, keep original QA answer
                        pass
            else:
                # generative
                qa_res = gen.generate(query, context_passages, target_chars=st.session_state.answer_len_chars)

        # Answer Card (green background) — mirror result formatting
        qa_conf_pct = qa_res.get('confidence', 0.0) * 100.0
        src = qa_res.get('source') or best
        src_url = src.get("wwwurl") or src.get("url") or src.get("link")
        total = len(results)
        rank_label = "Re‑Ranker Rang" if (st.session_state.reranker_model != "Deaktiviert (nur Embedding)") else "Embedding Rang"
        rank_badge = f"<span class='badge badge-purple'>{rank_label}: 1/{total}</span>"
        sim_pct = src.get('embedding_sim', 0.0) * 100.0
        rr_pct = src.get('reranker_score')
        trust_label = "Vertrauen (QA)" if st.session_state.answer_mode == "QA" else "Vertrauen (LLM)"
        rr_part = (f" · Re‑Ranker: {rr_pct*100:.1f}%" if rr_pct is not None else "")
        pe_part = (" · Nachbearbeitet" if qa_res.get("post_edited") else "")
        meta_line = (
            f"Wahrscheinlichkeit: {sim_pct:.1f}%{rr_part}{pe_part}"
            + f" • {trust_label}: {qa_conf_pct:.1f}%"
            + (f" • <a href='{src_url}' target='_blank'>Quelle</a>" if src_url else "")
        )
        # Standard message if QA fell back (no span)
        display_answer = qa_res.get('answer', '')
        if qa_res.get('fallback'):
            display_answer = (
                "Zu diesem Thema liegen leider keine ausreichenden Informationen im Datensatz vor. "
                "Bitte versuche es mit einer anderen Frage oder formuliere sie konkreter."
            )
        mode_badge = "<span class='badge badge-green'>Modell</span>" if st.session_state.answer_mode == "QA" else "<span class='badge badge-green'>Generativ</span>"
        qa_html = (
            f"<div class='card-success'>"
            f"<div><strong>🤖 KI‑Antwort</strong> {mode_badge} {rank_badge}</div>"
            f"<div class='small'>{meta_line}</div>"
            f"<div class='small'>{display_answer}</div>"
            f"</div>"
        )
        st.markdown(qa_html, unsafe_allow_html=True)

        # Debug span highlight (nur für Einzel-Span, nicht für Aggregat)
        if (not qa_res.get("aggregate")) and qa_res.get("start") is not None and qa_res.get("end") is not None:
            st.markdown("**Kontextausschnitt**")
            st.markdown(highlight_span(qa_res["context"], qa_res["start"], qa_res["end"]), unsafe_allow_html=True)

        # Show results (unterhalb der KI‑Antwort)
        st.markdown("### Suchergebnisse")
        display_results = results[: min(int(st.session_state.display_k), len(results))]
        total = len(display_results)
        rank_label = "Re‑Ranker Rang" if (st.session_state.reranker_model != "Deaktiviert (nur Embedding)") else "Embedding Rang"
        for i, r in enumerate(display_results, start=1):
            url = r.get("wwwurl") or r.get("url") or r.get("link")
            header = f"{i}. {r.get('question','(ohne Frage)')}"
            rank_badge = f"<span class='badge badge-purple'>{rank_label}: {i}/{total}</span>"
            score_line = (
                f"Wahrscheinlichkeit: {r['embedding_sim']*100:.1f}%"
                + (f" · Re‑Ranker: {r['reranker_score']*100:.1f}%" if 'reranker_score' in r else "")
            )
            meta_line = score_line + (f" • <a href='{url}' target='_blank'>Quelle</a>" if url else "")
            st.markdown(
                f"<div class='card-gray'><div><strong>{header}</strong> {rank_badge}</div>"
                f"<div class='small'>{meta_line}</div>"
                f"<div class='small'>{r.get('answer','')}</div></div>",
                unsafe_allow_html=True,
            )

# -----------------------------
# Tab: Quizz
# -----------------------------
with tab_quiz:
    st.subheader("Quiz‑Modus (Chat)")

    # Dataset Auswahl (separat vom Such‑Tab)
    available_sets = list_embedding_sets()
    if not available_sets:
        st.info("Kein Embedding‑Datensatz gefunden. Bitte zuerst im Tab Precompute erstellen.")
    st.session_state.quiz_dataset = st.selectbox(
        "Embeddings‑Datensatz",
        options=available_sets,
        index=0 if available_sets else 0,
        disabled=(len(available_sets) == 0),
        key="quiz_dataset_select",
    )
    st.session_state.quiz_eval_model = st.selectbox(
        "Bewertung",
        [
            "Embedding (Standard)",
            "Generativ (google/mt5-small)",
            "Generativ (Shahm/t5-small-german)",
        ],
        index=["Embedding (Standard)", "Generativ (google/mt5-small)", "Generativ (Shahm/t5-small-german)"].index(st.session_state.quiz_eval_model)
        if st.session_state.quiz_eval_model in ["Embedding (Standard)", "Generativ (google/mt5-small)", "Generativ (Shahm/t5-small-german)"] else 0,
        help="Wie soll die Korrektheit bewertet werden? Embedding ist schnell; generativ liefert eine robustere Prozentbewertung.",
        key="quiz_eval_select",
    )

    col_qz1, col_qz2 = st.columns([1,1])
    with col_qz1:
        reset = st.button("🔄 Quiz zurücksetzen")
    with col_qz2:
        st.caption("Nur Chat: Es werden keine Trefferlisten/Kontexte angezeigt.")

    if reset:
        st.session_state.quiz_chat = []
        st.session_state.quiz_mode = "await_topic"
        st.session_state.quiz_candidates = []
        st.session_state.quiz_index = 0
        st.session_state.quiz_topic = ""

    # Initiale Begrüßung
    if len(st.session_state.quiz_chat) == 0 and st.session_state.quiz_mode == "await_topic":
        st.session_state.quiz_chat.append({
            "role": "assistant",
            "text": "Hallo! Ich bin dein Quiz‑Bot. Nenne mir bitte ein Thema, zu dem du üben möchtest (z. B. Vulkanismus, Fotosynthese, Bruchrechnung)."
        })

    # Chat‑Verlauf anzeigen
    for m in st.session_state.quiz_chat:
        if m.get("role") == "assistant":
            st.markdown(f"<div class='card-info'>🤖 {m.get('text','')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='card-gray'>🙋 {m.get('text','')}</div>", unsafe_allow_html=True)

    # Eingabefeld ggf. leeren (muss VOR Widget-Erzeugung passieren)
    if st.session_state.get("quiz_clear_input", False):
        st.session_state["quiz_user_input"] = ""
        st.session_state["quiz_clear_input"] = False
    # Eingabezeile
    user_msg = st.text_input("Deine Eingabe", st.session_state.get("quiz_user_input", ""), key="quiz_user_input")
    send = st.button("Senden", key="quiz_send")

    def _shorten(txt: str, max_chars: int = 220) -> str:
        txt = (txt or "").strip()
        if len(txt) <= max_chars:
            return txt
        cut = txt[:max_chars]
        # nicht mitten im Wort abbrechen
        return cut.rsplit(" ", 1)[0].strip() + "…"

    def _hint_from_answer(ans: str) -> str:
        # Nimm den ersten Satz als Tipp
        sents = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", (ans or "").strip()) if len(s.strip()) > 0]
        if sents:
            return _shorten(sents[0], 140)
        return "Denke an zentrale Begriffe und erkläre in einem vollständigen Satz."

    if send and user_msg.strip():
        st.session_state.quiz_chat.append({"role": "user", "text": user_msg.strip()})

        # Sicherstellen, dass ein Datensatz gewählt ist
        if not st.session_state.quiz_dataset:
            st.session_state.quiz_chat.append({
                "role": "assistant",
                "text": "Bitte wähle oben einen Embeddings‑Datensatz aus, bevor wir starten."
            })
        else:
            # Lazy load Backends
            emb = get_embedding_backend(st.session_state.embed_model)
            reranker = None
            reranker_enabled = st.session_state.reranker_model != "Deaktiviert (nur Embedding)"
            if reranker_enabled:
                reranker = get_reranker(st.session_state.reranker_model)
            evaluator = None
            if st.session_state.quiz_eval_model.startswith("Generativ"):
                eval_model = "google/mt5-small" if "mt5" in st.session_state.quiz_eval_model else "Shahm/t5-small-german"
                evaluator = get_quiz_evaluator(eval_model)

            # Lade Embeddings/Dataset bei Bedarf
            try:
                items, embeddings, _meta = load_embeddings(st.session_state.quiz_dataset)
            except Exception as e:
                st.session_state.quiz_chat.append({
                    "role": "assistant",
                    "text": f"Konnte Embeddings nicht laden: {e}"
                })
                items, embeddings = [], np.zeros((0, 1), dtype=np.float32)

            if st.session_state.quiz_mode == "await_topic":
                st.session_state.quiz_topic = user_msg.strip()
                # Suche bis zu 10 passende QA‑Paare, Mindest‑Embedding ≥ 50%
                q_vec = emb.encode([st.session_state.quiz_topic])[0]
                sims, idx = vector_search(q_vec, embeddings, top_k=min(10, len(items)))
                results = []
                for s, i in zip(sims, idx):
                    if s < 0.5:
                        continue
                    doc = items[int(i)].copy()
                    doc["embedding_sim"] = float(s)
                    doc["text_for_rerank"] = (doc.get("question", "") + "\n\n" + doc.get("answer", "")).strip()
                    results.append(doc)
                if not results:
                    st.session_state.quiz_chat.append({
                        "role": "assistant",
                        "text": "Ich habe leider keine passenden Fragen über 50% Ähnlichkeit gefunden. Bitte wähle ein anderes Thema oder formuliere es konkreter."
                    })
                else:
                    # Optionales Re‑Ranking (gleiches Verfahren wie Suche)
                    if reranker is not None:
                        rerank_texts = [r["text_for_rerank"] for r in results]
                        scores = reranker.rerank(st.session_state.quiz_topic, rerank_texts)
                        for doc, sc in zip(results, scores):
                            doc["reranker_score"] = float(sc)
                        results = [d for d in results if d.get("reranker_score", 0) >= st.session_state.threshold]
                        results.sort(key=lambda d: d.get("reranker_score", 0), reverse=True)
                        if not results:
                            st.session_state.quiz_chat.append({
                                "role": "assistant",
                                "text": "Nach Re‑Ranking keine passenden Fragen oberhalb der Schwelle. Bitte Thema anpassen."
                            })
                    else:
                        results.sort(key=lambda d: d.get("embedding_sim", 0), reverse=True)

                    if results:
                        st.session_state.quiz_candidates = results
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_mode = "await_answer"
                        q0 = (results[0].get("question") or "").strip() or "Hier ist eine Einstiegsfrage: Was weißt du darüber?"
                        st.session_state.quiz_chat.append({
                            "role": "assistant",
                            "text": f"Prima, wir starten das Quiz zum Thema „{st.session_state.quiz_topic}“.\n\nFrage 1: {q0}"
                        })

            elif st.session_state.quiz_mode == "await_answer":
                # Prüfe Antwort gegen aktuelle Ziel‑Antwort
                if not st.session_state.quiz_candidates:
                    st.session_state.quiz_chat.append({"role": "assistant", "text": "Kein Fragenpool geladen. Bitte gib ein Thema ein."})
                else:
                    idx = st.session_state.quiz_index
                    idx = max(0, min(idx, len(st.session_state.quiz_candidates) - 1))
                    doc = st.session_state.quiz_candidates[idx]
                    gold = (doc.get("answer") or "").strip()
                    # Bewertung (Embedding oder Generativ)
                    if evaluator is None:
                        # Embedding-basiert: Cosine in [−1,1] → [0,1], mit Jaccard-Überlappung
                        u, g = user_msg.strip(), gold
                        vec_u, vec_g = emb.encode([u, g])
                        # vec_u/vec_g sind bereits normalisiert (cos = dot)
                        cos = float(np.dot(vec_u, vec_g))
                        sem = max(0.0, min(1.0, (cos + 1.0) / 2.0))
                        # einfache Token-Überlappung
                        # Fallback mit Python re (ohne \p{L})
                        tok = lambda s: set(re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", s.lower()))
                        # Wenn das 'regex'-Modul verfügbar ist, nutze \p{L}-Klassen
                        try:
                            import regex as re2
                            tok = lambda s: set(re2.findall(r"\p{L}+[\p{L}0-9-]*|\d+", s.lower()))
                        except Exception:
                            pass
                        a = tok(u); b = tok(g)
                        jac = (len(a & b) / len(a | b)) if (a and b) else 0.0
                        conf = 0.65 * sem + 0.35 * jac
                    else:
                        # Generativer Bewerter liefert 0..1
                        try:
                            conf = evaluator.score(user_msg.strip(), gold)
                        except Exception:
                            conf = 0.0

                    if conf >= 0.75:
                        # korrekt – vollständige Lösung zeigen, dann weiter
                        st.session_state.quiz_chat.append({
                            "role": "assistant",
                            "text": f"Richtig! ✅ (Bewertung: {conf*100:.0f}%)\n\nLösung: {gold}"
                        })
                        # nächste Frage oder Abschluss
                        st.session_state.quiz_index += 1
                        if st.session_state.quiz_index < len(st.session_state.quiz_candidates):
                            qn = st.session_state.quiz_candidates[st.session_state.quiz_index].get("question") or "Weiter geht's:"
                            st.session_state.quiz_chat.append({
                                "role": "assistant",
                                "text": f"Nächste Frage ({st.session_state.quiz_index+1}/{len(st.session_state.quiz_candidates)}): {qn}"
                            })
                        else:
                            st.session_state.quiz_mode = "await_restart"
                            st.session_state.quiz_chat.append({
                                "role": "assistant",
                                "text": "Super, das waren alle Fragen zu deinem Thema. Möchtest du ein anderes Thema üben? (ja/nein)"
                            })
                    elif conf >= 0.4:
                        # ausreichend für Weitergang – vollständige Lösung zeigen, dann weiter
                        st.session_state.quiz_chat.append({
                            "role": "assistant",
                            "text": f"Fast! (Bewertung: {conf*100:.0f}%) Hier ist die vollständige Lösung:\n\n{gold}"
                        })
                        st.session_state.quiz_index += 1
                        if st.session_state.quiz_index < len(st.session_state.quiz_candidates):
                            qn = st.session_state.quiz_candidates[st.session_state.quiz_index].get("question") or "Weiter geht's:"
                            st.session_state.quiz_chat.append({
                                "role": "assistant",
                                "text": f"Nächste Frage ({st.session_state.quiz_index+1}/{len(st.session_state.quiz_candidates)}): {qn}"
                            })
                        else:
                            st.session_state.quiz_mode = "await_restart"
                            st.session_state.quiz_chat.append({
                                "role": "assistant",
                                "text": "Super, das waren alle Fragen zu deinem Thema. Möchtest du ein anderes Thema üben? (ja/nein)"
                            })
                    else:
                        # deutlich daneben – erneuter Versuch mit Tipp
                        hint = _hint_from_answer(gold)
                        st.session_state.quiz_chat.append({
                            "role": "assistant",
                            "text": f"Das ist noch nicht ganz richtig. (Bewertung: {conf*100:.0f}%) Versuche es noch einmal. Tipp: {hint}"
                        })

            elif st.session_state.quiz_mode == "await_restart":
                ans = user_msg.strip().lower()
                if ans in ("ja", "j", "yes", "y"):
                    st.session_state.quiz_mode = "await_topic"
                    st.session_state.quiz_candidates = []
                    st.session_state.quiz_index = 0
                    st.session_state.quiz_topic = ""
                    st.session_state.quiz_chat.append({
                        "role": "assistant",
                        "text": "Gerne! Nenne mir bitte das nächste Thema für dein Quiz."
                    })
                elif ans in ("nein", "n", "no"):
                    st.session_state.quiz_mode = "done"
                    st.session_state.quiz_chat.append({
                        "role": "assistant",
                        "text": "Alles klar. Danke fürs Mitmachen und bis bald! 👋"
                    })
                else:
                    st.session_state.quiz_chat.append({
                        "role": "assistant",
                        "text": "Bitte antworte mit „ja“ oder „nein“. Möchtest du ein anderes Thema üben?"
                    })

        # Eingabefeld leeren und rerun für bessere UX (über Flag vor Widget-Erzeugung)
        st.session_state.quiz_clear_input = True
        st.rerun()

# -----------------------------
# Tab: Über
# -----------------------------
with tab_about:
    st.subheader("Über diese App")
    st.markdown(
        """
        Diese Streamlit‑App kombiniert:
        
        - Embedding‑Suche (Sentence‑Transformers)
        - Cross‑Encoder Re‑Ranking (MSMARCO MiniLM EN‑DE)
        - Extraktive QA (multilingual, XLM‑RoBERTa SQuAD2)
        - Precompute‑Tab zum Vorberechnen & Speichern von Embeddings
        
        Design‑Hinweise:
        - Top‑1 Kontext wird dem QA‑Modell übergeben (Token‑Limit‑sicher)
        - QA nutzt Fensterung/Überlappung bei langen Kontexten
        - Re‑Ranker‑Scores und Embedding‑Wahrscheinlichkeit werden angezeigt
        
        Hinweis: Modell-Downloads können groß sein und etwas dauern.
        """
    )
