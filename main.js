// Minimal, API-free QA chatbot using SBERT embeddings in the browser
// - Loads compact quantized assets from /quant/<datasetId>.*
// - Embeds user input with Transformers.js (paraphrase-multilingual-MiniLM-L12-v2)
// - Projects into PCA space from meta/components/mean to match dataset embeddings
// - Top-K retrieval by cosine similarity (supports int8 or float32 dataset)
// - Optional simple extractive highlight: selects top similar sentences from the answer

import { env, pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0/dist/transformers.min.js';

// Configure WASM paths for onnxruntime-web backend used by Transformers.js
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0/dist/wasm/';

const DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2';
const DEFAULT_QA_MODEL = 'deepset/gelectra-base-germanquad';
const STORAGE_KEY = 'srqb_settings';

const el = {
  statusProvider: document.getElementById('status-provider'),
  statusDataset: document.getElementById('status-dataset'),
  statusCount: document.getElementById('status-count'),
  btnSettings: document.getElementById('btn-settings'),
  settings: document.getElementById('settings'),
  datasetId: document.getElementById('dataset-id'),
  btnLoadDataset: document.getElementById('btn-load-dataset'),
  topK: document.getElementById('top-k'),
  extractiveToggle: document.getElementById('extractive-toggle'),
  messages: document.getElementById('messages'),
  composer: document.getElementById('composer'),
  input: document.getElementById('user-input'),
};

let state = {
  datasetId: '',
  topK: 1,
  extractive: true,
  // Quant assets
  items: [], // [{question, answer, url, ...}]
  meta: null, // { providerId, model, pca_dim, quant, rows, ... }
  embeddings: null, // Int8Array or Float32Array flattened row-major
  components: null, // Float32Array (source_dim x pca_dim)
  mean: null, // Float32Array (source_dim)
  rows: 0,
  pcaDim: 0,
  sourceDim: 0,
  quant: 'int8',
  // Model pipeline
  embedPipe: null,
  qaPipe: null,
};

function saveSettings() {
  const payload = {
    datasetId: state.datasetId,
    topK: state.topK,
    extractive: state.extractive,
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch {}
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (s.datasetId) state.datasetId = s.datasetId;
    if (Number.isFinite(+s.topK) && +s.topK >= 1) state.topK = +s.topK;
    if (typeof s.extractive === 'boolean') state.extractive = s.extractive;
  } catch {}
}

function setStatus() {
  el.statusDataset.textContent = `Dataset: ${state.datasetId || '–'}`;
  const count = state.items?.length || state.meta?.rows || 0;
  el.statusCount.textContent = `Items: ${count}`;
}

function quantUrlBase(id) {
  // Self-contained: load from local subfolder within this app
  return `./quant/${id}`;
}

async function loadQuantAssets(id) {
  const base = quantUrlBase(id);
  const metaResp = await fetch(`${base}.meta.json`, { cache: 'force-cache' });
  if (!metaResp.ok) throw new Error(`Meta nicht gefunden für ${id} unter ${base}.meta.json`);
  const meta = await metaResp.json();

  const [embBuf, compBuf, meanBuf, items] = await Promise.all([
    fetch(`${base}.embeddings.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
    fetch(`${base}.pca_components.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
    fetch(`${base}.pca_mean.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
    fetch(`${base}.items.json`, { cache: 'force-cache' }).then(async r => (r.ok ? r.json() : [])),
  ]);

  const isInt8 = meta.quant === 'int8';
  const embeddings = isInt8 ? new Int8Array(embBuf) : new Float32Array(embBuf);
  const components = new Float32Array(compBuf);
  const mean = new Float32Array(meanBuf);

  state.meta = meta;
  state.embeddings = embeddings;
  state.components = components;
  state.mean = mean;
  state.rows = meta.rows | 0;
  state.pcaDim = meta.pca_dim | 0;
  state.sourceDim = meta.source_dim | 0;
  state.quant = meta.quant;
  state.items = Array.isArray(items) ? items : [];

  // Provider must be SBERT to be compatible with local model
  if ((meta.providerId || '').toLowerCase() !== 'sbert') {
    throw new Error(`Inkompatible Assets: providerId=${meta.providerId}. Bitte SBERT-Assets mit scripts/precompute_sbert_embeddings.py erzeugen.`);
  }
}

async function ensureEmbedModel() {
  if (state.embedPipe) return state.embedPipe;
  el.statusProvider.textContent = 'Embedding: SBERT (lädt Modell)…';
  const pipe = await pipeline('feature-extraction', DEFAULT_MODEL);
  state.embedPipe = pipe;
  el.statusProvider.textContent = 'Embedding: SBERT (Browser) · QA: optional';
  return pipe;
}

async function ensureQAPipeline() {
  if (state.qaPipe) return state.qaPipe;
  try {
    el.statusProvider.textContent = 'Embedding: SBERT (Browser) · QA: lädt…';
    const qa = await pipeline('question-answering', DEFAULT_QA_MODEL);
    state.qaPipe = qa;
    el.statusProvider.textContent = 'Embedding: SBERT (Browser) · QA: aktiv';
    return qa;
  } catch (e) {
    // Model may be unavailable in CDN or too large; keep optional
    el.statusProvider.textContent = 'Embedding: SBERT (Browser) · QA: nicht verfügbar';
    return null;
  }
}

// Mean pooling helper over token dimension.
// features is a Tensor from Transformers.js with dims [1, seq_len, hidden]
function meanPool(features) {
  const lastHidden = features.data; // Float32Array
  const [batch, seqLen, hidden] = features.dims;
  const out = new Float32Array(hidden);
  let count = 0;
  for (let t = 0; t < seqLen; t++) {
    const offset = t * hidden;
    for (let h = 0; h < hidden; h++) {
      out[h] += lastHidden[offset + h];
    }
    count++;
  }
  const denom = count || 1;
  for (let h = 0; h < hidden; h++) out[h] /= denom;
  return out;
}

function l2Normalize(vec) {
  let n = 0;
  for (let i = 0; i < vec.length; i++) n += vec[i] * vec[i];
  n = Math.sqrt(n) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = vec[i] / n;
  return out;
}

function pcaProjectNormalize(vec, mean, components, pcaDim) {
  const D0 = mean.length;
  if (!vec || vec.length !== D0) return null;
  const out = new Float32Array(pcaDim);
  for (let j = 0; j < pcaDim; j++) {
    let sum = 0;
    let offset = j;
    for (let i = 0; i < D0; i++, offset += pcaDim) {
      sum += (vec[i] - mean[i]) * components[offset];
    }
    out[j] = sum;
  }
  return l2Normalize(out);
}

function quantizeInt8Normalized(vecNorm) {
  const q = new Int8Array(vecNorm.length);
  for (let i = 0; i < vecNorm.length; i++) {
    let v = Math.round(127 * vecNorm[i]);
    if (v < -127) v = -127; else if (v > 127) v = 127;
    q[i] = v;
  }
  return q;
}

function dotInt8(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function cosineFromInt8Dot(dot) {
  const scale = 127 * 127;
  return dot / scale;
}

async function embedTextToPCA(text) {
  const pipe = await ensureEmbedModel();
  const features = await pipe(text, { pooling: 'none', normalize: false });
  const pooled = meanPool(features);
  // pooled is source_dim (should equal meta.source_dim)
  const projected = pcaProjectNormalize(pooled, state.mean, state.components, state.pcaDim);
  return projected;
}

function topKSimilar(queryVecNorm, k) {
  const d = state.pcaDim;
  const rows = state.rows;
  const sims = new Float32Array(rows);
  if (state.quant === 'int8') {
    const q = quantizeInt8Normalized(queryVecNorm);
    for (let r = 0; r < rows; r++) {
      const start = r * d;
      const dot = dotInt8(q, state.embeddings.subarray(start, start + d));
      sims[r] = cosineFromInt8Dot(dot);
    }
  } else {
    for (let r = 0; r < rows; r++) {
      let dot = 0, nq = 1, nx = 1; // both normalized
      const start = r * d;
      for (let j = 0; j < d; j++) {
        dot += queryVecNorm[j] * state.embeddings[start + j];
      }
      sims[r] = dot; // both normalized => cosine = dot
    }
  }
  // Get indices sorted desc
  const idxs = Array.from({ length: rows }, (_, i) => i);
  idxs.sort((a, b) => sims[b] - sims[a]);
  const out = [];
  for (let i = 0; i < Math.min(k, rows); i++) {
    out.push({ row: idxs[i], score: sims[idxs[i]] });
  }
  return out;
}

function splitSentencesGerman(text) {
  return (text || '')
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+(?=[A-ZÄÖÜ])/)
    .map(s => s.trim())
    .filter(Boolean);
}

async function extractiveHighlight(answer, queryVecNorm, questionText) {
  if (!state.extractive) return { highlighted: answer, topSentences: [] };
  const sents = splitSentencesGerman(answer);
  if (sents.length === 0) return { highlighted: answer, topSentences: [] };

  // Try dedicated QA model first (optional)
  try {
    const qa = await ensureQAPipeline();
    if (qa) {
      const question = (questionText || '').trim();
      if (question) {
        const res = await qa({ question, context: answer });
        if (res && Number.isInteger(res.start) && Number.isInteger(res.end) && res.end > res.start) {
          const before = escapeHTML(answer.slice(0, res.start));
          const span = escapeHTML(answer.slice(res.start, res.end));
          const after = escapeHTML(answer.slice(res.end));
          return { highlighted: `${before}<mark>${span}</mark>${after}`, topSentences: [] };
        }
      }
    }
  } catch {}

  // Embed sentences in source space and project to PCA
  const pipe = await ensureEmbedModel();
  const outputs = await Promise.all(sents.map(s => pipe(s, { pooling: 'none', normalize: false })));
  const scores = [];
  for (let i = 0; i < outputs.length; i++) {
    const features = outputs[i];
    const pooled = meanPool(features);
    const proj = pcaProjectNormalize(pooled, state.mean, state.components, state.pcaDim);
    // cosine since both normalized
    let dot = 0;
    for (let j = 0; j < proj.length; j++) dot += proj[j] * queryVecNorm[j];
    scores.push({ i, s: dot });
  }
  scores.sort((a, b) => b.s - a.s);
  const top = scores.slice(0, Math.min(3, scores.length)).map(t => t.i);

  // Mark top sentences
  const highlighted = sents.map((s, i) => top.includes(i) ? `<mark>${escapeHTML(s)}</mark>` : escapeHTML(s)).join(' ');
  return { highlighted, topSentences: top.map(i => sents[i]) };
}

function escapeHTML(str) {
  return str.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function renderMessage(role, html) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = html;
  el.messages.appendChild(div);
  el.messages.scrollTop = el.messages.scrollHeight;
}

function renderAnswer(item, score, extracted) {
  const urlPart = item.url ? `<div class="meta"><a href="${item.url}" target="_blank" rel="noopener">Quelle</a></div>` : '';
  const verifiedLabel = '<strong>[GEPRÜFTE ANTWORT AUF QA-BASIS]</strong>';
  let ansHtml = `
    <div class="answer">
      <div class="label">${verifiedLabel} (Score: ${(score*100).toFixed(1)}%)</div>
      <div class="text">${extracted?.highlighted || escapeHTML(item.answer)}</div>
      ${urlPart}
    </div>
  `;
  // Add uncertain paragraph if match score is low and we cannot augment with KI
  const THRESHOLD = 0.5;
  if (typeof score === 'number' && score < THRESHOLD) {
    ansHtml += `
      <div class="answer secondary">
        <div class="label"><strong>[UNSICHERE ANTWORT AUF KI-BASIS]</strong></div>
        <div class="text">Die Übereinstimmung ist gering. Im Offline-Modus ohne KI-Ergänzung kann ich nur die geprüfte QA-Antwort anzeigen.</div>
      </div>
    `;
  }
  renderMessage('assistant', ansHtml);
}

async function onSend(e) {
  e.preventDefault();
  const text = el.input.value.trim();
  if (!text) return;
  renderMessage('user', `<div class="bubble">${escapeHTML(text)}</div>`);
  el.input.value = '';

  try {
    if (!state.meta) throw new Error('Kein Dataset geladen.');
    const qVec = await embedTextToPCA(text);
    if (!qVec) throw new Error('Embedding fehlgeschlagen.');

    const results = topKSimilar(qVec, state.topK);
    if (results.length === 0) {
      renderMessage('assistant', '<div class="error">Keine Treffer gefunden.</div>');
      return;
    }

    // Map row to item index via meta.row_to_item_index
    const rowToItem = state.meta.row_to_item_index || [];
    for (const r of results) {
      const itemIdx = rowToItem[r.row] ?? r.row;
      const item = state.items[itemIdx];
      const extracted = await extractiveHighlight(item?.answer || '', qVec, text);
      renderAnswer(item, r.score, extracted);
    }
  } catch (err) {
    renderMessage('assistant', `<div class="error">Fehler: ${escapeHTML(err.message || String(err))}</div>`);
  }
}

async function loadDatasetAndModel() {
  const id = el.datasetId.value.trim();
  if (!id) return;
  renderMessage('system', `<div class="info">Lade Dataset "${escapeHTML(id)}"…</div>`);
  try {
    await loadQuantAssets(id);
    setStatus();
    state.datasetId = id;
    saveSettings();
    renderMessage('system', `<div class="ok">Dataset geladen: ${escapeHTML(id)} (${state.items.length || state.rows} Einträge)</div>`);
    await ensureEmbedModel();
  } catch (e) {
    renderMessage('system', `<div class="error">${escapeHTML(e.message || String(e))}</div>`);
  }
}

function initUI() {
  el.btnSettings.addEventListener('click', () => {
    el.settings.classList.toggle('hidden');
  });
  el.btnLoadDataset.addEventListener('click', loadDatasetAndModel);
  el.topK.addEventListener('change', () => {
    const v = Math.max(1, Math.min(10, +el.topK.value || 1));
    state.topK = v;
    el.topK.value = String(v);
    saveSettings();
  });
  el.extractiveToggle.addEventListener('change', () => {
    state.extractive = !!el.extractiveToggle.checked;
    saveSettings();
  });
  el.composer.addEventListener('submit', onSend);
}

function applyInitialSettings() {
  loadSettings();
  if (state.datasetId) el.datasetId.value = state.datasetId;
  else el.datasetId.placeholder = 'z.B. qa_Klexikon-Prod-180825_sbert';
  el.topK.value = String(state.topK);
  el.extractiveToggle.checked = !!state.extractive;
  setStatus();
}

// Bootstrap
initUI();
applyInitialSettings();
