import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

// Runtime embedding via Transformers.js (loaded dynamically from CDN)
async function getTransformers() {
  if (window.__tf_loaded) return window.__tf_loaded
  const mod = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0/dist/transformers.min.js')
  const { env } = mod
  env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/@xenova/transformers@3.0.0/dist/wasm/'
  window.__tf_loaded = mod
  return mod
}

const DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
const DEFAULT_QA_MODEL = 'deepset/gelectra-base-germanquad'

function classNames(...c) { return c.filter(Boolean).join(' ') }

function Label({ children }) {
  return <span className="text-xs text-muted">{children}</span>
}

function Bubble({ role, html }) {
  return (
    <div className={classNames('flex', role === 'user' ? 'justify-end' : 'justify-start')}>
      <div className={classNames(
        'max-w-[70ch] rounded-lg px-3 py-2',
        role === 'user' ? 'bg-slate-800' : 'bg-panel border border-slate-700/60'
      )} dangerouslySetInnerHTML={{ __html: html }} />
    </div>
  )
}

export default function App() {
  const [datasets, setDatasets] = useState([])
  const [datasetId, setDatasetId] = useState('')
  const [status, setStatus] = useState('')
  const [items, setItems] = useState([])
  const [meta, setMeta] = useState(null)
  const [embeddings, setEmbeddings] = useState(null) // Int8Array | Float32Array
  const [components, setComponents] = useState(null) // Float32Array
  const [mean, setMean] = useState(null) // Float32Array
  const [topK, setTopK] = useState(1)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [extractive, setExtractive] = useState(true)

  const [embedPipe, setEmbedPipe] = useState(null)
  const [qaPipe, setQaPipe] = useState(null)

  const listRef = useRef(null)

  useEffect(() => {
    const loadManifest = async () => {
      try {
        const resp = await fetch('/quant/datasets.json', { cache: 'no-cache' })
        if (resp.ok) {
          const j = await resp.json()
          const ds = Array.isArray(j?.datasets) ? j.datasets : []
          setDatasets(ds)
          if (ds.length > 0) {
            setDatasetId(ds[0].id)
          }
        } else {
          // Fallback: try to probe a known default
          setDatasets([{ id: 'qa_Klexikon-Prod-180825_sbert', name: 'Klexikon Prod (SBERT)' }])
          setDatasetId('qa_Klexikon-Prod-180825_sbert')
        }
      } catch {
        setDatasets([{ id: 'qa_Klexikon-Prod-180825_sbert', name: 'Klexikon Prod (SBERT)' }])
        setDatasetId('qa_Klexikon-Prod-180825_sbert')
      }
    }
    loadManifest()
  }, [])

  useEffect(() => {
    if (!datasetId) return
    const loadAssets = async () => {
      try {
        setStatus('Lade Dataset…')
        const base = `/quant/${datasetId}`
        const metaResp = await fetch(`${base}.meta.json`, { cache: 'force-cache' })
        if (!metaResp.ok) throw new Error('Meta nicht gefunden')
        const m = await metaResp.json()
        const [embBuf, compBuf, meanBuf, itemsList] = await Promise.all([
          fetch(`${base}.embeddings.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
          fetch(`${base}.pca_components.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
          fetch(`${base}.pca_mean.bin`, { cache: 'force-cache' }).then(r => r.arrayBuffer()),
          fetch(`${base}.items.json`, { cache: 'force-cache' }).then(async r => (r.ok ? r.json() : [])),
        ])
        const isInt8 = m.quant === 'int8'
        setEmbeddings(isInt8 ? new Int8Array(embBuf) : new Float32Array(embBuf))
        setComponents(new Float32Array(compBuf))
        setMean(new Float32Array(meanBuf))
        setItems(Array.isArray(itemsList) ? itemsList : [])
        setMeta(m)
        setStatus(`Dataset: ${datasetId} · Items: ${m.rows ?? itemsList.length}`)
      } catch (e) {
        setStatus(`Dataset-Ladefehler: ${e?.message || String(e)}`)
      }
    }
    loadAssets()
  }, [datasetId])

  useEffect(() => {
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages])

  async function ensureEmbedModel() {
    if (embedPipe) return embedPipe
    setStatus('Embedding lädt…')
    const { pipeline, env } = await getTransformers()
    const pipe = await pipeline('feature-extraction', DEFAULT_MODEL)
    setEmbedPipe(pipe)
    setStatus(s => s.replace('Embedding lädt…', 'Embedding aktiv'))
    return pipe
  }

  async function ensureQAModel() {
    if (qaPipe) return qaPipe
    try {
      setStatus('QA lädt…')
      const { pipeline } = await getTransformers()
      const p = await pipeline('question-answering', DEFAULT_QA_MODEL)
      setQaPipe(p)
      setStatus(s => s.replace('QA lädt…', 'QA aktiv'))
      return p
    } catch (e) {
      setStatus('QA nicht verfügbar')
      return null
    }
  }

  function l2Normalize(vec) {
    let n = 0
    for (let i = 0; i < vec.length; i++) n += vec[i] * vec[i]
    n = Math.sqrt(n) || 1
    const out = new Float32Array(vec.length)
    for (let i = 0; i < vec.length; i++) out[i] = vec[i] / n
    return out
  }

  function pcaProjectNormalize(vec, meanArr, compArr, d) {
    const D0 = meanArr.length
    if (!vec || vec.length !== D0) return null
    const out = new Float32Array(d)
    for (let j = 0; j < d; j++) {
      let sum = 0
      let offset = j
      for (let i = 0; i < D0; i++, offset += d) {
        sum += (vec[i] - meanArr[i]) * compArr[offset]
      }
      out[j] = sum
    }
    return l2Normalize(out)
  }

  function quantizeInt8Normalized(vecNorm) {
    const q = new Int8Array(vecNorm.length)
    for (let i = 0; i < vecNorm.length; i++) {
      let v = Math.round(127 * vecNorm[i])
      if (v < -127) v = -127; else if (v > 127) v = 127
      q[i] = v
    }
    return q
  }

  function dotInt8(a, b) {
    let s = 0
    for (let i = 0; i < a.length; i++) s += a[i] * b[i]
    return s
  }

  function cosineFromInt8Dot(dot) {
    const scale = 127 * 127
    return dot / scale
  }

  async function embedToPCA(text) {
    const pipe = await ensureEmbedModel()
    const features = await pipe(text, { pooling: 'none', normalize: false })
    // features.dims = [1, seq_len, hidden]
    const lastHidden = features.data
    const [_, seqLen, hidden] = features.dims
    const pooled = new Float32Array(hidden)
    for (let t = 0; t < seqLen; t++) {
      const off = t * hidden
      for (let h = 0; h < hidden; h++) pooled[h] += lastHidden[off + h]
    }
    for (let h = 0; h < hidden; h++) pooled[h] /= (seqLen || 1)
    return pcaProjectNormalize(pooled, mean, components, meta.pca_dim)
  }

  function topKSimilar(queryVecNorm, k) {
    const d = meta.pca_dim
    const rows = meta.rows | 0
    const sims = new Float32Array(rows)
    if (meta.quant === 'int8') {
      const q = quantizeInt8Normalized(queryVecNorm)
      for (let r = 0; r < rows; r++) {
        const start = r * d
        const dot = dotInt8(q, embeddings.subarray(start, start + d))
        sims[r] = cosineFromInt8Dot(dot)
      }
    } else {
      for (let r = 0; r < rows; r++) {
        let dot = 0
        const start = r * d
        for (let j = 0; j < d; j++) dot += queryVecNorm[j] * embeddings[start + j]
        sims[r] = dot
      }
    }
    const idxs = Array.from({ length: rows }, (_, i) => i)
    idxs.sort((a, b) => sims[b] - sims[a])
    return idxs.slice(0, Math.min(k, rows)).map(i => ({ row: i, score: sims[i] }))
  }

  function formatAssistantAnswer(item, score, highlightedHtml) {
    const verified = '<strong>[GEPRÜFTE ANTWORT AUF QA-BASIS]</strong>'
    const url = item?.url ? `<div class=\"mt-1 text-xs text-muted\"><a class=\"text-primary hover:underline\" href=\"${item.url}\" target=\"_blank\">Quelle</a></div>` : ''
    let html = `
      <div class=\"space-y-2\">
        <div class=\"text-xs text-muted\">${verified} (Score: ${(score*100).toFixed(1)}%)</div>
        <div>${highlightedHtml || escapeHTML(item?.answer || '')}</div>
        ${url}
      </div>
    `
    if (score < 0.5) {
      html += `
        <div class=\"mt-2 border-t border-slate-700/50 pt-2\">
          <div class=\"text-xs text-muted\"><strong>[UNSICHERE ANTWORT AUF KI-BASIS]</strong></div>
          <div class=\"text-sm text-muted\">Geringe Übereinstimmung. Im Offline-Modus ohne KI-Ergänzung kann ich nur die geprüfte QA-Antwort anzeigen.</div>
        </div>`
    }
    return html
  }

  function escapeHTML(str) {
    return (str || '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))
  }

  async function extractiveHighlight(answer, queryVecNorm, questionText) {
    if (!extractive) return { highlighted: escapeHTML(answer) }
    // Try QA model
    try {
      const qa = await ensureQAModel()
      if (qa && (questionText || '').trim()) {
        const res = await qa({ question: questionText, context: answer })
        if (res && Number.isInteger(res.start) && Number.isInteger(res.end) && res.end > res.start) {
          const before = escapeHTML(answer.slice(0, res.start))
          const span = escapeHTML(answer.slice(res.start, res.end))
          const after = escapeHTML(answer.slice(res.end))
          return { highlighted: `${before}<mark>${span}</mark>${after}` }
        }
      }
    } catch {}
    // Fallback: sentence similarity using embed model
    const { pipeline } = await getTransformers()
    const pipe = await ensureEmbedModel()
    const sents = (answer || '').replace(/\s+/g, ' ').split(/(?<=[.!?])\s+(?=[A-ZÄÖÜ])/).filter(Boolean)
    const outs = await Promise.all(sents.map(s => pipe(s, { pooling: 'none', normalize: false })))
    const scores = outs.map(feat => {
      const hidden = feat.dims[2]
      const seq = feat.dims[1]
      const pooled = new Float32Array(hidden)
      const data = feat.data
      for (let t = 0; t < seq; t++) {
        const off = t * hidden
        for (let h = 0; h < hidden; h++) pooled[h] += data[off + h]
      }
      for (let h = 0; h < hidden; h++) pooled[h] /= (seq || 1)
      const proj = pcaProjectNormalize(pooled, mean, components, meta.pca_dim)
      let dot = 0
      for (let j = 0; j < proj.length; j++) dot += proj[j] * queryVecNorm[j]
      return dot
    })
    const top = new Set(scores.map((s, i) => [s, i]).sort((a,b)=>b[0]-a[0]).slice(0,3).map(x=>x[1]))
    const highlighted = sents.map((s, i) => top.has(i) ? `<mark>${escapeHTML(s)}</mark>` : escapeHTML(s)).join(' ')
    return { highlighted }
  }

  async function onSend(e) {
    e.preventDefault()
    const text = input.trim()
    if (!text || !meta) return
    setMessages(m => [...m, { role: 'user', html: escapeHTML(text) }])
    setInput('')
    try {
      const qVec = await embedToPCA(text)
      const results = topKSimilar(qVec, Math.max(1, +topK || 1))
      const rowToItem = meta.row_to_item_index || []
      const replies = []
      for (const r of results) {
        const idx = rowToItem[r.row] ?? r.row
        const item = items[idx]
        const ex = await extractiveHighlight(item?.answer || '', qVec, text)
        const html = formatAssistantAnswer(item, r.score, ex.highlighted)
        replies.push({ role: 'assistant', html })
      }
      setMessages(m => [...m, ...replies])
    } catch (err) {
      setMessages(m => [...m, { role: 'assistant', html: `<div class=\"text-danger\">Fehler: ${escapeHTML(err?.message || String(err))}</div>` }])
    }
  }

  return (
    <div className="min-h-screen bg-bg text-text flex flex-col">
      <header className="border-b border-slate-700/50 bg-panel">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center gap-3 justify-between">
          <div className="flex items-center gap-3">
            <div className="font-semibold">Sachrichtiger QA-Chatbot</div>
            <div className="text-xs text-muted">{status}</div>
          </div>
          <nav className="text-sm text-muted flex items-center gap-4">
            <Link className="hover:text-text" to="/impressum">Impressum</Link>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl w-full flex-1 grid grid-rows-[auto,1fr,auto] px-4">
        {/* Settings row */}
        <div className="py-3 grid grid-cols-1 md:grid-cols-3 gap-3 items-center">
          <div className="flex items-center gap-2">
            <Label>Dataset</Label>
            <select value={datasetId} onChange={e => setDatasetId(e.target.value)} className="rounded-md bg-[#0b0f18] border border-slate-700/60 px-2 py-1">
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name || d.id}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-2">
            <Label>Top-K</Label>
            <input type="number" min={1} max={10} value={topK} onChange={e => setTopK(Math.max(1, Math.min(10, +e.target.value || 1)))} className="w-20 rounded-md bg-[#0b0f18] border border-slate-700/60 px-2 py-1" />
          </div>
          <div className="flex items-center gap-2">
            <input id="extr" type="checkbox" checked={extractive} onChange={e => setExtractive(!!e.target.checked)} />
            <label htmlFor="extr" className="text-sm">Extraktives Highlight</label>
          </div>
        </div>

        {/* Messages */}
        <div ref={listRef} className="overflow-auto space-y-3 py-2">
          {messages.map((m, i) => (
            <Bubble key={i} role={m.role} html={m.html} />
          ))}
        </div>

        {/* Composer */}
        <form onSubmit={onSend} className="py-3 grid grid-cols-[1fr_auto] gap-2">
          <input value={input} onChange={e => setInput(e.target.value)} placeholder="Frage hier eingeben…" className="rounded-md bg-[#0b0f18] border border-slate-700/60 px-3 py-2" />
          <button type="submit" className="rounded-md bg-primary px-4 py-2 text-white hover:brightness-110">Senden</button>
        </form>
      </main>

      <footer className="border-t border-slate-700/50 bg-panel">
        <div className="mx-auto max-w-6xl px-4 py-3 text-sm text-muted">
          © {new Date().getFullYear()} Jan Schachtschabel — <a className="text-primary hover:underline" href="mailto:jan@schachtschabel.net">Kontakt</a>
        </div>
      </footer>
    </div>
  )
}
