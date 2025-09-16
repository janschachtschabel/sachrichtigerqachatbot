# QA Bot (Streamlit)

Eine schlanke Python/Streamlit‑Variante des QA‑Bots mit:

- Embedding‑Suche (Sentence‑Transformers) und Precompute‑Workflow
- Optionalem Cross‑Encoder Re‑Ranking (deaktivierbar, „nur Embedding“)
- Extraktiver QA (mehrere Modelle; per‑Passage‑Prüfung, Aggregation, Confidence‑Schwelle)
- Generativen Antworten (kleine LLMs) mit streng kontextgebundenem Prompt und Fallback
- Optionaler QA‑Nachbearbeitung (Grammatik/Satzbau) über kleines LLM (ohne neue Fakten)
- Umfassendem Einstellungs‑Panel (Embedding Top‑K, Cross‑Encoder Prüf‑K, Anzeige Top‑K, QA Top‑N, QA‑Confidence, Antwort‑Länge)

## Schnellstart

1) Virtuelle Umgebung erstellen und Abhängigkeiten installieren

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) App starten

```bash
streamlit run app.py
```

Die App öffnet sich im Browser (Standard: http://localhost:8501).

## Deployment

### Streamlit Community Cloud

- App-URL: Repo verbinden, Branch wählen.
- App file: `qabotstreamlit/app.py`
- Requirements file: `qabotstreamlit/requirements.txt`
- Python: 3.10 oder höher

### Hugging Face Spaces (SDK: Streamlit)

- Neues Space anlegen → SDK „Streamlit“.
- `app_file`: `qabotstreamlit/app.py`
- `requirements.txt` automatisch erkannt, sonst Pfad `qabotstreamlit/requirements.txt` setzen.
- Optional: `TRANSFORMERS_CACHE` als persistenten Speicher konfigurieren (reduziert Kaltstart, vermeidet erneute Modelldownloads).

### Docker (empfohlen für Server)

Beispiel `Dockerfile` (Repo-Root als Build-Kontext):

```dockerfile
FROM python:3.10-slim

# System-Pakete für Tokenizer/SSL (optional, aber oft hilfreich)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app/qabotstreamlit
COPY qabotstreamlit/ /app/qabotstreamlit/

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    HF_HOME=/app/cache/hf \
    TRANSFORMERS_CACHE=/app/cache/hf

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
```

Build & Run:

```bash
docker build -t qa-streamlit .
docker run --rm -p 8501:8501 -v $(pwd)/modelcache:/app/cache/hf qa-streamlit
```

### Bare‑Metal / Systemd

1) Virtuelle Umgebung wie oben (Schnellstart) anlegen und `pip install -r qabotstreamlit/requirements.txt` ausführen.
2) Optional `~/.streamlit/config.toml` erstellen:

```toml
[server]
headless = true
enableCORS = false
address = "0.0.0.0"
port = 8501
```

3) Start: `streamlit run qabotstreamlit/app.py`

4) Systemd-Service (optional, Beispiel):

```ini
[Unit]
Description=QA Streamlit App
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/qa-chatbot
Environment="TRANSFORMERS_CACHE=/opt/qa-chatbot/cache/hf"
ExecStart=/opt/qa-chatbot/.venv/bin/streamlit run qabotstreamlit/app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Umgebung & Hinweise für Produktion

- Modelle werden beim ersten Lauf heruntergeladen (Hugging Face). Für stabile Deployments einen persistenten Cache (`TRANSFORMERS_CACHE`) nutzen.
- CPU/GPU: PyTorch wählt automatisch; für GPU auf passende CUDA‑Wheels achten.
- Ports: `PORT` wird von einigen PaaS (Render/Railway/Heroku) vorgegeben. Mit `--server.port $PORT --server.address 0.0.0.0` starten.
- Speicher: Kleine LLMs (0.5–1.7B) laufen meist noch auf CPU, dennoch an RAM/Storage denken.
- Firewalls/Proxies: Evtl. HF‑Mirror/Offline‑Cache einrichten.

## Datenformat

Die App erwartet ein Dataset mit folgenden Feldern:

- `question`: Frage (String)
- `answer`: Antwort‑Text (String)
- `wwwurl` (optional): Quelle/Link (String)

Unterstützte Upload‑Formate im Precompute‑Tab:

- JSON Lines (`.jsonl`) – eine JSON‑Zeile pro Objekt
- JSON (`.json`) – Array von Objekten
- CSV (`.csv`) – Spaltennamen wie oben

## Precompute & Ablage

- Nach dem Precompute werden die Dateien unter `qabotstreamlit/embeddings/` gespeichert:
  - `<name>.npz` – Vektor‑Matrix (float32)
  - `<name>.json` – Metadaten (Modell, Dim, Zeit)
  - `<name>_items.json` – Originaleinträge (Frage/Antwort/URL)

- Im Tab "Suche" wählst du anschließend denselben `<name>` als Embeddings‑Datensatz aus.

## Modelle (Standard)

- Embedding (Default):
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Alternativen: `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/multilingual-e5-small`
- Cross‑Encoder (Default):
  - `cross-encoder/ms-marco-MiniLM-L-12-v2` (Option: „Deaktiviert (nur Embedding)“)
- QA (Extraktiv, Default):
  - `LLukas22/all-MiniLM-L12-v2-qa-all`
- Generativ (klein, Auswahl u. a.):
  - `PleIAs/Pleias-RAG-350M`, `Shahm/t5-small-german`, `google/gemma-3-1b-it`, `microsoft/Phi-3-mini-4k-instruct`, `Qwen/Qwen2-0.5B-Instruct`
- QA‑Nachbearbeitung (optional):
  - `Shahm/t5-small-german`, `google/mt5-small`, `aiassociates/t5-small-grammar-correction-german` u. a.

Alle Modelle werden beim ersten Start automatisch geladen (Internetverbindung notwendig).

## Modi: QA vs. Generativ

- QA (Extraktiv):
  - Top‑K Embedding‑Treffer → optional Cross‑Encoder → Top‑N Passagen werden einzeln vom QA‑Modell geprüft.
  - Nur Antworten mit `confidence ≥ QA‑Schwelle` werden übernommen. Sätze werden dedupliziert und zu einer Antwort aggregiert, ohne mitten im Satz zu schneiden.
  - Optional: Nachbearbeitung (Grammatik/Satzbau) via kleinem LLM, ohne neue Fakten.

- Generativ:
  - Strenger, kurzer Prompt („Nur aus <KONTEXT>; sonst ‚Nicht im Kontext gefunden.‘; 1–2 vollständige deutsche Sätze“).
  - Decoder‑only: niedrige Temperatur (0.15), top_p=0.9, repetition_penalty≈1.15, no_repeat_ngram_size=3.
  - T5/mT0: Beam‑Search (num_beams=4), length_penalty≈1.1, keine Sampling‑Streuung.

## Einstellungen (Tab „Einstellungen“)

- Embedding‑Modell
- Cross‑Encoder (Re‑Ranker): Modellwahl oder „Deaktiviert (nur Embedding)“
- Antwort‑Modus: „QA“ oder „Generativ“
- QA‑Modell (extraktiv)
- QA Nachbearbeitung (Grammatik/Satzbau): „Deaktiviert“ (Default) oder kleines LLM
- Schwelle (Embedding & Re‑Ranker, ≥)
- QA Confidence‑Schwelle (≥)
- Embedding Top‑K (Kandidaten)
- Cross‑Encoder Prüf‑K (wie viele Kandidaten tatsächlich bewertet werden)
- Anzeige Top‑K (wie viele Ergebnisse gelistet werden)
- QA Top‑N (einzeln prüfen)
- Antwort‑Länge (Ziel, Zeichen) – Slider 100–2000

Im Status‑Panel siehst du die aktive Konfiguration inkl. optionalem Post‑Edit‑Modell.

## Hinweise

- QA verarbeitet Passagen fensterweise (Token‑Limit‑sicher) und aggregiert ganze Sätze.
- Re‑Ranker kann abgeschaltet werden (nur Embedding); Rang und Anzeige passen sich an.
- CPU‑Betrieb ist möglich; GPU (CUDA) wird automatisch genutzt, falls verfügbar.
- Bei ersten Modellstarts werden Gewichte heruntergeladen (Firewall/Proxy beachten).
- Für stabile Ergebnisse können generative Modelle auf kleine, deterministische Einstellungen gesetzt werden.

## Ordnerstruktur

```
qabotstreamlit/
├─ app.py
├─ requirements.txt
├─ README.md
├─ data/               # optional: Rohdaten
├─ embeddings/         # abgelegte Embeddings (.npz, .json, _items.json)
└─ utils/
   ├─ __init__.py
   ├─ embeddings.py    # Backend für Sentence‑Transformers
   ├─ search.py        # Vektor‑Suche
   ├─ reranker.py      # Cross‑Encoder Re‑Ranking
   ├─ qa.py            # Extraktive QA (Fensterung, Aggregation)
   ├─ generative.py    # Generative Antworten + Prompting/Decoding
   ├─ post_edit.py     # QA‑Nachbearbeitung (Grammatik/Satzbau)
   └─ eval.py          # Generative Quiz‑Bewertung (0–100%)
```

## Troubleshooting

- „Keine Treffer über der Embedding‑Schwelle“: Schwelle reduzieren oder Embedding‑Modell prüfen.
- „Keine Treffer nach Re‑Ranker‑Schwelle“: Re‑Ranker deaktivieren oder Schwelle senken.
- Zu knappe/große Antworten: „Antwort‑Länge (Ziel, Zeichen)“ anpassen; Aggregation schneidet keine Sätze.
- Langsame Downloads/Ladevorgänge: Internet/Timeouts prüfen; Modelle werden gecacht.
