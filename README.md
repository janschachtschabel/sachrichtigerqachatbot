# Sachrichtiger QA-Chatbot (API-frei)

Ein schlanker, sachrichtiger QA-Chatbot für den Browser. Keine Server-LLMs, keine API-Keys. Die App nutzt:

- React + Vite + Tailwind CSS
- In-Browser Embeddings via Transformers.js (CDN)
  - SBERT: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Ähnlichkeitssuche gegen vorab berechnete, komprimierte QA-Embeddings (PCA 256D, optional int8)
- Optionales extraktives Highlighting (deutsches QA-Modell per Transformers.js) – lazy geladen

Die App lädt Datensätze aus `public/quant/` und zeigt sie im Dropdown an. Das erste Dataset wird automatisch geladen.

---

## Projektstruktur

```
./sachrichtigerqachatbot/
  ├─ index.html               # Vite-Entry (mountet #root)
  ├─ vercel.json             # SPA-Rewrite für Deep Links (z.B. /impressum)
  ├─ package.json
  ├─ vite.config.js
  ├─ tailwind.config.js
  ├─ postcss.config.js
  ├─ public/
  │   └─ quant/
  │       ├─ datasets.json                      # Manifest mit verfügbaren Datensätzen
  │       ├─ <dataset_id>.meta.json             # Meta (providerId=sbert, model, rows, pca_dim, ...)
  │       ├─ <dataset_id>.items.json            # QA-Paare (question/answer/...)
  │       ├─ <dataset_id>.embeddings.bin        # Embeddings (int8 oder float32, zeilenweise)
  │       ├─ <dataset_id>.pca_components.bin    # Float32 (source_dim x pca_dim), row-major
  │       └─ <dataset_id>.pca_mean.bin          # Float32 (source_dim)
  └─ src/
      ├─ index.css            # Tailwind-Einstieg
      ├─ main.jsx             # Router ("/" Chat, "/impressum" Impressum)
      └─ pages/
          ├─ App.jsx         # Chat UI (Dropdown Datensatz, Top-K, Toggle Extraktion)
          └─ Impressum.jsx   # Impressum
```

Begleitende Tools (Repo‑Wurzel):

```
./scripts/
  └─ precompute_sbert_embeddings.py   # SBERT-Precompute (PCA + Quantisierung), schreibt Assets nach <app>/public/quant/
```

> Hinweis: Wenn Du dieses Verzeichnis als eigenes Repo an Vercel übergibst, ist alles vollständig „self‑contained“. 

---

## Datensätze bereitstellen

Alle Dateien eines Datensatzes gehören in `public/quant/`. Beispiel (SBERT-Version des Klexikon-Datensatzes):

```
public/quant/
  qa_Klexikon-Prod-180825_sbert.meta.json
  qa_Klexikon-Prod-180825_sbert.items.json
  qa_Klexikon-Prod-180825_sbert.embeddings.bin
  qa_Klexikon-Prod-180825_sbert.pca_components.bin
  qa_Klexikon-Prod-180825_sbert.pca_mean.bin
```

Das Manifest `public/quant/datasets.json` listet auswählbare Datensätze:

```json
{
  "datasets": [
    {
      "id": "qa_Klexikon-Prod-180825_sbert",
      "name": "Klexikon Prod (SBERT)",
      "description": "QA-Paare Klexikon, SBERT-Embeddings (256D PCA, int8)"
    }
  ]
}
```

- Die App lädt beim Start das erste Manifest-Element als Default.
- Zusätzlich kann der User im Dropdown einen anderen Datensatz wählen.

---

## Embeddings vorberechnen (ohne API)

Im Ordner `./scripts/` liegt das Script `precompute_sbert_embeddings.py` (Python), um aus einer QA-JSON-Datei kompakte SBERT-Assets zu generieren. Das Script verwendet dasselbe Embedding‑Modell wie die App zur Laufzeit (SBERT: `paraphrase-multilingual-MiniLM-L12-v2`), damit der Vektorraum 1:1 kompatibel ist.

### 1) Abhängigkeiten installieren

```powershell
pip install --upgrade sentence-transformers numpy tqdm
```

(Optional: Falls SciPy/SciKit Warnungen auftreten und Dich stören, ggf. `numpy<2.3.0`, `scipy<1.13` verwenden.)

### 2) Ausführen

Beispiel (Windows PowerShell), wenn Script und JSON im gleichen Ordner liegen:

```powershell
python .\precompute_sbert_embeddings.py \
  -i .\qa_Klexikon-Prod-180825.json \
  --out-dir ..\public\quant \
  --dataset-id qa_Klexikon-Prod-180825_sbert \
  --pca-dim 256 --quantize \
  --question-key question \
  --answer-key answer
```

- Flags & Verhalten:
  - `-i`: Pfad zur QA‑JSON (Array von Objekten mit Frage/Antwort‑Feldern)
  - `--question-key` / `--answer-key`: Feldnamen, falls nicht exakt `question`/`answer`
  - `--pca-dim 256`: PCA‑Zieldimension (Standard 256, passt zur App)
  - `--quantize`: Int8‑Quantisierung (empfohlen, schnell & klein)
  - `--out-dir`: Zielordner für die fünf Exportdateien (unter `public/quant/` der App)
  - `--dataset-id`: Basename der Ausgabedateien (muss mit Manifest/Dropdown übereinstimmen)

- Das Script lädt automatisch das Modell `paraphrase-multilingual-MiniLM-L12-v2` (384D), reduziert via PCA und normalisiert die Vektoren. Es schreibt fünf Dateien (siehe Struktur oben). In der `.meta.json` steht u. a. `providerId: "sbert"` – dieser Provider muss zur Laufzeit identisch sein (die App prüft das).

Das Script schreibt nach `--out-dir` fünf Dateien (s. Struktur oben) und meldet u.a. die Anzahl erkannter QA-Paare:

```
[SBERT] Loaded items: 34465; non-empty 'question': 34465
```

Tipp: Für große Datensätze ist ein größerer Batch sinnvoll, z. B. `-c 128`.

Hinweis: Die Fortschrittsanzeige zeigt Batches, nicht die Item‑Gesamtzahl. Beispiel: 34.465 Items bei Batchgröße 64 → ca. 539 Batches (Anzeige `20/539`).

Optional: Statt einer großen Ursprungs‑JSON kann auch das kompakte Items‑JSON verwendet werden (z. B. `public/quant/qa_...items.json`) – dann entfällt ggf. Feld‑Mapping.

---

## Lokale Entwicklung

```bash
# Im Ordner sachrichtigerqachatbot/
npm install
npm run dev
```

- Öffne die Dev-URL (z. B. `http://localhost:5173/`).
- Die App lädt `public/quant/datasets.json` und danach das erste Dataset automatisch.

---

## Produktion / Vercel

1. Dieses Verzeichnis als Repo zu Vercel verbinden (Projekt-Root = `sachrichtigerqachatbot/`).
2. Build-Einstellungen (werden i. d. R. automatisch erkannt):
   - Build Command: `npm run build`
   - Output Directory: `dist`
3. SPA-Rewrite: `vercel.json` ist enthalten, damit Deep Links wie `/impressum` funktionieren.
4. Umgebungsvariablen: **nicht nötig**. Alles läuft im Browser.

---

## UI- und Funktionsüberblick

- **Datensatz-Auswahl**: Dropdown (erstes Manifest-Element wird automatisch geladen)
- **Top-K**: Anzahl der zurückgegebenen QA-Treffer
- **Extraktives Highlight**: Optionales Markieren relevanter Antwortpassagen
  - Versucht zunächst ein deutsches QA-Modell zu laden (lazy). Wenn nicht verfügbar, fallback auf satzbasierte SBERT-Ähnlichkeit.
- **Labels**: Antworten werden mit
  - **[GEPRÜFTE ANTWORT AUF QA-BASIS]**
  - und bei geringer Übereinstimmung zusätzlich **[UNSICHERE ANTWORT AUF KI-BASIS]** ausgezeichnet.

---

## Impressum

- Seite: `/impressum`

---

## Lizenz

Apache License 2.0. Siehe [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) oder Kurzfassung unten.

```
Copyright 2025 Jan Schachtschabel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
