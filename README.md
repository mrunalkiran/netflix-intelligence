# 🎬 Netflix Intelligence Platform

> An end-to-end AI analytics platform built on 8,800+ Netflix titles — combining a full RAG pipeline, vector embeddings, and GPT-3.5-turbo with interactive data visualizations.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-412991?logo=openai&logoColor=white)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange)](https://trychroma.com)
[![Plotly](https://img.shields.io/badge/Plotly-6.8-3F4F75?logo=plotly&logoColor=white)](https://plotly.com)

**[🚀 Live Demo](https://netflixintelligenceplatform.streamlit.app/)**  &nbsp;|&nbsp;  **[GitHub](https://github.com/mrunalkiran/netflix-intelligence)**

---

## What This Is

Netflix Intelligence Platform is a full-stack data intelligence app that lets you explore, query, and get AI-powered recommendations from Netflix's entire content catalogue — using natural language.

Under the hood it runs a **Retrieval-Augmented Generation (RAG)** pipeline: every title is embedded with OpenAI's `text-embedding-3-small` model and stored in a ChromaDB vector store. When you ask a question or request a recommendation, the app retrieves the most semantically relevant titles and feeds them as context to GPT-3.5-turbo — grounding the AI's answers in real data rather than hallucinated facts.

---

## Features

| Tab | What It Does |
|-----|-------------|
| 🎭 Genres | Top genres by title count, content type split, ratings distribution |
| 🌍 Countries | Top content-producing countries with an interactive choropleth map |
| 📈 Growth | Year-over-year content growth with peak year detection and custom tooltips |
| 🎬 Directors | Most prolific directors with title count, type, and genre breakdown |
| 🔍 Explorer | Filter 8,800+ titles by type, genre, rating, and country — instant card grid |
| 🍿 Recommender | Input any title → RAG retrieves similar content → GPT explains why you'll like them |
| 🤖 Ask AI | Natural language Q&A grounded in real Netflix data via RAG |

---

## Architecture

```
Netflix Titles CSV (raw)
        │
        ▼
  pipeline/clean.py          ← data cleaning, feature engineering
        │
        ▼
  netflix_clean.csv
        │
   ┌────┴────────────────────────────┐
   │                                 │
   ▼                                 ▼
dashboard/app.py              rag/ingest.py
(Streamlit UI)          (OpenAI text-embedding-3-small)
   │                                 │
   │                                 ▼
   │                          ChromaDB vector store
   │                                 │
   └──────────► rag/query.py ◄───────┘
                (retrieve → GPT-3.5-turbo → answer)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | Streamlit, Plotly, custom CSS |
| **AI / LLM** | OpenAI GPT-3.5-turbo (`chat.completions`) |
| **Embeddings** | OpenAI `text-embedding-3-small` |
| **Vector Store** | ChromaDB (persistent, local) |
| **Data** | Pandas, NumPy |
| **Dataset** | [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) — 8,800+ titles |
| **Language** | Python 3.11 |

---

## RAG Pipeline Detail

1. **Ingest** (`rag/ingest.py`) — generates two types of fact strings per title:
   - *Aggregate facts* — dataset-level statistics (genre distribution, country counts, year trends)
   - *Title facts* — per-title descriptions including type, year, genre, rating, country, and plot summary
2. **Embed** — each fact is embedded via `text-embedding-3-small` in batches of 50
3. **Store** — embeddings + raw text upserted into a ChromaDB persistent collection (`netflix_facts`)
4. **Retrieve** (`rag/query.py`) — query is embedded at runtime, top-k nearest neighbours retrieved
5. **Generate** — retrieved context injected into GPT-3.5-turbo system prompt; response streamed back

---

## Running Locally

```bash
# 1. Clone
git clone https://github.com/mrunalkiran/netflix-intelligence.git
cd netflix-intelligence

# 2. Create and activate venv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. (First run only) Re-build the vector store
python rag/ingest.py

# 6. Launch the app
streamlit run dashboard/app.py
```

> **Note:** `chroma_db/` and `data/netflix_clean.csv` are already committed — step 5 is only needed if you modify the dataset.

---

## Project Structure

```
netflix-intelligence/
├── dashboard/
│   └── app.py              # Streamlit app — 7 tabs, all visualizations
├── pipeline/
│   ├── clean.py            # Data cleaning and feature engineering
│   └── analyse.py          # EDA functions (genres, countries, growth, directors)
├── rag/
│   ├── ingest.py           # Embedding + ChromaDB ingestion pipeline
│   └── query.py            # Vector retrieval + GPT-3.5-turbo answer generation
├── data/
│   └── netflix_clean.csv   # Cleaned dataset (8,800+ titles)
├── chroma_db/              # Persistent ChromaDB vector store
├── images/
│   └── netflix_logo.png
├── .streamlit/
│   └── config.toml         # Dark theme config
├── requirements.txt
└── runtime.txt
```

---

## Dataset

[Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) from Kaggle — 8,807 titles spanning 1925–2021, with fields for type, title, director, cast, country, date added, release year, rating, duration, genre, and description.

---

*Built by [Mrunal Kiran](https://github.com/mrunalkiran)*
