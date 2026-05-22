# ResearchAgent

A multi-agent AI system that takes a complex research question, automatically breaks it into subtasks, retrieves relevant information from documents using RAG, synthesizes the findings, and returns a structured report with confidence scoring and human-review flagging.

## Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Router Agent│  Decomposes question into 2-3 focused subtasks
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Retriever Agent  │  RAG: queries ChromaDB for each subtask
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Summarizer Agent │  Synthesizes retrieved chunks per subtask
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Reporter Agent   │  Generates structured report + confidence score
└──────┬───────────┘
       │
       ▼
Confidence < threshold?
  YES → flagged_for_review
  NO  → auto_approved
```

**Tech Stack:** LangGraph · Anthropic API · ChromaDB · FastAPI · Docker

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker + Docker Compose (for containerized deployment)
- Anthropic API key → [console.anthropic.com](https://console.anthropic.com)

### 2. Clone & Configure

```bash
git clone <repo-url>
cd research-agent
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

### 3. Add Research Papers

Download 5-10 AI research PDFs from arxiv and place them in the `documents/` folder.
See the **Recommended Papers** section below for a curated list.

### 4a. Run with Docker (Recommended)

```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### 4b. Run Locally (without Docker)

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> PDFs in `./documents/` are **auto-ingested on startup** if present.

---

## API Endpoints

### `POST /research`
Run the full multi-agent research pipeline.

```json
{
  "question": "What are the key techniques used in retrieval-augmented generation and how do they improve LLM factual accuracy?",
  "top_k": 5
}
```

**Response includes:**
- `subtasks` — decomposed queries with retrieved chunks and summaries
- `sections` — structured report sections
- `executive_summary` — high-level synthesis
- `confidence_score` — 0.0–1.0 blended score (LLM self-assessment + retrieval quality)
- `review_status` — `auto_approved` or `flagged_for_review`
- `review_reason` — human-readable explanation when flagged

---

### `POST /ingest/upload`
Upload PDF files directly via multipart form.

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "files=@attention_is_all_you_need.pdf" \
  -F "files=@rag_paper.pdf"
```

---

### `POST /ingest/directory`
Ingest all PDFs from a server-side directory.

```bash
curl -X POST "http://localhost:8000/ingest/directory?directory=./documents"
```

---

### `GET /health`
System health check with document count.

```json
{"status": "ok", "chroma_documents": 1247, "model": "claude-3-5-sonnet-20241022"}
```

---

## Confidence Scoring

The confidence score blends three signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| LLM self-assessment | 50% | LLM rates the evidence quality in the report |
| Average retrieval similarity | 30% | Cosine similarity of retrieved chunks to queries |
| Coverage ratio | 20% | Fraction of subtasks with retrieved evidence |

**Threshold:** `CONFIDENCE_THRESHOLD=0.65` (configurable via `.env`)

When confidence < threshold → `review_status: flagged_for_review` with a `review_reason` explaining why.

---

## Recommended Papers (arxiv)

Download these PDFs from arxiv.org and place in `documents/`:

| Paper | arxiv ID | Topic |
|-------|----------|-------|
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | [2005.11401](https://arxiv.org/abs/2005.11401) | RAG |
| REALM: Retrieval-Augmented Language Model Pre-Training | [2002.08909](https://arxiv.org/abs/2002.08909) | RAG |
| Attention Is All You Need | [1706.03762](https://arxiv.org/abs/1706.03762) | Transformers |
| Language Models are Few-Shot Learners (GPT-3) | [2005.14165](https://arxiv.org/abs/2005.14165) | LLMs |
| Chain-of-Thought Prompting Elicits Reasoning in LLMs | [2201.11903](https://arxiv.org/abs/2201.11903) | Reasoning |
| ReAct: Synergizing Reasoning and Acting in Language Models | [2210.03629](https://arxiv.org/abs/2210.03629) | Agents |
| Toolformer: Language Models Can Teach Themselves to Use Tools | [2302.04761](https://arxiv.org/abs/2302.04761) | Agents |
| Self-RAG: Learning to Retrieve, Generate, and Critique | [2310.11511](https://arxiv.org/abs/2310.11511) | RAG |
| HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels | [2212.10496](https://arxiv.org/abs/2212.10496) | Retrieval |
| LLM-based Agent Survey | [2308.11432](https://arxiv.org/abs/2308.11432) | Agents |

---

## Configuration

All settings are configurable via `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022
CONFIDENCE_THRESHOLD=0.65
MAX_RETRIEVAL_DOCS=5
CHUNK_SIZE=800
CHUNK_OVERLAP=100
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=research_papers
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Project Structure

```
research-agent/
├── app/
│   ├── agents/
│   │   ├── graph.py
│   │   ├── router.py
│   │   ├── retriever.py
│   │   ├── summarizer.py
│   │   ├── reporter.py
│   │   └── state.py
│   ├── config.py
│   ├── ingestion.py
│   ├── main.py
│   └── models.py
├── documents/
├── tests/
│   ├── test_agents.py
│   ├── test_ingestion.py
│   └── test_api.py
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
