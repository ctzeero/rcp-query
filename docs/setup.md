# Setup

This page covers everything needed to get rcp-query running locally — from prerequisites through a verified healthy system.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.9+ | `python3 --version` to check |
| Google AI API key | For Gemini LLM calls and embeddings. [Get one here](https://aistudio.google.com/apikey) |
| Pinecone API key | For vector storage. [Sign up here](https://www.pinecone.io/) (free tier is sufficient) |

## Install

```bash
cd rcp-query

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

16 direct dependencies, none version-pinned:

```
langchain              langchain-google-genai   langchain-community
google-generativeai    pinecone[grpc]           pydantic
fastapi                uvicorn                  httpx
streamlit              python-dateutil          holidays
rapidfuzz              lark                     python-dotenv
pytest
```

## Configure

```bash
cp .env.example .env
```

Fill in the two required keys:

```
GOOGLE_API_KEY=<your key>
PINECONE_API_KEY=<your key>
```

Everything else has sensible defaults. The full list of 30+ environment variables is documented in `.env.example` with groupings for LLM, embeddings, Pinecone, query parsing, retrieval, validation, and frontend settings.

Key defaults worth knowing:

| Variable | Default | Why it matters |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.0-flash` | Model used for both parsing and response generation |
| `EMBEDDING_DIMENSION` | `768` | Gemini supports configurable dimensionality; 768 balances quality and cost |
| `REFERENCE_DATE` | today's date | Anchors relative expressions like "last week" |
| `DATASET_START_DATE` | `2023-11-01` | Earliest receipt in the dataset |
| `DATASET_END_DATE` | `2024-01-31` | Latest receipt in the dataset |
| `FUZZY_MATCH_THRESHOLD` | `70` | rapidfuzz score for merchant name matching (0-100) |

## Ingest

```bash
python ingest.py
```

This runs the full ingestion pipeline:

1. **Parse** — Gemini structured output extracts typed `Receipt` objects from 100 `.txt` files
2. **Chunk** — Each receipt produces 1 receipt-level chunk + N item-level chunks (~700 total)
3. **Embed** — `gemini-embedding-001` generates 768-dim vectors
4. **Upsert** — Batch upsert to Pinecone with retry logic and post-upsert verification

Takes **3-5 minutes**. Progress is logged to stdout:

```
[INFO] Parsing 1/100: receipt_001_grocery_20231107.txt
...
[INFO] Total vectors upserted: 712/712
[INFO] Ingestion complete. 100/100 receipts processed. 0 skipped.
```

To re-ingest from a clean slate, delete the Pinecone index first:

```bash
python -c "
from src.config import PINECONE_INDEX
from src.vectorstore.pinecone_client import get_pinecone_client
pc = get_pinecone_client()
pc.delete_index(PINECONE_INDEX)
print('Index deleted.')
"
```

## Run

Two processes are needed — the FastAPI backend and the Streamlit UI:

```bash
# Terminal 1: Backend
uvicorn api:app --reload

# Terminal 2: Frontend (with venv activated)
streamlit run app.py
```

| Service | URL |
|---|---|
| FastAPI API | `http://localhost:8000` |
| Streamlit UI | `http://localhost:8501` |
| API docs (auto-generated) | `http://localhost:8000/docs` |

## Verify

Health check:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{
  "status": "healthy",
  "index": "receipt-intelligence",
  "total_vectors": 712,
  "dimension": 768
}
```

Quick API test:

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did I spend in December?"}' | python -m json.tool
```

## Tests

```bash
python -m pytest tests/ -v
```

All tests are self-contained with mocked dependencies. No API keys or running services are required.

| Test file | Coverage |
|---|---|
| `test_aggregator.py` | sum, avg, min, max, count, group-by, tip breakdown |
| `test_chunker.py` | Receipt-level and item-level chunk generation, metadata |
| `test_retriever_filters.py` | Filter construction, Pinecone filter translation |
| `test_query_log.py` | Query logging behavior |
| `test_receipt_samples.py` | Sample receipt validation |
