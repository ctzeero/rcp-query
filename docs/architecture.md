# Architecture

rcp-query is a RAG-based receipt query system with two distinct pipelines and a clean separation between API and UI.

## System Overview

> **Interactive flowchart**: Open [`rcp-query-flowchart-v1.html`](rcp-query-flowchart-v1.html) in a browser to view the full system diagram.

## Project Structure

```
src/
├── config.py                      Centralized settings (env vars + defaults)
├── models.py                      Pydantic v2: Receipt, QueryRequest, QueryResponse, AggregationResult
├── api/
│   ├── routes.py                  FastAPI /query, /ingest, /health
│   ├── dependencies.py            AppState, lifespan, env var checks
│   └── prompts.py                 Response prompt template + formatting helpers
├── ingestion/
│   ├── parser.py                  Raw .txt -> Receipt (Gemini structured output + regex fallbacks)
│   └── chunker.py                 Receipt -> LangChain Documents (receipt-level + item-level)
├── vectorstore/
│   ├── pinecone_client.py         Pinecone connection, index management, embedding instances
│   └── indexer.py                 Batch embed + upsert with exponential backoff
└── querying/
    ├── date_resolver.py           Deterministic temporal pre-parser (regex + holidays + dateutil)
    ├── query_parser.py            Fast-path + Gemini structured output + post-parse fixes
    ├── retriever.py               Vector + metadata retrieval with fuzzy matching (rapidfuzz)
    └── aggregator.py              Pure Python aggregation: sum, avg, min, max, count, group-by
```

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| LLM | Gemini 2.0 Flash (`langchain-google-genai`) | Receipt parsing, query parsing, response generation |
| Embeddings | `gemini-embedding-001` (768-dim) | Document and query vectorization |
| Vector DB | Pinecone Serverless (gRPC) | Vector storage and metadata-filtered retrieval |
| Framework | LangChain | Structured output, document model, embedding abstraction |
| API | FastAPI + Uvicorn | Business logic, request validation, error handling |
| UI | Streamlit | Chat interface (thin client, calls API via `httpx`) |
| Data Models | Pydantic v2 | Receipt schema, query/response models, validation |
| Date Parsing | `python-dateutil`, `holidays` | Deterministic temporal resolution |
| Fuzzy Matching | `rapidfuzz` | Post-retrieval merchant name matching |

## Ingestion Pipeline

Run once via `python ingest.py`.

1. **Parse** — Each `.txt` receipt is sent to Gemini `with_structured_output()`. Category and receipt ID are extracted from the filename. Regex fallbacks catch fields the LLM occasionally misses (`card_last_four`, `tax_rate`, `tip_percentage`).
2. **Chunk** — Each receipt produces a receipt-level chunk (full summary + metadata) and one item-level chunk per line item (item text + context). Dates stored as epoch seconds for Pinecone range filtering.
3. **Embed + Upsert** — Batched with `gemini-embedding-001`, deterministic vector IDs (SHA-256), exponential backoff on rate limits, post-upsert verification polling.

## Query Pipeline

Triggered per request to `POST /query`.

1. **Date Resolution** — Deterministic regex + `holidays` + `dateutil` resolves temporal expressions (explicit dates, quarters, holidays, relative expressions). Uses two reference points: `REFERENCE_DATE` for relative expressions, `DATASET_END_DATE` for year inference.
2. **Fast-Path** — If date resolution succeeded and no complex signals detected (merchants, categories, aggregation, tips), builds `ParsedQuery` locally. Skips the LLM entirely (~0ms vs ~35s).
3. **LLM Parsing** — Gemini structured output converts natural language into a `ParsedQuery` with search text, filters, date range, aggregation type, and retrieval strategy flags.
4. **Post-Parse Fixes** — Deterministic overrides for date ranges, tip filters, and category routing the LLM missed.
5. **Out-of-Range Check** — If the date range falls entirely outside the dataset window (Nov 2023 - Jan 2024), returns a formatted message immediately. No Pinecone or LLM call.
6. **Retrieval** — Three strategies: metadata-only (aggregation queries, zero-vector + filters), vector search (semantic queries), or hybrid (over-fetch 3x, then fuzzy match with `rapidfuzz`).
7. **Aggregation** — Python computes all math (sum, avg, min, max, count, group-by) from document metadata. The LLM never performs arithmetic.
8. **Response Generation** — The LLM receives retrieved receipts, pre-computed aggregation, and instructions to report numbers verbatim.

## Key Design Decisions

**Hybrid chunking** — Receipt-level + item-level chunks (~700 vectors for 100 receipts). Receipt chunks handle totals/aggregation, item chunks handle specific item searches. Query parser routes between them via `chunk_type` metadata.

**Deterministic date resolution before LLM** — LLMs are unreliable with temporal expressions (wrong year, missing end dates, timezone artifacts). The resolver runs before the LLM to enable fast-path, and again after to override any dates the LLM got wrong.

**Fast-path query parsing** — Simple temporal queries skip the LLM entirely. Conservative signal detection ensures false negatives (unnecessary LLM calls) over false positives (wrong results).

**Fuzzy merchant matching** — Pinecone only supports exact `$eq` on strings. Soft filters (merchant, category, city) are applied post-retrieval using `rapidfuzz.fuzz.partial_ratio` so "Akiko Sushi" matches "AKIKO'S SUSHI".

**Anti-hallucination** — The LLM handles language only. All computation is Python-side. Zero results return a canned message. Receipt IDs enable source verification. Queries are validated against the dataset window before retrieval.

**Gemini over OpenAI** — Configurable embedding dimensionality (768 vs fixed 1536) was the primary driver — high-quality retrieval at half the storage cost.

**Custom query parser over SelfQueryRetriever** — `langchain-pinecone` has unresolved Python 3.12+ compatibility issues. Beyond that, `SelfQueryRetriever` has no hooks for fuzzy matching, aggregation routing, fast-path optimization, or deterministic date resolution.

**Category from filename** — The synthetic dataset has merchant-item mismatches (e.g., "Gap" selling grocery items). Filename labels are ground truth. A production system would need LLM-based classification.

**Receipt parsing: Gemini over regex** — 9 receipt categories with different structures (SKUs, tips, prescriptions, loyalty discounts). One Gemini prompt handles all variations; regex catches the three fields the LLM occasionally misses.

## Known Limitations

| Limitation | Why | Production Solution |
|---|---|---|
| No OCR | Only handles pre-digitized `.txt` files | Tesseract or Google Vision |
| Category from filename | Ground truth for synthetic data | LLM classification with confidence thresholds |
| Single-user | Portfolio project scope | Auth + per-user Pinecone namespaces |
| No caching | Every query hits Pinecone + Gemini | Redis cache for common aggregations |
| Fixed dataset window | Nov 2023 - Jan 2024 | Dynamic range detection from index metadata |
| No streaming | Response returned as a single block | FastAPI `StreamingResponse` + Streamlit incremental rendering |
