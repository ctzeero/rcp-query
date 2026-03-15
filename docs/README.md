# Technical Documentation

Technical documentation for **rcp-query**, a RAG-based receipt query system built with Gemini, Pinecone, FastAPI, and Streamlit.

This documentation is written for technical interviewers evaluating the project. It covers how to run the system, how it works, what it can do, and why it was built this way.

## Contents

| Document | Description |
|---|---|
| [Setup](setup.md) | Prerequisites, install, configure, ingest, run, verify, and test |
| [Architecture](architecture.md) | System overview, tech stack, pipelines, key design decisions, and known limitations |
## Quick Orientation

rcp-query ingests 100 synthetic receipt text files and makes them queryable via natural language. The system:

1. **Parses** receipts with Gemini structured output into typed Pydantic models
2. **Chunks** each receipt at two granularities (receipt-level + item-level) for hybrid search
3. **Embeds** ~700 chunks using `gemini-embedding-001` (768-dim) and stores them in Pinecone
4. **Answers** natural language questions through an 8-stage query pipeline that combines deterministic date resolution, LLM-based query parsing, vector/metadata retrieval, Python aggregation, and LLM response formatting

The core engineering challenge is making the LLM reliable: dates are resolved deterministically before the LLM, all arithmetic is computed in Python, responses are grounded in retrieved data, and multiple fallback layers correct LLM mistakes.
