
# Receipt Query

A RAG-based receipt querying system that ingests 100 receipt text files, vectorizes them with a hybrid chunking strategy, stores embeddings in Pinecone, and answers natural language queries via a Streamlit chat UI powered by Gemini 1.5 Flash.

## Quick Start

### Prerequisites

- Python 3.9+
- Google AI API key ([Get one here](https://aistudio.google.com/apikey))
- Pinecone API key ([Sign up here](https://www.pinecone.io/))

### Setup

```bash
cd rcp-query

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Ingest Receipts

```bash
python ingest.py
```

This parses all 100 receipt files, creates hybrid chunks, generates embeddings via `gemini-embedding-001`, and upserts vectors to Pinecone. Takes approximately 3-5 minutes.

### Run the Application

You need two separate terminals — one for the API backend and one for the Streamlit UI.

**Terminal 1: Start the FastAPI backend**

```bash
cd rcp-query
source venv/bin/activate
uvicorn api:app --reload
```

**Terminal 2: Start the Streamlit UI**

```bash
cd rcp-query
source venv/bin/activate
streamlit run app.py
```

The API runs on `http://localhost:8000` and the Streamlit UI on `http://localhost:8501`.

### Run Tests

```bash
# Unit tests (no API keys required)
python -m pytest tests/ -v

# Or query the API directly
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How much did I spend in December?"}'
```
