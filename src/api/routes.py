"""FastAPI route handlers: /query, /ingest, /health."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from src.config import (
    CONTENT_PREVIEW_LENGTH,
    DATASET_END_DATE,
    DATASET_START_DATE,
    MAX_DISPLAY_RECEIPTS,
    PINECONE_INDEX,
    RECEIPT_DIR,
    REFERENCE_DATE,
)
from src.models import QueryRequest, QueryResponse
from src.querying.aggregator import aggregate
from src.querying.date_resolver import dates_outside_dataset
from src.querying.retriever import retrieve
from src.vectorstore.pinecone_client import get_index_stats, get_pinecone_client, ensure_index

from src.api.dependencies import AppState, check_env_vars, lifespan
from src.api.prompts import (
    RESPONSE_PROMPT,
    build_out_of_range_answer,
    build_search_context,
    format_aggregation_info,
    format_receipt_for_context,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Receipt Query", version="1.0.0", lifespan=lifespan)


def get_app_state() -> AppState:
    """FastAPI dependency that provides the shared application state."""
    return app.state.deps


@app.get("/health")
async def health():
    """System health check."""
    try:
        pc = get_pinecone_client()
        stats = get_index_stats(pc, PINECONE_INDEX)
        return {
            "status": "healthy",
            "index": PINECONE_INDEX,
            "total_vectors": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_receipts(
    request: QueryRequest,
    state: AppState = Depends(get_app_state),
):
    """Query receipts using natural language."""
    missing = check_env_vars()
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Service not configured. Check server logs for details.",
        )

    try:
        parsed = state.query_parser.parse(request.query)

        if dates_outside_dataset(parsed.date_range, DATASET_START_DATE, DATASET_END_DATE):
            dr = parsed.date_range
            answer = build_out_of_range_answer(
                REFERENCE_DATE, dr.start, dr.end,
                DATASET_START_DATE, DATASET_END_DATE,
            )
            return QueryResponse(
                answer=answer,
                receipts=[],
                aggregation=None,
                query_metadata={
                    "parsed_filters": parsed.filters,
                    "chunk_type": parsed.chunk_type,
                    "aggregation_type": parsed.aggregation,
                    "date_range": dr.model_dump(),
                    "total_results": 0,
                    "displayed_results": 0,
                    "skipped_reason": "date_range_outside_dataset",
                },
            )

        docs = retrieve(parsed)

        aggregation_result = None
        if parsed.aggregation:
            aggregation_result = aggregate(docs, parsed.aggregation)

        receipt_dicts = []
        for doc in docs[:MAX_DISPLAY_RECEIPTS]:
            receipt_dicts.append({
                "content": doc.page_content[:CONTENT_PREVIEW_LENGTH],
                **{k: v for k, v in doc.metadata.items() if k != "score"},
            })

        context_lines = [format_receipt_for_context(doc.metadata) for doc in docs[:MAX_DISPLAY_RECEIPTS]]
        context = "\n".join(context_lines) if context_lines else "No matching receipts found."

        aggregation_info = format_aggregation_info(aggregation_result)
        search_context = build_search_context(REFERENCE_DATE, parsed.date_range)

        prompt = RESPONSE_PROMPT.format(
            query=request.query,
            count=len(docs),
            max_display=MAX_DISPLAY_RECEIPTS,
            context=context,
            aggregation_info=aggregation_info,
            search_context=search_context,
        )

        response = state.response_llm.invoke(prompt)
        answer = response.content

        extra_count = max(0, len(docs) - MAX_DISPLAY_RECEIPTS)
        if extra_count > 0:
            answer += f"\n\n(Showing top {MAX_DISPLAY_RECEIPTS} of {len(docs)} matching receipts)"

        query_metadata = {
            "parsed_filters": parsed.filters,
            "chunk_type": parsed.chunk_type,
            "aggregation_type": parsed.aggregation,
            "date_range": parsed.date_range.model_dump() if parsed.date_range else None,
            "total_results": len(docs),
            "displayed_results": min(len(docs), MAX_DISPLAY_RECEIPTS),
        }

        return QueryResponse(
            answer=answer,
            receipts=receipt_dicts,
            aggregation=aggregation_result,
            query_metadata=query_metadata,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except ConnectionError as e:
        logger.exception("Upstream service unreachable")
        raise HTTPException(status_code=502, detail=f"Upstream service error: {e}")
    except TimeoutError as e:
        logger.exception("Upstream service timed out")
        raise HTTPException(status_code=504, detail=f"Upstream timeout: {e}")
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "rate" in err_str or "quota" in err_str:
            logger.warning("Rate limited by upstream: %s", e)
            raise HTTPException(status_code=429, detail="Rate limited by upstream AI service. Try again shortly.")
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Something went wrong. Check server logs.")


@app.post("/ingest")
async def ingest_receipts():
    """Trigger ingestion of receipt files."""
    missing = check_env_vars()
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Service not configured. Check server logs for details.",
        )

    try:
        from src.ingestion.chunker import chunk_receipts
        from src.ingestion.parser import ReceiptParser
        from src.vectorstore.indexer import upsert_documents

        pc = get_pinecone_client()
        index_name = ensure_index(pc)

        parser = ReceiptParser()
        receipts, failures = parser.parse_directory(RECEIPT_DIR)

        if not receipts:
            raise HTTPException(status_code=500, detail="No receipts parsed successfully")

        documents = chunk_receipts(receipts)
        upserted = upsert_documents(documents, index_name=index_name, pc=pc)

        return {
            "status": "success",
            "receipts_parsed": len(receipts),
            "receipts_failed": len(failures),
            "documents_created": len(documents),
            "vectors_upserted": upserted,
            "failures": [{"file": f, "error": e} for f, e in failures],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during ingestion")
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")
