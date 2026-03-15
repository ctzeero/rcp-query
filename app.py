"""Streamlit chat UI -- thin client calling FastAPI via httpx."""

from __future__ import annotations

import re

import httpx
import streamlit as st

from src.config import (
    API_BASE_URL,
    HEALTH_CHECK_TIMEOUT,
    MAX_CONVERSATION_HISTORY,
    MAX_QUERY_CHARS,
    MAX_QUERY_SENTENCES,
    QUERY_TIMEOUT,
)


def _count_sentences(text: str) -> int:
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(sentences)


def _escape_dollars(text: str) -> str:
    """Escape bare $ signs so Streamlit doesn't treat them as LaTeX delimiters."""
    return text.replace("$", "\\$")


def _check_api_health() -> dict | None:
    try:
        resp = httpx.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        return resp.json()
    except Exception:
        return None


def _query_api(query: str, history: list[dict]) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_BASE_URL}/query",
            json={"query": query, "conversation_history": history},
            timeout=QUERY_TIMEOUT,
        )
        if resp.status_code == 400:
            return {"error": resp.json().get("detail", "Invalid request")}
        if resp.status_code == 503:
            return {"error": "API keys not configured. Set GOOGLE_API_KEY and PINECONE_API_KEY in your .env file."}
        if resp.status_code == 500:
            return {"error": "Something went wrong on the server. Check the FastAPI terminal for details."}
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API error ({e.response.status_code}). Check the FastAPI terminal for details."}
    except httpx.ConnectError:
        return {"error": "Cannot connect to API. Is the FastAPI server running? (uvicorn api:app)"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def _format_date(raw) -> str:
    """Convert an epoch timestamp or date string to a readable date."""
    if isinstance(raw, (int, float)) and raw > 1_000_000:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(raw, tz=timezone.utc).strftime("%Y-%m-%d")
    return str(raw) if raw else "?"


def _render_receipts(receipts: list[dict]) -> None:
    """Render a list of receipt dicts inside a Streamlit expander."""
    if not receipts:
        return
    with st.expander(f"View {len(receipts)} receipts"):
        for r in receipts:
            chunk = r.get("chunk_type", "")
            display_date = r.get("date_str") or _format_date(r.get("date"))
            if chunk == "receipt":
                st.markdown(
                    _escape_dollars(
                        f"**{r.get('merchant', '?')}** - "
                        f"${r.get('total_amount', 0):.2f} | "
                        f"{display_date} | "
                        f"{r.get('category', '?')} | "
                        f"`{r.get('receipt_id', '?')}`"
                    )
                )
            elif chunk == "item":
                st.markdown(
                    _escape_dollars(
                        f"  {r.get('item_name', '?')} - "
                        f"${r.get('item_price', 0):.2f} at "
                        f"{r.get('merchant', '?')} | "
                        f"{display_date} | "
                        f"`{r.get('receipt_id', '?')}`"
                    )
                )


def _render_aggregation(aggregation: dict | None) -> None:
    """Render an aggregation breakdown inside a Streamlit expander."""
    if not aggregation:
        return
    result = aggregation.get("result")
    if not result or not isinstance(result, dict):
        return
    with st.expander("Aggregation Breakdown"):
        for key, val in result.items():
            if isinstance(val, dict):
                st.markdown(
                    _escape_dollars(
                        f"**{key}**: ${val.get('total', 0):.2f} "
                        f"({val.get('count', 0)} receipts, "
                        f"avg ${val.get('avg', 0):.2f})"
                    )
                )
            else:
                st.markdown(f"**{key}**: {val}")


st.set_page_config(
    page_title="Receipt Intelligence",
    page_icon="",
    layout="wide",
)

st.title("Receipt Query")
st.caption("Ask questions about your receipts using natural language")

with st.sidebar:
    st.header("System Status")
    health = _check_api_health()
    if health and health.get("status") == "healthy":
        st.success(f"API Connected | {health.get('total_vectors', 0)} vectors indexed")
    else:
        st.error("API not connected. Start the FastAPI server first.")
        st.code("uvicorn api:app --reload", language="bash")

    st.divider()
    st.header("Example Queries")
    examples = [
        "How much did I spend in January 2024?",
        "What did I buy last week?",
        "Show me all receipts from December",
        "Find all Whole Foods receipts"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state["prefill_query"] = ex

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["conversation_history"] = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(_escape_dollars(msg["content"]))
        _render_receipts(msg.get("receipts", []))
        _render_aggregation(msg.get("aggregation"))

prefill = st.session_state.pop("prefill_query", None)
user_input = st.chat_input("Ask about your receipts...", max_chars=MAX_QUERY_CHARS)
query = prefill or user_input

if query:
    if _count_sentences(query) > MAX_QUERY_SENTENCES:
        st.warning(f"Please keep your query to {MAX_QUERY_SENTENCES} sentences or fewer.")
    else:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing receipts..."):
                result = _query_api(query, st.session_state["conversation_history"])

            if result and "error" in result:
                st.error(result["error"])
                st.session_state["messages"].append(
                    {"role": "assistant", "content": f"Error: {result['error']}"}
                )
            elif result:
                answer = result.get("answer", "No response received.")
                st.markdown(_escape_dollars(answer))

                receipts = result.get("receipts", [])
                aggregation = result.get("aggregation")

                _render_receipts(receipts)
                _render_aggregation(aggregation)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "receipts": receipts,
                    "aggregation": aggregation,
                })

                st.session_state["conversation_history"].append(
                    {"role": "user", "content": query}
                )
                st.session_state["conversation_history"].append(
                    {"role": "assistant", "content": answer}
                )
                if len(st.session_state["conversation_history"]) > MAX_CONVERSATION_HISTORY:
                    st.session_state["conversation_history"] = (
                        st.session_state["conversation_history"][-MAX_CONVERSATION_HISTORY:]
                    )
            else:
                st.error("No response from API.")
