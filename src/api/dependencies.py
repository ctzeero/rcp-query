"""Application state, lifespan, and FastAPI dependencies."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    GEMINI_MODEL,
    LLM_TEMPERATURE_RESPONSE,
    REQUIRED_ENV_VARS,
)
from src.querying.query_parser import QueryParser

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Shared application state initialised once at startup."""

    query_parser: QueryParser = field(default=None)  # type: ignore[assignment]
    response_llm: ChatGoogleGenerativeAI = field(default=None)  # type: ignore[assignment]


def check_env_vars() -> list[str]:
    """Return list of missing required environment variables."""
    return [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup, tear down on shutdown."""
    missing = check_env_vars()
    if missing:
        logger.error(
            "Missing required environment variables: %s. "
            "Copy .env.example to .env and fill in your API keys.",
            ", ".join(missing),
        )

    state = AppState(
        query_parser=QueryParser(),
        response_llm=ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=LLM_TEMPERATURE_RESPONSE,
        ),
    )
    application.state.deps = state
    yield
