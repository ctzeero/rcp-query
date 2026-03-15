"""Backward-compatible entry point.

Start the server with:
    uvicorn api:app --reload

All logic now lives in src/api/.
"""

import logging

from src.api.routes import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
