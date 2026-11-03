"""Every environment-shaped value, read once.

The point of this file is that the model name appears in exactly two places — here and the
`.env` that overrides it — which is what makes the multi-model eval in M4.8 a config change
rather than a code change (plans/datasurf-plan.md §4).

Nothing here reads a file at import beyond `.env`, and nothing here needs a key to be set:
the app must still run with no key at all, showing M1's ingestion and saying the chat is off.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
MODEL = os.getenv("DATASURF_MODEL", "").strip() or "openai/gpt-6-luna"

# One JSONL file per answered question. Gitignored by `data/*`, and the dir is created on
# first write rather than at import — an unused config should not litter the tree.
TRACE_DIR = ROOT / "data" / "traces"

MAX_RESULT_ROWS = 200    # hard cap on rows a query returns to the app
ANSWER_ROWS = 15         # rows of that result the model is shown. The full handle pattern
                         # (handle + shape + head(3), capped at 400 tokens) lands with M2.2.
