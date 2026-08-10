"""Load the portfolio-agents runtime config from the demo's own ``.env``.

Works both on the GPU box (where the agent fleet runs) and under the FastAPI
backend. All secrets come from ``.env`` — nothing is hard-coded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_env() -> None:
    """Best-effort .env loader (no python-dotenv dependency required)."""
    # Look for a .env next to the package, then the repo root, then CWD.
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / ".env",          # alongside pa_agents/
        here.parent.parent.parent / ".env",   # demo root
        Path.cwd() / ".env",
    ]
    for path in candidates:
        if path.is_file():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                os.environ.setdefault(key, val)
            break


_load_env()


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("SINGLESTORE_HOST", "127.0.0.1")
    port: int = int(os.environ.get("SINGLESTORE_PORT", "3306"))
    user: str = os.environ.get("SINGLESTORE_USER", "admin")
    password: str = os.environ.get("SINGLESTORE_PASSWORD", "")
    database: str = os.environ.get("SINGLESTORE_DATABASE", "portfolio_agents")
    project_id: str = os.environ.get("SINGLESTORE_PROJECT_ID", "")

    llm_endpoint: str = os.environ.get("LLM_ENDPOINT", "")

    # Model JWTs (per-model bearer keys issued by the SingleStore endpoint).
    opus_key: str = os.environ.get("OPUS_KEY", "")
    sonnet_key: str = os.environ.get("SONNET_KEY", "")
    haiku_key: str = os.environ.get("HAIKU_KEY", "")
    qwen_key: str = os.environ.get("QWEN_KEY", "")

    # Model IDs.
    opus_model: str = os.environ.get("OPUS_MODEL", "")
    sonnet_model: str = os.environ.get("SONNET_MODEL", "")
    haiku_model: str = os.environ.get("HAIKU_MODEL", "")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "")


settings = Settings()
