"""Runtime configuration.

Config comes from three places, in increasing order of precedence:

1. Defaults in the ``Config`` dataclass below.
2. An optional JSON file (``--config path/to/config.json``).
3. Environment variables (``KBH_*`` and ``ANTHROPIC_API_KEY``).

No secrets are ever stored in the repo or in config files: the only secret
(the Anthropic API key) is read from the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    """Everything the harness needs to run, in one place."""

    # --- retrieval -----------------------------------------------------
    chunk_size: int = 400        # characters per chunk
    chunk_overlap: int = 80      # shared characters between neighbours
    top_k: int = 4               # chunks retrieved per query

    # --- routing / answering -------------------------------------------
    min_route_confidence: float = 0.35  # below -> route "other"
    min_retrieval_score: float = 0.05   # below -> escalate, don't guess
    llm_model: str = "claude-sonnet-4-5"
    llm_max_tokens: int = 512
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2

    # --- observability -------------------------------------------------
    log_path: str = "reports/traces.jsonl"

    # --- paths ---------------------------------------------------------
    kb_dir: str = "data/kb"
    store_path: str = "data/store.json"
    eval_set_path: str = "evals/eval_set.json"
    multiturn_path: str = "evals/multiturn.json"

    @classmethod
    def load(cls, config_path: str | None = None) -> "Config":
        """Build a Config from defaults + optional JSON file + env vars."""
        cfg = cls()
        if config_path:
            data = json.loads(Path(config_path).read_text())
            known = {f.name for f in fields(cls)}
            for key, value in data.items():
                if key not in known:
                    raise ValueError(f"unknown config key: {key}")
                setattr(cfg, key, value)
        # Environment overrides (KBH_TOP_K=8 etc.). Ints/floats are coerced.
        for f in fields(cls):
            env_key = f"KBH_{f.name.upper()}"
            if env_key in os.environ:
                setattr(cfg, f.name, f.type(os.environ[env_key]))
        return cfg

    @property
    def anthropic_api_key(self) -> str | None:
        """The only secret in the system, read from the environment."""
        return os.environ.get("ANTHROPIC_API_KEY")
