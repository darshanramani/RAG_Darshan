"""Central configuration. Everything is overridable with environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # --- identity of the person the bot represents -------------------------
    owner_name: str = _env("OWNER_NAME", "Darshan Ramani")
    owner_linkedin: str = _env("OWNER_LINKEDIN", "https://www.linkedin.com/in/darshan-ramani/")
    owner_portfolio: str = _env("OWNER_PORTFOLIO", "https://darshanramani.github.io/portfolio/")

    # --- paths ---------------------------------------------------------------
    raw_dir: Path = ROOT / "data" / "raw"            # private, git-ignored
    processed_dir: Path = ROOT / "data" / "processed"  # scrubbed, safe to publish
    chunks_path: Path = ROOT / "data" / "processed" / "chunks.jsonl"
    chroma_dir: Path = Path(_env("CHROMA_DIR", str(ROOT / ".chroma")))

    # --- retrieval -------------------------------------------------------------
    embedding_backend: str = _env("EMBEDDING_BACKEND", "sentence-transformers")  # or "hash"
    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    reranker_model: str = _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = _env("USE_RERANKER", "true").lower() == "true"
    candidates_k: int = int(_env("CANDIDATES_K", "20"))
    top_k: int = int(_env("TOP_K", "6"))

    # --- LLM providers (OpenAI-compatible endpoints, tried in order) ----------
    groq_api_key: str = _env("GROQ_API_KEY", "")
    groq_model: str = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_api_key: str = _env("GEMINI_API_KEY", "")
    gemini_model: str = _env("GEMINI_MODEL", "gemini-2.5-flash")
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "30"))
    max_answer_tokens: int = int(_env("MAX_ANSWER_TOKENS", "500"))

    # --- public-facing protections ---------------------------------------------
    max_question_chars: int = int(_env("MAX_QUESTION_CHARS", "500"))
    rate_limit_per_minute: int = int(_env("RATE_LIMIT_PER_MINUTE", "8"))
    rate_limit_per_day: int = int(_env("RATE_LIMIT_PER_DAY", "60"))
    allowed_origins: list[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in _env(
                "ALLOWED_ORIGINS",
                "https://darshanramani.github.io,http://localhost:8000,http://127.0.0.1:8000",
            ).split(",")
            if o.strip()
        ]
    )


settings = Settings()
