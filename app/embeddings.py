"""Embedding backends.

* SentenceTransformerEmbedder: real local model (bge-small, ~130 MB, CPU friendly). Free, no API.
* HashEmbedder: dependency-free bag-of-words hashing, used in tests/CI smoke runs.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self._prefix = _BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self._prefix + text, normalize_embeddings=True).tolist()


class HashEmbedder:
    """Deterministic hashing-trick embedder. Not semantic, but fast and offline."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9+#.]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def get_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(model_name)
