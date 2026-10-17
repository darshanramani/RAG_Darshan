"""Hybrid retrieval: ChromaDB (dense) + BM25 (sparse), fused with RRF, then reranked.

Why hybrid: resume questions are full of exact tokens (XGBoost, AKS, Deloitte, 2023)
that BM25 nails, plus paraphrases ("has he shipped LLM apps?") that dense vectors
handle. Reciprocal Rank Fusion merges both without having to calibrate scores.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .embeddings import Embedder

_STOP = set("a an the of to in on for and or is are was were be by with at as he his him she her "
            "they their what which who does did do has have had about any from that this it its".split())


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9+#]+(?:[./-][a-z0-9+#]+)*", text.lower())
    return [t for t in toks if t not in _STOP]


@dataclass
class Hit:
    chunk: dict
    score: float

    @property
    def id(self) -> str:
        return self.chunk["id"]


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HybridRetriever:
    def __init__(self, chunks: list[dict], embedder: Embedder, reranker=None,
                 candidates_k: int = 20, rrf_k: int = 60):
        import chromadb

        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.embedder = embedder
        self.reranker = reranker
        self.candidates_k = candidates_k
        self.rrf_k = rrf_k

        # Dense index. Data is tiny, so an in-memory Chroma rebuilt at boot is ideal:
        # no volume needed on free hosting, and the index always matches the data.
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("resume")
        except Exception:
            pass
        self.collection = client.create_collection("resume", metadata={"hnsw:space": "cosine"})
        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            embeddings=embedder.embed_documents([c["text"] for c in chunks]),
            metadatas=[{"section": c["section"], "title": c["title"], "source": c["source"]} for c in chunks],
        )
        # Sparse index
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    # -- individual retrievers --------------------------------------------------
    def dense(self, query: str, k: int) -> list[str]:
        res = self.collection.query(query_embeddings=[self.embedder.embed_query(query)],
                                    n_results=min(k, len(self.chunks)))
        return res["ids"][0]

    def sparse(self, query: str, k: int) -> list[str]:
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["id"] for i in order[:k] if scores[i] > 0]

    # -- fusion + rerank ------------------------------------------------------------
    def search(self, query: str, top_k: int = 6, mode: str = "hybrid") -> list[Hit]:
        k = self.candidates_k
        ranked_lists = []
        if mode in ("hybrid", "dense"):
            ranked_lists.append(self.dense(query, k))
        if mode in ("hybrid", "sparse"):
            ranked_lists.append(self.sparse(query, k))

        fused: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, cid in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        candidates = sorted(fused, key=fused.get, reverse=True)[:k]

        if self.reranker is not None and candidates:
            scores = self.reranker.predict([(query, self.by_id[c]["text"]) for c in candidates])
            pairs = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
            return [Hit(self.by_id[c], float(s)) for c, s in pairs[:top_k]]
        return [Hit(self.by_id[c], fused[c]) for c in candidates[:top_k]]


def get_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cpu")
