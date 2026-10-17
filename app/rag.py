"""The RAG pipeline: question -> retrieval -> grounded, cited answer."""
from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass

from .config import Settings
from .guard import clean_question, looks_like_injection
from .prompts import INJECTION_NOTE, SYSTEM_PROMPT, USER_TEMPLATE
from .retriever import Hit, HybridRetriever

log = logging.getLogger("rag")

MAX_HISTORY_TURNS = 4


@dataclass
class Prepared:
    question: str
    hits: list[Hit]
    messages: list[dict]
    flagged: bool


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm, settings: Settings):
        self.retriever = retriever
        self.llm = llm
        self.s = settings
        self.contact = f"LinkedIn: {settings.owner_linkedin}"

    def _retrieval_query(self, question: str, history: list[dict]) -> str:
        # Cheap conversational query rewriting with zero extra LLM calls (saves free quota):
        # short follow-ups like "what about at Deloitte?" borrow the previous user turn.
        prev = [h["content"] for h in history if h.get("role") == "user"]
        if prev and len(question.split()) <= 8:
            return f"{prev[-1]} {question}"
        return question

    def prepare(self, question: str, history: list[dict] | None = None) -> Prepared:
        history = [h for h in (history or []) if h.get("role") in ("user", "assistant")]
        history = history[-2 * MAX_HISTORY_TURNS:]
        q = clean_question(question, self.s.max_question_chars)
        flagged = looks_like_injection(q)

        hits = self.retriever.search(self._retrieval_query(q, history), top_k=self.s.top_k)
        context = "\n".join(f"[{i}] ({h.chunk['title']}) {h.chunk['text']}" for i, h in enumerate(hits, 1))

        system = SYSTEM_PROMPT.format(owner=self.s.owner_name, contact=self.contact)
        messages = [{"role": "system", "content": system}]
        for h in history:
            messages.append({"role": h["role"], "content": clean_question(h["content"], 1500)})
        messages.append({"role": "user", "content": USER_TEMPLATE.format(
            owner=self.s.owner_name, context=context, question=q,
            note=INJECTION_NOTE if flagged else "")})
        return Prepared(q, hits, messages, flagged)

    @staticmethod
    def sources(hits: list[Hit]) -> list[dict]:
        return [{"n": i, "title": h.chunk["title"], "section": h.chunk["section"],
                 "text": h.chunk["text"], "score": round(h.score, 4)}
                for i, h in enumerate(hits, 1)]

    def stream(self, prepared: Prepared) -> Iterator[str]:
        t0 = time.perf_counter()
        n = 0
        for delta in self.llm.stream(prepared.messages):
            n += len(delta)
            yield delta
        log.info("answered q=%r chars=%d provider=%s flagged=%s latency_ms=%d",
                 prepared.question[:80], n, getattr(self.llm, "last_provider", None),
                 prepared.flagged, (time.perf_counter() - t0) * 1000)

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        p = self.prepare(question, history)
        return {"answer": "".join(self.stream(p)).strip(), "sources": self.sources(p.hits)}
