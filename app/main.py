"""FastAPI app: chat API (streaming SSE + JSON), health, and the static chat UI."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, settings
from .embeddings import get_embedder
from .guard import RateLimiter
from .llm import ChatLLM, FakeLLM, LLMUnavailable, providers_from_settings
from .rag import RAGPipeline
from .retriever import HybridRetriever, get_reranker, load_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")
STATIC = ROOT / "static"
state: dict = {}


def build_pipeline() -> RAGPipeline:
    chunks = load_chunks(settings.chunks_path)
    embedder = get_embedder(settings.embedding_backend, settings.embedding_model)
    reranker = None
    if settings.use_reranker and settings.embedding_backend != "hash":
        reranker = get_reranker(settings.reranker_model)
    retriever = HybridRetriever(chunks, embedder, reranker, candidates_k=settings.candidates_k)
    if os.getenv("LLM_BACKEND") == "fake":
        llm = FakeLLM()
    else:
        llm = ChatLLM(providers_from_settings(settings), settings.llm_timeout_s, settings.max_answer_tokens)
    log.info("pipeline ready: %d chunks, reranker=%s, llm=%s", len(chunks), bool(reranker),
             [p.name for p in getattr(llm, "providers", [])] or type(llm).__name__)
    return RAGPipeline(retriever, llm, settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["rag"] = build_pipeline()
    state["limiter"] = RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_day)
    yield


app = FastAPI(title=f"Ask {settings.owner_name}", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)
    stream: bool = True


def client_id(request: Request) -> str:
    # Behind the HF proxy. The right-most X-Forwarded-For entry is the one the proxy itself
    # added, so a visitor can't dodge the rate limit by sending a fake header.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[-1].strip() if fwd else (request.client.host if request.client else "?")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/api/health")
def health():
    rag: RAGPipeline = state["rag"]
    return {"status": "ok", "chunks": len(rag.retriever.chunks), "llm_configured": rag.llm.available}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    allowed, retry = state["limiter"].check(client_id(request))
    if not allowed:
        return JSONResponse({"error": "Too many questions — please try again shortly."},
                            status_code=429, headers={"Retry-After": str(retry)})
    rag: RAGPipeline = state["rag"]
    if not rag.llm.available:
        return JSONResponse({"error": "The assistant isn't configured yet (no LLM API key)."},
                            status_code=503)

    prepared = rag.prepare(req.question, [t.model_dump() for t in req.history])
    sources = rag.sources(prepared.hits)

    if not req.stream:
        try:
            return {"answer": "".join(rag.stream(prepared)).strip(), "sources": sources}
        except LLMUnavailable as e:
            log.error("all providers failed: %s", e)
            return JSONResponse({"error": "The assistant is busy right now. Please try again in a minute."},
                                status_code=503)

    def events():
        yield _sse({"type": "sources", "sources": sources})
        try:
            for delta in rag.stream(prepared):
                yield _sse({"type": "token", "text": delta})
        except LLMUnavailable as e:
            log.error("all providers failed: %s", e)
            yield _sse({"type": "error", "message": "The assistant is busy right now. Please try again in a minute."})
        yield _sse({"type": "done"})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(Path(STATIC) / "index.html")
