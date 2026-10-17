---
title: Ask Darshan
emoji: 💬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG assistant that answers questions about my resume
---

# Ask Darshan — a resume RAG assistant

A public chatbot that answers recruiters' questions about my experience, grounded in my resume with clickable citations. It's built end to end on free infrastructure: local embeddings, free-tier LLM APIs, and Hugging Face Spaces.

**Live demo:** _add your Space URL here_ · **Portfolio:** https://darshanramani.github.io/portfolio/

## Architecture

```mermaid
flowchart LR
  subgraph Offline ingestion
    R[Resumes<br/>PDF · DOCX · LaTeX · MD] --> L[Loaders] --> P[PII scrubber<br/>phone · address · postal]
    P --> C[Structure-aware chunker<br/>role-contextual bullets,<br/>skills, timeline] --> J[(chunks.jsonl)]
  end
  subgraph Online, per question
    Q[Question + chat history] --> G[Guards<br/>rate limit · length · injection flag]
    G --> QR[Follow-up query rewrite]
    QR --> D[Dense: ChromaDB<br/>bge-small embeddings]
    QR --> S[Sparse: BM25]
    D --> F[Reciprocal Rank Fusion]
    S --> F
    F --> RR[Cross-encoder reranker] --> PR[Grounded prompt<br/>numbered context]
    PR --> LLM{Groq → Gemini<br/>auto failover}
    LLM --> A[Streamed answer + citations]
  end
  J --> D
  J --> S
```

| Layer | Choice | Why |
|---|---|---|
| Chunking | One chunk per experience bullet, prefixed with its role and dates, plus skill, education and timeline chunks | Resumes are dense, and fixed-size chunks blur roles together |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally on CPU | Free, fast, strong retrieval quality for its size |
| Vector store | ChromaDB, in memory, rebuilt at boot | The data is tiny, so no volume is needed on free hosting |
| Retrieval | BM25 + dense, fused with RRF, then the `ms-marco-MiniLM` cross-encoder | Exact tool names (XGBoost, AKS) suit BM25; paraphrases suit dense search |
| LLM | Groq (Llama) first, Gemini Flash as fallback, both through OpenAI-compatible APIs | Free tiers, with automatic failover on 429s and errors |
| API | FastAPI with SSE streaming, per-IP rate limiting, prompt-injection flagging | Built for a public endpoint |
| UI | Dependency-free chat page, plus a one-line embeddable widget | |
| Ops | Docker, GitHub Actions (tests + retrieval-eval quality gate), auto-deploy to HF Spaces | |

## Evaluation

`eval/qa.jsonl` holds 35 questions: 32 retrieval questions and 3 that the bot should decline (off-topic, prompt injection, salary). CI fails the build if hybrid hit@6 drops below 0.90.

```bash
python -m eval.run_eval              # retrieval: sparse vs dense vs hybrid vs hybrid+rerank
python -m eval.run_eval --answers    # also grade LLM answers (uses free API quota)
```

<!-- paste eval/report.md here after the first real run -->

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env             # add GROQ_API_KEY and/or GEMINI_API_KEY
python -m app.ingest             # data/raw/* -> data/processed/chunks.jsonl (PII scrubbed)
set -a; source .env; set +a
uvicorn app.main:app --reload    # http://localhost:8000
pytest -q                        # offline tests, no keys needed
```

No keys yet? Run `LLM_BACKEND=fake EMBEDDING_BACKEND=hash uvicorn app.main:app` to try the UI offline.

## Updating the resume

1. Put new or updated resume files in `data/raw/`. This folder is git-ignored, so originals never leave your machine.
2. Run `python -m app.ingest` and check `data/processed/` (the scrubbed text is what gets published).
3. Commit `data/processed/` and push. CI re-runs the evals, then deploys.

## Deploy (free)

1. Create a **Docker** Space at huggingface.co/new-space on the free CPU hardware.
2. In the Space's **Settings → Variables and secrets**, add the secrets `GROQ_API_KEY` and `GEMINI_API_KEY`.
3. In this GitHub repo, add the secret `HF_TOKEN` (an HF token with write access) and the variable `HF_SPACE` (e.g. `darshanramani/ask-darshan`).
4. Push to `main`. CI runs, and on success `deploy.yml` pushes to the Space.
5. To add the widget to the portfolio, put this before `</body>`:
   ```html
   <script src="https://<your-space>.hf.space/static/widget.js" defer></script>
   ```

Free Spaces sleep when idle, so the first question after a quiet spell takes a little longer while the Space wakes up.

## Privacy

- Phone numbers, street addresses and postal codes are stripped at ingestion. A test fails the build if any survive in the published chunks.
- The system prompt limits answers to professional topics. It declines salary, immigration, age and other personal questions, and treats user text as data, not instructions.
