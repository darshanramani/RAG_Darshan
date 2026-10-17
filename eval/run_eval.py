"""Evaluate retrieval (and optionally answers) on eval/qa.jsonl.

Retrieval only (free, offline-capable):
    python -m eval.run_eval                          # real embeddings + reranker
    python -m eval.run_eval --backend hash --no-rerank   # fully offline smoke test

With answers (uses your free LLM quota, ~35 calls):
    python -m eval.run_eval --answers

A question counts as a retrieval hit@k if any of its `expect` strings appears
(case-insensitive) in one of the top-k chunks. MRR uses the first such chunk.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.retriever import HybridRetriever, get_reranker, load_chunks  # noqa: E402

DECLINE = re.compile(r"\b(can't|cannot|can not|unable|not able|only (answer|discuss|help)|isn't covered|"
                     r"not covered|not (mentioned|listed|included)|happy to discuss|directly|instead)\b", re.I)


def first_hit_rank(hits, expect):
    for rank, h in enumerate(hits, 1):
        t = h.chunk["text"].lower()
        if any(e.lower() in t for e in expect):
            return rank
    return None


def retrieval_eval(retriever, items, k, mode):
    ranks = [first_hit_rank(retriever.search(it["q"], top_k=k, mode=mode), it["expect"]) for it in items]
    n = len(items)
    return {
        "mode": mode,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        f"hit@{k}": sum(r is not None for r in ranks) / n,
        "MRR": sum(1 / r for r in ranks if r) / n,
        "misses": [it["q"] for it, r in zip(items, ranks) if r is None],
    }


def answer_eval(rag, items):
    rows = []
    for it in items:
        out = rag.answer(it["q"])
        a = out["answer"]
        if it.get("should_decline"):
            ok = bool(DECLINE.search(a))
            kind = "decline"
        else:
            ok = all(x.lower() in a.lower() for x in it.get("answer_has", [])) and bool(re.search(r"\[\d+\]", a))
            kind = "answer"
        rows.append({"q": it["q"], "kind": kind, "pass": ok, "answer": a})
        time.sleep(2.5)  # stay well under free-tier requests/minute
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=settings.embedding_backend)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k", type=int, default=settings.top_k)
    ap.add_argument("--answers", action="store_true", help="also grade LLM answers (uses API quota)")
    ap.add_argument("--min-hit", type=float, default=0.0, help="exit 1 if hybrid hit@k is below this")
    args = ap.parse_args()

    items = [json.loads(l) for l in (ROOT / "eval" / "qa.jsonl").read_text().splitlines() if l.strip()]
    retrieval_items = [it for it in items if it["expect"]]
    chunks = load_chunks(settings.chunks_path)
    embedder = get_embedder(args.backend, settings.embedding_model)

    results = []
    base = HybridRetriever(chunks, embedder, None, settings.candidates_k)
    for mode in ("sparse", "dense", "hybrid"):
        results.append(retrieval_eval(base, retrieval_items, args.k, mode))
    reranked = None
    if not args.no_rerank and args.backend != "hash":
        reranked = HybridRetriever(chunks, embedder, get_reranker(settings.reranker_model), settings.candidates_k)
        r = retrieval_eval(reranked, retrieval_items, args.k, "hybrid")
        r["mode"] = "hybrid + rerank"
        results.append(r)

    lines = [f"# Retrieval eval — {len(retrieval_items)} questions, embeddings: "
             f"{'hash (offline)' if args.backend == 'hash' else settings.embedding_model}\n",
             f"| mode | hit@1 | hit@3 | hit@{args.k} | MRR |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['mode']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r[f'hit@{args.k}']:.2f} | {r['MRR']:.3f} |")
    for r in results:
        if r["misses"]:
            lines.append(f"\n**{r['mode']} misses:** " + "; ".join(r["misses"]))

    if args.answers:
        from app.llm import ChatLLM, providers_from_settings
        from app.rag import RAGPipeline

        llm = ChatLLM(providers_from_settings(settings), settings.llm_timeout_s, settings.max_answer_tokens)
        if not llm.available:
            sys.exit("Set GROQ_API_KEY and/or GEMINI_API_KEY to grade answers.")
        rag = RAGPipeline(reranked or base, llm, settings)
        rows = answer_eval(rag, items)
        passed = sum(r["pass"] for r in rows)
        lines += [f"\n# Answer eval — {passed}/{len(rows)} passed\n", "| question | type | pass |", "|---|---|---|"]
        lines += [f"| {r['q']} | {r['kind']} | {'✅' if r['pass'] else '❌'} |" for r in rows]
        (ROOT / "eval" / "answers.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    report = "\n".join(lines)
    (ROOT / "eval" / "report.md").write_text(report + "\n")
    print(report)

    hybrid = next(r for r in results if r["mode"] == ("hybrid + rerank" if reranked else "hybrid"))
    if hybrid[f"hit@{args.k}"] < args.min_hit:
        sys.exit(f"hit@{args.k} {hybrid[f'hit@{args.k}']:.2f} below threshold {args.min_hit}")


if __name__ == "__main__":
    main()
