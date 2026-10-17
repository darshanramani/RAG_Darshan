"""Ingestion CLI: data/raw/* -> scrubbed text + chunks.jsonl in data/processed/.

    python -m app.ingest

Only data/processed/ is committed and deployed. data/raw/ (the original resumes,
with phone number etc.) stays on your machine.
"""
from __future__ import annotations

import json
import re
import sys

from .chunking import chunk_resume
from .config import settings
from .loaders import SUPPORTED, load
from .pii import find_pii, scrub


def _norm(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def run() -> int:
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in settings.raw_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No resumes found in {settings.raw_dir}")
        return 1

    seen, chunks = set(), []
    for path in files:
        clean = scrub(load(path))
        leaks = find_pii(clean)
        if leaks:
            print(f"PII survived scrubbing in {path.name}: {leaks}")
            return 2
        (settings.processed_dir / f"{path.stem}.txt").write_text(clean, encoding="utf-8")
        new = 0
        for c in chunk_resume(clean, source=path.name, owner=settings.owner_name,
                                extra_links=[settings.owner_linkedin, settings.owner_portfolio]):
            key = _norm(c.text)
            if key in seen:  # same bullet in several resume versions -> keep once
                continue
            seen.add(key)
            chunks.append(c)
            new += 1
        print(f"{path.name}: {new} chunks")

    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    print(f"Wrote {len(chunks)} chunks -> {settings.chunks_path.relative_to(settings.processed_dir.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
