"""Protections for a public endpoint: input limits, injection flags, rate limiting."""
from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque

_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(previous|prior|above|your)\s+(instructions|rules|prompt)"
    r"|system\s*prompt|you\s+are\s+now|act\s+as\s+(?!a\s+recruiter)|developer\s+mode|jailbreak"
    r"|reveal\s+(your\s+)?(instructions|prompt)|repeat\s+the\s+(text|words)\s+above",
    re.I,
)


def clean_question(q: str, max_chars: int) -> str:
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", q or "")
    return re.sub(r"\s+", " ", q).strip()[:max_chars]


def looks_like_injection(q: str) -> bool:
    return bool(_INJECTION.search(q))


class RateLimiter:
    """Sliding-window limits per client (minute + day). In-memory: fine for one replica."""

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute, self.per_day = per_minute, per_day
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, int]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        with self._lock:
            q = self._hits[client]
            while q and now - q[0] > 86400:
                q.popleft()
            last_min = sum(1 for t in q if now - t < 60)
            if len(q) >= self.per_day:
                return False, int(86400 - (now - q[0])) + 1
            if last_min >= self.per_minute:
                oldest_in_min = next(t for t in q if now - t < 60)
                return False, int(60 - (now - oldest_in_min)) + 1
            q.append(now)
            return True, 0
