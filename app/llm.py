"""LLM access through free-tier, OpenAI-compatible endpoints with automatic failover.

Groq and Gemini both expose an OpenAI-style /chat/completions API, so one small
client covers both. Providers are tried in order; if one is rate-limited (429),
erroring (5xx) or times out before producing a token, the next one is used.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

log = logging.getLogger("llm")


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


def providers_from_settings(s) -> list[Provider]:
    out = []
    if s.groq_api_key:
        out.append(Provider("groq", "https://api.groq.com/openai/v1", s.groq_api_key, s.groq_model))
    if s.gemini_api_key:
        out.append(Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                            s.gemini_api_key, s.gemini_model))
    return out


class ChatLLM:
    def __init__(self, providers: list[Provider], timeout: float = 30, max_tokens: int = 500):
        self.providers = providers
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_provider: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        errors = []
        for p in self.providers:
            started = False
            try:
                for delta in self._stream_one(p, messages):
                    if not started:
                        started = True
                        self.last_provider = p.name
                    yield delta
                if started:
                    return
                errors.append(f"{p.name}: empty response")
            except (httpx.HTTPError, ValueError) as e:
                if started:  # can't fail over mid-answer; surface the cut-off
                    log.warning("stream from %s broke mid-answer: %s", p.name, e)
                    return
                log.warning("provider %s failed, trying next: %s", p.name, e)
                errors.append(f"{p.name}: {e}")
        raise LLMUnavailable("; ".join(errors) or "no LLM providers configured")

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))

    def _stream_one(self, p: Provider, messages: list[dict]) -> Iterator[str]:
        payload = {"model": p.model, "messages": messages, "stream": True,
                   "temperature": 0.2, "max_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {p.api_key}"}
        with httpx.stream("POST", f"{p.base_url}/chat/completions", json=payload,
                          headers=headers, timeout=self.timeout) as r:
            if r.status_code != 200:
                r.read()
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:200]}",
                                            request=r.request, response=r)
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    yield delta


class FakeLLM:
    """Offline stand-in for tests and local UI work: echoes the top source."""

    available = True
    last_provider = "fake"

    def stream(self, messages: list[dict]) -> Iterator[str]:
        ctx = messages[-1]["content"]
        first = ctx.split("[1]", 1)[-1].split("\n[", 1)[0].strip()
        for word in f"(offline demo answer) {first[:300]} [1]".split(" "):
            yield word + " "

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))
