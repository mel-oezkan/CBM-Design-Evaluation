"""Thin LLM/VLM client for concept discovery, with an on-disk response cache."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..utils import JsonCache

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeClient:
    """Calls the Claude Messages API (``pip install cbm-eval[llm]``).

    Credentials come from the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
    Server-side refusal fallback is enabled so a declined request is retried on another model.
    """

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 4000,
                 cache_dir: str | Path | None = None):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.cache = JsonCache(cache_dir) if cache_dir else None

    def complete(self, prompt: str, images: list[str] | None = None, system: str | None = None) -> str:
        key = {"model": self.model, "effort": self.effort, "prompt": prompt, "images": images or [], "system": system}
        if self.cache and (hit := self.cache.get(key)) is not None:
            return hit

        content: list[dict] = []
        for path in images or []:
            media_type = mimetypes.guess_type(path)[0] or "image/jpeg"
            data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": prompt})

        kwargs = {"system": system} if system else {}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Concept discovery request was refused: {response.stop_details}")
        text = "".join(b.text for b in response.content if b.type == "text")
        if self.cache:
            self.cache.set(key, text)
        return text


def parse_list(text: str) -> list[str]:
    """Parse a bulleted / numbered / one-per-line list into clean items."""
    items = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•").strip()
        if line[:1].isdigit():
            line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
        line = line.strip(" .;,\"'`").lower()
        if line and len(line) < 80 and not line.endswith(":"):
            items.append(line)
    return items
