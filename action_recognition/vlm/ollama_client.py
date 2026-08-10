"""Small, dependency-light client for a local or remote Ollama VLM.

The server is normally a MacBook on the same trusted network. Configure it with
``OLLAMA_BASE_URL=http://<mac-ip>:11434`` and optionally
``OLLAMA_VLM_MODEL=qwen3-vl:8b-instruct``.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen3-vl:8b-instruct")


class OllamaConnectionError(RuntimeError):
    """The Ollama server could not complete a request."""


@dataclass(frozen=True)
class OllamaResponse:
    """Parsed structured output plus Ollama timing metadata."""

    data: dict[str, Any]
    seconds: float
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ValueError("Ollama base URL cannot be empty")
    if not re.match(r"^https?://", value, re.I):
        value = f"http://{value}"
    return value


def encode_image(image: Any, jpeg_quality: int = 88) -> str:
    """Return an image as base64 accepted by Ollama's REST API."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        raw = bytes(image)
    elif isinstance(image, (str, Path)):
        raw = Path(image).read_bytes()
    else:
        try:
            import cv2

            ok, buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
        except Exception as exc:  # pragma: no cover - error message is the API
            raise TypeError(f"Unsupported image value: {type(image)!r}") from exc
        if not ok:
            raise ValueError("JPEG encoding failed")
        raw = buffer.tobytes()
    return base64.b64encode(raw).decode("ascii")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a markdown fence from older models."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"VLM returned no JSON object: {text[:200]!r}")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM returned invalid JSON: {text[:200]!r}") from exc
    if not isinstance(value, dict):
        raise ValueError("VLM JSON response must be an object")
    return value


class OllamaVLMClient:
    """Synchronous Ollama client with retry and structured-output support."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        timeout: float = 300.0,
        retries: int = 2,
        keep_alive: str = "30m",
        # Ollama defaults to 4096. That is not enough here: five 512 px frames
        # cost ~3300 prompt tokens, and qwen3-vl is a *thinking* model that
        # ignores ``think: false`` over the API — it spends 800+ tokens
        # reasoning before it answers. At 4096 the answer never arrives:
        # generation stops on ``done_reason: "length"`` mid-thought and
        # ``content`` comes back empty, which looks exactly like a model that
        # cannot do the task. At 8192 the same request answers correctly.
        num_ctx: int = 8192,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url or DEFAULT_BASE_URL)
        self.model = model or DEFAULT_MODEL
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.keep_alive = keep_alive
        self.num_ctx = int(num_ctx)
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                if response.status_code >= 500 or response.status_code == 429:
                    raise requests.HTTPError(
                        f"Ollama returned HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2**attempt, 4))
        hint = (
            f"Cannot reach Ollama at {self.base_url}. Check the Mac IP, firewall, "
            "and OLLAMA_HOST=0.0.0.0:11434."
        )
        raise OllamaConnectionError(hint) from last_error

    def version(self) -> str:
        data = self._request("GET", "/api/version").json()
        return str(data.get("version", "unknown"))

    def models(self) -> list[str]:
        data = self._request("GET", "/api/tags").json()
        return [
            str(item.get("name"))
            for item in data.get("models", [])
            if item.get("name")
        ]

    def check(self, *, require_model: bool = True) -> dict[str, Any]:
        models = self.models()
        available = self.model in models or any(
            name.split(":", 1)[0] == self.model for name in models
        )
        if require_model and not available:
            raise OllamaConnectionError(
                f"Ollama is reachable, but model {self.model!r} is not installed. "
                f"Installed: {', '.join(models) or '(none)'}"
            )
        return {
            "base_url": self.base_url,
            "version": self.version(),
            "model": self.model,
            "model_available": available,
            "models": models,
        }

    def chat_json(
        self,
        prompt: str,
        images: Sequence[Any],
        schema: Mapping[str, Any],
        *,
        temperature: float = 0.0,
        jpeg_quality: int = 88,
    ) -> OllamaResponse:
        """Ask the VLM for output constrained by ``schema``.

        Grammar-constrained decoding is not reliable across vision models: with
        ollama 0.32.5, ``qwen3-vl:8b`` answers a schema request with seven tokens
        and an empty string, every time, while the identical request without
        ``format`` answers correctly. Rather than lose the model, the schema is
        treated as a preference: if constrained decoding returns nothing usable,
        the same prompt is retried unconstrained and the JSON object is pulled
        out of whatever came back.

        An empty ``content`` is otherwise indistinguishable from a refusal, so
        note the other cause: a thinking model that ran out of context mid-thought
        (see ``num_ctx``). Both are handled here rather than surfaced as "the VLM
        cannot do this", which is the wrong conclusion to draw from either.
        """
        encoded = [encode_image(image, jpeg_quality=jpeg_quality) for image in images]

        def build(constrained: bool) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": encoded,
                    }
                ],
                "stream": False,
                "think": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": float(temperature),
                    "seed": 0,
                    "num_ctx": self.num_ctx,
                },
            }
            if constrained:
                payload["format"] = dict(schema)
            return payload

        started = time.perf_counter()
        response = self._request("POST", "/api/chat", json=build(True))
        body = response.json()
        content = body.get("message", {}).get("content", "")
        if not content.strip():
            response = self._request("POST", "/api/chat", json=build(False))
            body = response.json()
            content = body.get("message", {}).get("content", "")
        elapsed = time.perf_counter() - started
        return OllamaResponse(
            data=parse_json_object(content),
            seconds=elapsed,
            total_duration_ns=int(body.get("total_duration", 0) or 0),
            load_duration_ns=int(body.get("load_duration", 0) or 0),
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=int(body.get("eval_count", 0) or 0),
        )
