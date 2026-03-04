"""OpenAI-compatible HTTP client for local vLLM endpoints."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback for limited envs.
    httpx = None  # type: ignore[assignment]


class VLLMClient:
    """Client for `/v1/chat/completions` with optional vision input support."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: int = 60,
        enable_vision: bool = True,
        concurrency: int = 16,
        min_interval_sec: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.enable_vision = enable_vision
        self.concurrency = concurrency
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._rate_limit_lock = asyncio.Lock()
        self._last_request_ts = 0.0

    async def _throttle(self) -> None:
        if self.min_interval_sec <= 0:
            return
        async with self._rate_limit_lock:
            now = asyncio.get_running_loop().time()
            wait_for = self.min_interval_sec - (now - self._last_request_ts)
            if wait_for > 0:
                await asyncio.sleep(wait_for)
                now = asyncio.get_running_loop().time()
            self._last_request_ts = now

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _image_to_data_url(image_path: str | Path) -> str:
        path = Path(image_path)
        raw = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type not in {"image/png", "image/jpeg"}:
            mime_type = "image/png"
        b64 = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def build_messages(self, user_text: str, image_path: str | None) -> list[dict[str, Any]]:
        """Construct OpenAI-compatible messages payload for text-only or vision input."""

        if self.enable_vision and image_path and Path(image_path).exists():
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_to_data_url(image_path),
                            },
                        },
                    ],
                }
            ]
        return [{"role": "user", "content": user_text}]

    async def _request_with_retry(
        self,
        client: Any,
        payload: dict[str, Any],
        max_retries: int = 3,
    ) -> str:
        delay = 0.5
        for attempt in range(max_retries + 1):
            try:
                await self._throttle()
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"]
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retryable_status = status_code == 429 or (status_code is not None and status_code >= 500)
                request_err = httpx is not None and isinstance(exc, httpx.RequestError)
                if attempt < max_retries and (retryable_status or request_err):
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")

    async def _agenerate_one(
        self,
        user_text: str,
        image_path: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.base_url:
            return f"[dummy vllm output] {user_text[:64]}"
        if httpx is None:
            raise RuntimeError("httpx is required when vLLM base_url is set")

        payload = {
            "model": self.model,
            "messages": self.build_messages(user_text=user_text, image_path=image_path),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._request_with_retry(client=client, payload=payload)

    def generate_one(
        self,
        user_text: str,
        image_path: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate one model response."""

        return asyncio.run(
            self._agenerate_one(
                user_text=user_text,
                image_path=image_path,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    async def _agenerate_batch(self, list_of_inputs: list[dict[str, Any]]) -> list[str]:
        if not self.base_url:
            return [f"[dummy vllm output] {item['user_text'][:64]}" for item in list_of_inputs]
        if httpx is None:
            raise RuntimeError("httpx is required when vLLM base_url is set")

        sem = asyncio.Semaphore(self.concurrency)
        results: list[str | None] = [None] * len(list_of_inputs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:

            async def _run_one(idx: int, item: dict[str, Any]) -> None:
                async with sem:
                    payload = {
                        "model": self.model,
                        "messages": self.build_messages(
                            user_text=item["user_text"],
                            image_path=item.get("image_path"),
                        ),
                        "temperature": item["temperature"],
                        "max_tokens": item["max_tokens"],
                    }
                    results[idx] = await self._request_with_retry(client=client, payload=payload)

            await asyncio.gather(*[_run_one(i, inp) for i, inp in enumerate(list_of_inputs)])

        return [r if r is not None else "" for r in results]

    def generate_batch(self, list_of_inputs: list[dict[str, Any]]) -> list[str]:
        """Batch generation with async concurrency control and retries."""

        return asyncio.run(self._agenerate_batch(list_of_inputs))

    def ping(self) -> dict[str, Any]:
        """Ping OpenAI-compatible endpoint via `/v1/models`."""

        if not self.base_url:
            return {"data": [{"id": "dummy-model", "object": "model"}]}
        if httpx is None:
            raise RuntimeError("httpx is required when vLLM base_url is set")

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/v1/models", headers=self._headers())
            resp.raise_for_status()
            return resp.json()
