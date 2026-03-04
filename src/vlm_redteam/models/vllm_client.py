"""vLLM client placeholder."""

import httpx


class VLLMClient:
    """Minimal HTTP client wrapper for target vLLM."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=30)
