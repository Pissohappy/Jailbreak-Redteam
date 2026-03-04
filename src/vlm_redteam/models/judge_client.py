"""Judge model client placeholder."""

import httpx


class JudgeClient:
    """Minimal HTTP client wrapper for external judge service."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30)
