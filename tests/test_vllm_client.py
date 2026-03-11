import asyncio
import base64
from pathlib import Path

import httpx

from vlm_redteam.models.vllm_client import VLLMClient


def test_build_messages_vision_and_text_fallback(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n")

    client = VLLMClient(base_url="http://localhost:8000", model="m", enable_vision=True)
    vision_messages = client.build_messages("hello", str(img))
    content = vision_messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["text"] == "hello"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    encoded = content[1]["image_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(encoded) == b"\x89PNG\r\n"

    text_only_messages = client.build_messages("hello", None)
    assert text_only_messages == [{"role": "user", "content": "hello"}]


def test_request_with_retry_for_5xx() -> None:
    calls = {"n": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, json={"error": "server busy"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = VLLMClient(base_url="http://localhost:8000", model="m", enable_vision=False)

    async def run() -> str:
        async with httpx.AsyncClient(transport=transport, timeout=10) as async_client:
            return await client._request_with_retry(
                client=async_client,
                payload={"model": "m", "messages": [{"role": "user", "content": "x"}]},
            )

    result = asyncio.run(run())
    assert result == "ok"
    assert calls["n"] == 2


def test_generate_batch_keeps_order(monkeypatch) -> None:
    client = VLLMClient(base_url="http://localhost:8000", model="m", enable_vision=False)

    async def fake_request_with_retry(*, client, payload, max_retries=3):
        return payload["messages"][0]["content"] + "-resp"

    monkeypatch.setattr(client, "_request_with_retry", fake_request_with_retry)

    outputs = client.generate_batch(
        [
            {"user_text": "a", "image_path": None, "temperature": 0.1, "max_tokens": 8},
            {"user_text": "b", "image_path": None, "temperature": 0.1, "max_tokens": 8},
        ]
    )
    assert outputs == ["a-resp", "b-resp"]


def test_generate_one_with_messages_uses_provided_messages(monkeypatch) -> None:
    client = VLLMClient(base_url="http://localhost:8000", model="m", enable_vision=False)

    async def fake_request_with_retry(*, client, payload, max_retries=3):
        return payload["messages"][0]["content"] + "-" + payload["messages"][1]["content"]

    monkeypatch.setattr(client, "_request_with_retry", fake_request_with_retry)

    output = client.generate_one_with_messages(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        temperature=0.1,
        max_tokens=16,
    )
    assert output == "sys-usr"
