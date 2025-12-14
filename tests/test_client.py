import json
from pathlib import Path

import httpx
import pytest

from cudara_client import CudaraClient, CudaraError, GenerationOptions, Message


def _make_client(handler):
    transport = httpx.MockTransport(handler)
    c = CudaraClient("http://test")
    c._client = httpx.Client(transport=transport)
    return c


def test_list_models_parsing():
    def handler(request: httpx.Request):
        assert request.method == "GET"
        assert str(request.url) == "http://test/api/tags"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "Qwen/Qwen2.5-3B-Instruct",
                        "status": "ready",
                        "description": "Test model",
                        "details": {"format": "transformers", "family": "AutoModelForCausalLM"},
                    }
                ]
            },
        )

    c = _make_client(handler)
    try:
        models = c.list_models()
        assert len(models) == 1
        assert models[0].name == "Qwen/Qwen2.5-3B-Instruct"
        assert models[0].status == "ready"
        # NOTE: your client currently stores details.format into ModelInfo.task
        assert models[0].task == "transformers"
    finally:
        c.close()


def test_generate_sends_expected_payload_and_parses_response():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "http://test/api/generate"
        payload = json.loads(request.content.decode())
        assert payload["model"] == "m"
        assert payload["prompt"] == "Hello"
        assert payload["stream"] is False
        assert payload["system"] == "You are helpful"
        assert payload["options"]["temperature"] == 0.5
        return httpx.Response(
            200,
            json={
                "model": "m",
                "response": "Hi!",
                "done": True,
                "total_duration": 123,
                "eval_count": 7,
            },
        )

    c = _make_client(handler)
    try:
        resp = c.generate(
            "m",
            "Hello",
            system="You are helpful",
            options=GenerationOptions(temperature=0.5),
        )
        assert resp.content == "Hi!"
        assert resp.model == "m"
        assert resp.eval_count == 7
        assert resp.total_duration_ns == 123
    finally:
        c.close()


def test_chat_accepts_string_and_parses_message_content():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "http://test/api/chat"
        payload = json.loads(request.content.decode())
        assert payload["model"] == "m"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "Hello back"},
                "done": True,
                "total_duration": 999,
                "eval_count": 3,
            },
        )

    c = _make_client(handler)
    try:
        resp = c.chat("m", "Hello")
        assert resp.content == "Hello back"
        assert resp.eval_count == 3
    finally:
        c.close()


def test_chat_accepts_Message_objects():
    def handler(request: httpx.Request):
        payload = json.loads(request.content.decode())
        assert payload["messages"] == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        return httpx.Response(
            200,
            json={"model": "m", "message": {"role": "assistant", "content": "OK"}, "done": True},
        )

    c = _make_client(handler)
    try:
        resp = c.chat(
            "m",
            [Message(role="system", content="You are helpful"), Message(role="user", content="Hi")],
        )
        assert resp.content == "OK"
    finally:
        c.close()


def test_embed_wraps_single_string_into_list():
    def handler(request: httpx.Request):
        assert request.method == "POST"
        assert str(request.url) == "http://test/api/embeddings"
        payload = json.loads(request.content.decode())
        assert payload["model"] == "e"
        assert payload["input"] == ["hello"]
        return httpx.Response(
            200, json={"model": "e", "embeddings": [[0.1, 0.2]], "total_duration": 10}
        )

    c = _make_client(handler)
    try:
        out = c.embed("e", "hello")
        assert out.model == "e"
        assert out.embedding == [0.1, 0.2]
    finally:
        c.close()


def test_request_error_uses_server_error_message():
    def handler(request: httpx.Request):
        return httpx.Response(
            400, json={"error": {"code": "invalid_request", "message": "Bad input"}}
        )

    c = _make_client(handler)
    try:
        with pytest.raises(CudaraError) as e:
            c.generate("m", "x")
        assert "Bad input" in str(e.value)
        assert "400" in str(e.value)
    finally:
        c.close()


def test_transcribe_does_not_force_json_content_type(monkeypatch, tmp_path: Path):
    # create fake audio file
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....WAVEfmt ")  # minimal bytes; server isn't called

    captured = {}

    class DummyResp:
        def __init__(self):
            self.status_code = 200

        def json(self):
            return {"model": "asr", "text": "ok", "total_duration": 1}

        @property
        def text(self):
            return "ok"

    c = CudaraClient("http://test")

    def fake_post(url, *, files=None, data=None, headers=None, **kwargs):
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        captured["headers"] = headers or {}
        return DummyResp()

    monkeypatch.setattr(c._client, "post", fake_post)

    out = c.transcribe("asr", str(audio), language="en")

    assert out.text == "ok"
    assert captured["url"].endswith("/api/transcribe")
    assert "file" in captured["files"]
    assert captured["data"]["model"] == "asr"
    # must NOT be forced to application/json for multipart
    assert "content-type" not in {k.lower(): v for k, v in captured["headers"].items()}
