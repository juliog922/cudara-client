# cudara-client

[![PyPI](https://img.shields.io/pypi/v/cudara-client?color=blue)](https://pypi.org/project/cudara-client/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/juliog922/cudara-client/actions/workflows/publish.yml/badge.svg)](https://github.com/juliog922/cudara-client/actions)

Python client library for the **Cudara** inference server (Ollama-compatible API).

This package provides a small, synchronous `httpx`-based client for:

- Chat (`/api/chat`)
- Text generation (`/api/generate`)
- Embeddings (`/api/embeddings`)
- Vision helpers (send images via `/api/generate`)
- Audio transcription (`/api/transcribe`)
- Prompt building + output parsing utilities

---

## Install

### With uv

```bash
uv add cudara-client
```

### With pip

```bash
pip install cudara-client
```

---

## Run the Cudara API (server)

You need a running Cudara server to use this client.

### Option A: Docker (recommended)

```bash
# Pull and run
docker run --gpus all -p 8000:8000 ghcr.io/juliog922/cudara:latest

# With persistent models
docker run --gpus all -p 8000:8000 \
  -v cudara_models:/app/models \
  ghcr.io/juliog922/cudara:latest
```

### Option B: uv (development)

```bash
git clone https://github.com/juliog922/cudara
cd cudara
uv sync

# Run server
uv run cudara serve
```

Health check:

```bash
curl http://localhost:8000/health
```

---

## Quickstart

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    # Optionally pull/download a model (if enabled on your server)
    client.pull("Qwen/Qwen2.5-3B-Instruct")

    resp = client.chat("Qwen/Qwen2.5-3B-Instruct", "Hello!")
    print(resp.content)
```

---

## Chat

```python
from cudara_client import CudaraClient, Message, GenerationOptions

with CudaraClient("http://localhost:8000") as client:
    resp = client.chat(
        "Qwen/Qwen2.5-3B-Instruct",
        [
            Message(role="system", content="You are concise."),
            Message(role="user", content="Explain embeddings in 2 bullets."),
        ],
        options=GenerationOptions(temperature=0.2, max_tokens=128),
    )
    print(resp.content)
```

---

## Text generation

```python
from cudara_client import CudaraClient, GenerationOptions

with CudaraClient("http://localhost:8000") as client:
    resp = client.generate(
        "Qwen/Qwen2.5-3B-Instruct",
        "Write a haiku about GPUs.",
        system="You are a poet.",
        options=GenerationOptions(max_tokens=64, temperature=0.8),
    )
    print(resp.content)
```

---

## Embeddings

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    out = client.embed("sentence-transformers/all-MiniLM-L6-v2", ["hello", "world"])
    print(len(out.embeddings), len(out.embeddings[0]))
```

Single string convenience:

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    out = client.embed("sentence-transformers/all-MiniLM-L6-v2", "hello")
    vec = out.embedding
    print(len(vec))
```

---

## Vision (image-to-text)

If your server is running a vision-language model, you can send images.
`vision()` reads an image and sends it as base64 via `/api/generate`.

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    resp = client.vision(
        "your-vision-model",
        "Describe the image.",
        image_path="cat.jpg",
    )
    print(resp.content)
```

### OCR helper

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    text = client.ocr("your-vision-model", "receipt.png")
    print(text)
```

---

## Transcription (ASR)

Cudara supports multipart transcription at `/api/transcribe`.

```python
from cudara_client import CudaraClient

with CudaraClient("http://localhost:8000") as client:
    out = client.transcribe("openai/whisper-small", "audio.wav", language="en")
    print(out.text)
```

---

## Error handling

Any 4xx/5xx response raises `CudaraError`.

```python
from cudara_client import CudaraClient, CudaraError

try:
    with CudaraClient("http://localhost:8000") as client:
        client.generate("unknown-model", "hello")
except CudaraError as e:
    print("Request failed:", e)
```

---

## Development

```bash
uv sync --dev --locked
uv run pytest -v
```

---

## License

MIT
