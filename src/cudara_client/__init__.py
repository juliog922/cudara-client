"""
cudara_client
=============

A small, batteries-included Python client for the **Cudara inference server**.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import (
    Any,
    Literal,
    overload,
)

import httpx

__version__ = "1.0.0"

__all__ = [
    "CudaraClient",
    "CudaraError",
    "connect",
    "Message",
    "GenerationOptions",
    "Response",
    "EmbeddingResponse",
    "TranscriptionResponse",
    "ModelInfo",
    "PromptTemplate",
    "PromptBuilder",
    "OutputParser",
    "ModelType",
    "__version__",
]

JSON = dict[str, Any]
Headers = dict[str, str]


class ModelType(str, Enum):
    """Logical model families supported by Cudara."""

    CHAT = "text-generation"
    VISION = "image-to-text"
    EMBEDDING = "feature-extraction"
    ASR = "automatic-speech-recognition"


@dataclass
class Message:
    """
    A chat message compatible with Cudara `/api/chat`.
    """

    role: Literal["system", "user", "assistant"]
    content: str
    images: list[str] | None = None

    def to_dict(self) -> JSON:
        """Convert this message into a JSON-serializable dict."""
        d: JSON = {"role": self.role, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d


@dataclass
class GenerationOptions:
    """
    Text generation parameters.
    """

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: list[str] | None = None

    def to_dict(self) -> JSON:
        """Convert to the server-side `options` dictionary."""
        opts: JSON = {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.stop:
            opts["stop"] = self.stop
        return opts


@dataclass
class Response:
    """
    Standardized wrapper for chat/generate responses.
    """

    content: str
    model: str
    done: bool = True
    total_duration_ns: int = 0
    eval_count: int = 0
    raw: JSON | None = None

    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        return self.total_duration_ns / 1_000_000

    @property
    def tokens_per_second(self) -> float:
        """Estimated tokens/second based on eval_count and total duration."""
        if self.total_duration_ns > 0:
            return self.eval_count / (self.total_duration_ns / 1_000_000_000)
        return 0.0


@dataclass
class EmbeddingResponse:
    """
    Wrapper for embedding responses.
    """

    embeddings: list[list[float]]
    model: str
    total_duration_ns: int = 0

    @property
    def embedding(self) -> list[float]:
        """Convenience: returns the first embedding vector (or empty list)."""
        return self.embeddings[0] if self.embeddings else []


@dataclass
class TranscriptionResponse:
    """
    Wrapper for ASR responses.
    """

    text: str
    model: str
    total_duration_ns: int = 0


@dataclass
class ModelInfo:
    """
    Model information returned by `/api/tags`.
    """

    name: str
    status: str
    description: str
    task: str
    details: JSON = field(default_factory=dict)


class PromptTemplate:
    """Common system prompt templates for convenience."""

    CONCISE = "Answer concisely and directly. Be specific but brief."
    DETAILED = "Provide a comprehensive and detailed explanation."
    STEP_BY_STEP = "Think step by step and explain your reasoning."
    CODE_EXPERT = "You are an expert programmer. Write clean, well-documented code."
    JSON_OUTPUT = (
        "Respond ONLY with valid JSON. No explanations or markdown.\nFormat: {format_spec}"
    )
    MARKDOWN_OUTPUT = "Format your response in clean Markdown with headers, code blocks, and lists."
    OCR = "Extract all text from this image exactly as shown. Output as plain text."
    DESCRIBE = "Describe this image in detail, including objects, colors, and composition."
    ANALYZE = "Analyze this image and provide insights about its content and context."
    CHAIN_OF_THOUGHT = (
        "Let's approach this step by step:\n"
        "1. First, understand the problem\n"
        "2. Break it down into parts\n"
        "3. Solve each part\n"
        "4. Combine for final answer\n\n"
        "Problem: {problem}"
    )


class PromptBuilder:
    """
    Small helper to build a (system, user) pair using context/instructions/examples.
    """

    def __init__(self) -> None:
        self._system: str | None = None
        self._context: list[str] = []
        self._instructions: list[str] = []
        self._examples: list[tuple[str, str]] = []
        self._format: str | None = None

    def system(self, prompt: str) -> PromptBuilder:
        """Set the system prompt."""
        self._system = prompt
        return self

    def context(self, text: str) -> PromptBuilder:
        """Append context text."""
        self._context.append(text)
        return self

    def instruction(self, text: str) -> PromptBuilder:
        """Append a user instruction line."""
        self._instructions.append(text)
        return self

    def example(self, input_text: str, output_text: str) -> PromptBuilder:
        """Append an input/output example pair."""
        self._examples.append((input_text, output_text))
        return self

    def format_json(self, schema: Mapping[str, Any]) -> PromptBuilder:
        """Request a JSON response matching the provided schema."""
        self._format = (
            f"Respond with valid JSON matching this schema:\n{json.dumps(dict(schema), indent=2)}"
        )
        return self

    def format_list(self) -> PromptBuilder:
        """Request a numbered list response."""
        self._format = "Respond with a numbered list."
        return self

    def build(self, user_input: str) -> tuple[str | None, str]:
        """
        Build the final (system, user) strings.
        """
        system_parts: list[str] = []
        if self._system:
            system_parts.append(self._system)
        if self._format:
            system_parts.append(self._format)
        system = "\n\n".join(system_parts) if system_parts else None

        user_parts: list[str] = []
        if self._context:
            user_parts.append("Context:\n" + "\n".join(self._context))
        if self._examples:
            examples = [f"Input: {inp}\nOutput: {out}" for inp, out in self._examples]
            user_parts.append("Examples:\n" + "\n\n".join(examples))
        if self._instructions:
            user_parts.append("Instructions:\n" + "\n".join(f"- {i}" for i in self._instructions))
        user_parts.append(user_input)

        return system, "\n\n".join(user_parts)

    def reset(self) -> PromptBuilder:
        """Clear all builder state so the instance can be reused."""
        self._system = None
        self._context = []
        self._instructions = []
        self._examples = []
        self._format = None
        return self


class OutputParser:
    """Helpers to parse structured outputs from model responses."""

    @staticmethod
    def json(response: str) -> JSON:
        """
        Parse JSON from a model response.
        """
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1 if lines[0].startswith("```") else 0
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end])
        return json.loads(text)

    @staticmethod
    def list_items(response: str) -> list[str]:
        """Extract bullet/numbered list items from a response."""
        lines = response.strip().split("\n")
        items: list[str] = []
        for line in lines:
            line = line.strip()
            for prefix in ["- ", "* ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix) :]
                    break
            if line and line[0].isdigit():
                for sep in [". ", ") ", ": "]:
                    idx = line.find(sep)
                    if idx != -1 and idx < 4:
                        line = line[idx + len(sep) :]
                        break
            if line:
                items.append(line)
        return items

    @staticmethod
    def code_blocks(response: str, language: str | None = None) -> list[str]:
        """
        Extract fenced code blocks.
        """
        blocks: list[str] = []
        in_block = False
        current_block: list[str] = []
        target_lang = language.lower() if language else None

        for line in response.split("\n"):
            if line.strip().startswith("```"):
                if in_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                    in_block = False
                else:
                    block_lang = line.strip()[3:].lower()
                    if target_lang is None or block_lang == target_lang or block_lang == "":
                        in_block = True
            elif in_block:
                current_block.append(line)
        return blocks


class CudaraError(Exception):
    """Raised when the server returns an error response (HTTP 4xx/5xx)."""


class CudaraClient:
    """
    Synchronous client for the Cudara inference server.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 300.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._headers: Headers = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        # Prevents race conditions from client side if reused across threads
        self._inference_lock = threading.Lock()

    def __enter__(self) -> CudaraClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._client.close()

    def _request(self, method: str, endpoint: str, **kwargs: Any) -> JSON:
        """
        Perform an HTTP request and return decoded JSON.
        """
        url = f"{self.base_url}{endpoint}"

        # Merge headers safely
        headers: Headers = dict(kwargs.get("headers") or {})
        headers.update(self._headers)
        kwargs["headers"] = headers

        response = self._client.request(method, url, **kwargs)
        if response.status_code >= 400:
            try:
                error = response.json()
                msg = error.get("error", {}).get("message", response.text)
            except Exception:
                msg = response.text
            raise CudaraError(f"API error ({response.status_code}): {msg}")
        return response.json()

    def list_models(self) -> list[ModelInfo]:
        """List available models from `/api/tags`."""
        # Non-inference method, no lock required.
        data = self._request("GET", "/api/tags")
        return [
            ModelInfo(
                name=m["name"],
                status=m.get("status", "unknown"),
                description=m.get("description", ""),
                task=m.get("details", {}).get("format", "unknown"),
                details=m.get("details", {}),
            )
            for m in data.get("models", [])
        ]

    def pull(self, model: str) -> JSON:
        """Trigger a model pull/download via `/api/pull`."""
        # Non-inference method, no lock required.
        return self._request("POST", "/api/pull", json={"name": model})

    def delete(self, model: str) -> JSON:
        """Delete a model via `/api/delete`."""
        # Non-inference method, no lock required.
        return self._request("DELETE", "/api/delete", json={"name": model})

    def show(self, model: str) -> JSON:
        """Show model details via `/api/show`."""
        # Non-inference method, no lock required.
        return self._request("POST", "/api/show", json={"name": model})

    def health(self) -> JSON:
        """Check server health via `/health`."""
        # Non-inference method, no lock required.
        return self._request("GET", "/health")

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        images: list[str] | None = None,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Text generation using `/api/generate`.
        Acquires client lock to enforce serial inference.
        """
        payload: JSON = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        opts: JSON = options.to_dict() if options else {}
        opts.update(kwargs)
        if opts:
            payload["options"] = opts

        with self._inference_lock:
            data = self._request("POST", "/api/generate", json=payload)

        return Response(
            content=data.get("response", ""),
            model=data.get("model", model),
            done=data.get("done", True),
            total_duration_ns=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            raw=data,
        )

    @overload
    def chat(
        self,
        model: str,
        messages: str,
        *,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @overload
    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @overload
    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response: ...

    def chat(
        self,
        model: str,
        messages: str | Sequence[Message] | Sequence[Mapping[str, Any]],
        *,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Chat completion using `/api/chat`.
        Acquires client lock to enforce serial inference.
        """
        if isinstance(messages, str):
            msgs: list[JSON] = [{"role": "user", "content": messages}]
        else:
            messages_list = list(messages)
            if messages_list and isinstance(messages_list[0], Message):
                msgs = [m.to_dict() for m in messages_list]  # type: ignore[arg-type]
            else:
                msgs = [dict(m) for m in messages_list]  # type: ignore[arg-type]

        payload: JSON = {"model": model, "messages": msgs, "stream": False}

        opts: JSON = options.to_dict() if options else {}
        opts.update(kwargs)
        if opts:
            payload["options"] = opts

        with self._inference_lock:
            data = self._request("POST", "/api/chat", json=payload)

        return Response(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", model),
            done=data.get("done", True),
            total_duration_ns=data.get("total_duration", 0),
            eval_count=data.get("eval_count", 0),
            raw=data,
        )

    def vision(
        self,
        model: str,
        prompt: str,
        *,
        image_path: str | None = None,
        image_bytes: bytes | None = None,
        image_base64: str | None = None,
        options: GenerationOptions | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Convenience wrapper for vision models.
        Locks handled by `self.generate` which this calls.
        """
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        elif image_bytes:
            img_b64 = base64.b64encode(image_bytes).decode()
        elif image_base64:
            img_b64 = image_base64
        else:
            raise ValueError("Must provide image_path, image_bytes, or image_base64")

        return self.generate(
            model=model, prompt=prompt, images=[img_b64], options=options, **kwargs
        )

    def ocr(self, model: str, image_path: str, *, options: GenerationOptions | None = None) -> str:
        """OCR helper using `PromptTemplate.OCR`. Locks handled by vision/generate."""
        response = self.vision(
            model=model, prompt=PromptTemplate.OCR, image_path=image_path, options=options
        )
        return response.content

    def embed(self, model: str, texts: str | Sequence[str], **kwargs: Any) -> EmbeddingResponse:
        """
        Create embeddings via `/api/embeddings`.
        Acquires client lock to enforce serial inference.
        """
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        payload: JSON = {"model": model, "input": texts_list}
        if kwargs:
            payload["options"] = dict(kwargs)

        with self._inference_lock:
            data = self._request("POST", "/api/embeddings", json=payload)

        return EmbeddingResponse(
            embeddings=data.get("embeddings", []),
            model=data.get("model", model),
            total_duration_ns=data.get("total_duration", 0),
        )

    def transcribe(self, model: str, audio_path: str, **kwargs: Any) -> TranscriptionResponse:
        """
        Transcribe audio via `/api/transcribe`.
        Acquires client lock to enforce serial inference.
        """
        with open(audio_path, "rb") as f:
            files = {"file": (Path(audio_path).name, f)}
            data: dict[str, Any] = {"model": model}
            if kwargs:
                data["options"] = json.dumps(kwargs)

            headers = {k: v for k, v in self._headers.items() if k.lower() != "content-type"}

            with self._inference_lock:
                response = self._client.post(
                    f"{self.base_url}/api/transcribe",
                    files=files,
                    data=data,
                    headers=headers,
                )

        if response.status_code >= 400:
            try:
                error = response.json()
                msg = error.get("error", {}).get("message", response.text)
            except Exception:
                msg = response.text
            raise CudaraError(f"API error ({response.status_code}): {msg}")

        result = response.json()
        return TranscriptionResponse(
            text=result.get("text", ""),
            model=result.get("model", model),
            total_duration_ns=result.get("total_duration", 0),
        )

    def prompt(self) -> PromptBuilder:
        """Return a new PromptBuilder instance."""
        return PromptBuilder()

    @staticmethod
    def parse() -> OutputParser:
        """Return the OutputParser utility class."""
        return OutputParser()


def connect(base_url: str = "http://localhost:8000", timeout: float = 300.0) -> CudaraClient:
    """Convenience constructor for `CudaraClient`."""
    return CudaraClient(base_url=base_url, timeout=timeout)


def main() -> None:
    """CLI entry point for the `cudara-client` console script."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="cudara-client", description="Cudara Client CLI")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--version", action="version", version=f"cudara-client {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List models")

    pull_p = subparsers.add_parser("pull", help="Pull model")
    pull_p.add_argument("model", help="Model ID")

    chat_p = subparsers.add_parser("chat", help="Chat")
    chat_p.add_argument("model", help="Model ID")
    chat_p.add_argument("prompt", help="Prompt")

    subparsers.add_parser("health", help="Health check")

    args = parser.parse_args()
    client = CudaraClient(args.url)

    try:
        if args.command == "list":
            for m in client.list_models():
                status = "✓" if m.status == "ready" else "○"
                print(f"{status} {m.name} - {m.description}")
        elif args.command == "pull":
            print(f"Pulling {args.model}...")
            print(client.pull(args.model))
        elif args.command == "chat":
            response = client.chat(args.model, args.prompt)
            print(response.content)
            print(f"\n[{response.eval_count} tokens, {response.duration_ms:.0f}ms]")
        elif args.command == "health":
            print(json.dumps(client.health(), indent=2))
        else:
            parser.print_help()
    except CudaraError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()