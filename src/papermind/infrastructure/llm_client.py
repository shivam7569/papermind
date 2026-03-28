"""Unified LLM client supporting both local (transformers) and Ollama backends."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx

from papermind.config import get_settings


class LLMClient:
    """Unified LLM client.

    Routes to either the local transformers model (NF4 quantized) or
    Ollama's HTTP API based on settings.llm.backend.

    Usage:
        client = LLMClient()
        # Async API (works with both backends)
        response = await client.generate("Write a function")
        response = await client.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(self, backend: str | None = None):
        settings = get_settings()
        self.backend = backend or settings.llm.backend
        self.base_url = settings.llm.base_url.rstrip("/")
        self.model = settings.llm.model
        self.timeout = settings.llm.timeout
        self._local_model: "LocalModel | None" = None

    def _get_local_model(self):
        """Lazy-load the local model on first use."""
        if self._local_model is None:
            from papermind.infrastructure.local_model import LocalModel
            self._local_model = LocalModel()
            self._local_model.load()
        return self._local_model

    async def generate(self, prompt: str, system: str = "") -> str:
        """Generate a completion from a prompt."""
        if self.backend == "local":
            local = self._get_local_model()
            return await asyncio.to_thread(local.generate, prompt, system)
        return await self._ollama_generate(prompt, system)

    async def chat(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> str:
        """Send a chat completion request."""
        if self.backend == "local":
            local = self._get_local_model()
            return await asyncio.to_thread(local.chat, messages, system)
        return await self._ollama_chat(messages, system)

    async def generate_stream(
        self, prompt: str, system: str = ""
    ) -> AsyncIterator[str]:
        """Stream a generation token by token."""
        if self.backend == "local":
            local = self._get_local_model()
            for token in local.generate_stream(prompt, system):
                yield token
            return
        async for token in self._ollama_generate_stream(prompt, system):
            yield token

    async def is_available(self) -> bool:
        """Check if the backend is ready."""
        if self.backend == "local":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    # --- Ollama HTTP backend ---

    async def _ollama_generate(self, prompt: str, system: str = "") -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]

    async def _ollama_chat(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": all_messages, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def _ollama_generate_stream(
        self, prompt: str, system: str = ""
    ) -> AsyncIterator[str]:
        import json as json_mod

        payload: dict = {"model": self.model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        data = json_mod.loads(line)
                        if "response" in data:
                            yield data["response"]
