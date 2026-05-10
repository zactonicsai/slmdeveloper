"""Ollama HTTP client. Used by the inference server when INFERENCE_BACKEND=ollama."""

import os
from typing import Dict, Optional

import httpx


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
    ):
        host = host or os.getenv("OLLAMA_HOST", "localhost")
        port = port or int(os.getenv("OLLAMA_PORT", "11434"))
        self.base_url = f"http://{host}:{port}"
        self.model = model or os.getenv("OLLAMA_MODEL", "spring-coder")
        self._client = httpx.Client(timeout=timeout)

    def healthy(self) -> bool:
        """Check liveness AND that our target model is registered."""
        try:
            r = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return any(self.model in n for n in names)
        except httpx.HTTPError:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict] = None,
    ) -> str:
        """Non-streaming generation. Returns the response string."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options

        r = self._client.post(f"{self.base_url}/api/generate", json=payload)
        r.raise_for_status()
        body = r.json()
        return body.get("response", "")

    def list_models(self) -> list:
        r = self._client.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        return r.json().get("models", [])
