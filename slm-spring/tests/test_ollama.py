"""Ollama backend integration tests.

These run only when an Ollama daemon is reachable AND the spring-coder model
is registered. CI should run the full pipeline (train → merge → export → register)
before invoking these.
"""

import os
import socket

import pytest

from src.ollama_client import OllamaClient


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "spring-coder")


def _ollama_alive() -> bool:
    try:
        with socket.create_connection((OLLAMA_HOST, OLLAMA_PORT), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_alive(),
    reason="Ollama not reachable; start it with docker compose up -d ollama",
)


@pytest.fixture(scope="module")
def client():
    return OllamaClient()


def test_list_models(client):
    models = client.list_models()
    assert isinstance(models, list)


def test_target_model_registered(client):
    if not client.healthy():
        pytest.skip(
            f"Model '{OLLAMA_MODEL}' not yet registered. "
            f"Run scripts/register_ollama_model.py after exporting the GGUF."
        )
    assert client.healthy()


def test_basic_generation(client):
    if not client.healthy():
        pytest.skip("model not registered")
    out = client.generate(
        prompt="Generate a one-line Java comment that says hello.",
        options={"num_predict": 32, "temperature": 0.0},
    )
    assert isinstance(out, str)
    assert len(out) > 0


def test_uses_jakarta_not_javax(client):
    """The strongest signal that fine-tuning + Modelfile system prompt are working."""
    if not client.healthy():
        pytest.skip("model not registered")
    out = client.generate(
        prompt=(
            "Generate a Java 21 Spring Boot 3.5.13 DTO class for entity 'Tag' "
            "with field name (String, required, max 50). Use Lombok and Jakarta Validation."
        ),
        options={"num_predict": 400, "temperature": 0.0},
    )
    assert "javax.validation" not in out, \
        f"Generated code uses obsolete javax.validation:\n{out}"
    assert "jakarta.validation" in out or "@NotBlank" in out
