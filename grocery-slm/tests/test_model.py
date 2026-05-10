"""
Integration tests for the GroceryGPT stack.

These tests assume `docker compose up -d` has already brought the stack up and
the model has been registered (via the trainer + ollama-init services).

Run from the project root:

    pip install -r tests/requirements.txt
    pytest tests/ -v

What's covered:
  * /api/tags lists `grocery-slm`
  * /api/show returns metadata for it
  * /api/chat produces a non-empty reply
  * Persona check: response to a fruit/veg question contains relevant terms
  * Off-topic redirect: responses to non-produce questions stay on theme
  * Webui serves index.html and proxies /api correctly
"""

from __future__ import annotations

import json
import os

import pytest
import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
WEBUI_URL  = os.environ.get("WEBUI_URL",  "http://localhost:8080")
MODEL_NAME = os.environ.get("MODEL_NAME", "grocery-slm")
TIMEOUT    = int(os.environ.get("TEST_TIMEOUT", 120))


# ---------- helpers ---------------------------------------------------------

def _chat(prompt: str, *, system: str | None = None) -> str:
    """Send a single-turn chat to Ollama, return the assistant text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 200},
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


# ---------- model presence --------------------------------------------------

def test_ollama_is_reachable():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    assert r.status_code == 200, f"unexpected status: {r.status_code}"
    assert "models" in r.json()


def test_grocery_model_is_registered():
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    names = [m["name"] for m in r.json().get("models", [])]
    assert any(n.startswith(MODEL_NAME) for n in names), (
        f"{MODEL_NAME} not found in tags: {names}"
    )


def test_model_metadata_via_show():
    r = requests.post(
        f"{OLLAMA_URL}/api/show",
        json={"name": MODEL_NAME},
        timeout=10,
    )
    assert r.status_code == 200
    data = r.json()
    # Modelfile lives under either `modelfile` or `model_info` depending on
    # Ollama version — check both.
    blob = json.dumps(data).lower()
    assert "grocerygpt" in blob or "grocery" in blob, (
        "system prompt / Modelfile metadata missing produce branding"
    )


# ---------- behavioural -----------------------------------------------------

def test_chat_returns_nonempty_response():
    reply = _chat("How do I pick a ripe avocado?")
    assert isinstance(reply, str)
    assert len(reply.strip()) > 20, f"reply too short: {reply!r}"


@pytest.mark.parametrize(
    "prompt,must_contain_any",
    [
        ("How do I pick a ripe avocado?",
         ["squeeze", "yield", "soft", "firm", "stem", "color", "dark"]),
        ("Should I refrigerate tomatoes?",
         ["counter", "ripe", "fridge", "refrigerat", "room", "cold", "flavor"]),
        ("How long do bananas last on the counter?",
         ["day", "ripe", "brown", "yellow"]),
    ],
)
def test_responses_are_topical(prompt, must_contain_any):
    reply = _chat(prompt).lower()
    hits = [w for w in must_contain_any if w in reply]
    assert hits, f"reply lacks any topical term {must_contain_any}: {reply!r}"


def test_persona_holds_on_offtopic_question():
    """
    The system prompt instructs the model to redirect off-topic queries back
    to produce. A 135M model won't always do this perfectly, but the response
    shouldn't go far afield — we just check it doesn't, e.g., write code.
    """
    reply = _chat("Write me a Python function to reverse a string.").lower()
    # Heuristic: model should not produce code blocks / def / return statements.
    code_smells = ["def ", "return ", "```python", "lambda "]
    leaks = [s for s in code_smells if s in reply]
    assert not leaks, f"model went off-topic and produced code-like text: {reply!r}"


# ---------- web ui ----------------------------------------------------------

def test_webui_serves_index():
    r = requests.get(WEBUI_URL, timeout=10)
    assert r.status_code == 200
    assert "GroceryGPT" in r.text or "grocery" in r.text.lower()


def test_webui_proxies_ollama_api():
    r = requests.get(f"{WEBUI_URL}/api/tags", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
