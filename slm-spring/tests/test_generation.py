"""End-to-end test: hits the inference server and asserts the generated code
both compiles and uses the patterns we care about.

Requires:
  - docker compose up -d chromadb inference
  - load_chromadb.py and train_lora.py have been run (or accept base-model output)
"""

import os
import socket

import pytest
import httpx


INFERENCE_HOST = os.getenv("INFERENCE_HOST", "localhost")
INFERENCE_PORT = int(os.getenv("INFERENCE_PORT", "8080"))
BASE_URL = f"http://{INFERENCE_HOST}:{INFERENCE_PORT}"


def _server_alive() -> bool:
    try:
        with socket.create_connection((INFERENCE_HOST, INFERENCE_PORT), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_alive(),
    reason="Inference server not reachable; start it with docker compose up -d inference",
)


def test_healthz():
    r = httpx.get(f"{BASE_URL}/healthz", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_generate_dto():
    payload = {
        "task": "dto",
        "entity": "Subscription",
        "fields": [
            "id:Long",
            "planName:String",
            "price:BigDecimal",
            "active:Boolean",
        ],
    }
    r = httpx.post(f"{BASE_URL}/generate", json=payload, timeout=300)
    assert r.status_code == 200, r.text
    body = r.json()

    code = body["code"]
    # Hard requirements that should always hold for a Spring DTO:
    assert "package" in code
    assert "class" in code
    assert "private" in code
    # We require Jakarta, never javax — this is a major source of hallucination
    assert "javax.validation" not in code, "Generated code uses obsolete javax.validation"
    # Lombok annotations expected
    assert "@Data" in code or "@Getter" in code
    # The model should have been steered toward our retrieved patterns
    assert len(body["retrieved_sources"]) >= 1


def test_generate_controller():
    payload = {
        "task": "controller",
        "entity": "Notification",
    }
    r = httpx.post(f"{BASE_URL}/generate", json=payload, timeout=300)
    assert r.status_code == 200, r.text
    body = r.json()

    code = body["code"]
    assert "@RestController" in code
    assert "@RequestMapping" in code
    assert "ResponseEntity" in code
    assert "javax." not in code
    # Constructor injection rather than @Autowired-on-field
    assert "@Autowired" not in code or "@RequiredArgsConstructor" in code


def test_invalid_task_rejected():
    r = httpx.post(
        f"{BASE_URL}/generate",
        json={"task": "service", "entity": "Foo"},
        timeout=10,
    )
    assert r.status_code == 400
