import json
import os
import sys
import time
from pathlib import Path

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
BASE_MODEL = os.getenv("OLLAMA_BASE_MODEL", "qwen2.5-coder:0.5b")
CUSTOM_MODEL = os.getenv("OLLAMA_MODEL", "java-dto-assistant")
CREATE_JSON = Path(os.getenv("CREATE_JSON", "ollama/create-java-dto-assistant.json"))

def wait_for_ollama():
    print(f"Waiting for Ollama at {OLLAMA_BASE_URL}...")
    for _ in range(120):
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.ok:
                print("Ollama is ready.")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("Timed out waiting for Ollama.")

def post_json(path, payload):
    r = requests.post(f"{OLLAMA_BASE_URL}{path}", json=payload, timeout=600)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}

def main():
    print("=== Java DTO Custom Ollama Model Builder ===")
    print(f"Ollama URL:   {OLLAMA_BASE_URL}")
    print(f"Base model:   {BASE_MODEL}")
    print(f"Custom model: {CUSTOM_MODEL}")
    print()

    wait_for_ollama()

    print(f"Pulling base model: {BASE_MODEL}")
    print(json.dumps(post_json("/api/pull", {"name": BASE_MODEL, "stream": False}), indent=2))

    print(f"Creating custom model: {CUSTOM_MODEL}")
    create_payload = json.loads(CREATE_JSON.read_text(encoding="utf-8"))
    create_payload["model"] = CUSTOM_MODEL
    create_payload["stream"] = False
    print(json.dumps(post_json("/api/create", create_payload), indent=2))

    tags = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=30).json()
    print("Installed models:")
    print(json.dumps(tags, indent=2))

    names = [m.get("name", "") for m in tags.get("models", [])]
    if not any(n == CUSTOM_MODEL or n.startswith(CUSTOM_MODEL + ":") for n in names):
        raise RuntimeError(f"Custom model was not found after create. Available: {names}")

    print("Testing custom model...")
    test = post_json("/api/generate", {
        "model": CUSTOM_MODEL,
        "prompt": "Create a Java DTO named OrderRequest with Lombok Builder and Jakarta Validation.",
        "stream": False,
    })
    print(test.get("response", json.dumps(test, indent=2)))
    print(f"\nCustom model build complete: {CUSTOM_MODEL}")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        print("Check: docker compose ps && docker compose logs -f ollama ollama-setup", file=sys.stderr)
        sys.exit(1)
