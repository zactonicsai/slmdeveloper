"""Register the GGUF model with Ollama by calling `ollama create` over HTTP.

This works without docker exec — it just hits the Ollama API. Run from the
trainer container (where it can reach the ollama service by hostname) or
from the host (where ollama is on localhost:11434).
"""

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv


load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
MODEL_NAME = os.getenv("OLLAMA_MODEL", "spring-coder")
MODELFILE_PATH = Path(os.getenv("MODELFILE_PATH", "./Modelfile"))

BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"


def wait_for_ollama(timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise RuntimeError(f"Ollama did not become reachable at {BASE_URL} within {timeout}s")


def main() -> None:
    if not MODELFILE_PATH.exists():
        print(f"ERROR: Modelfile not found at {MODELFILE_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Waiting for Ollama at {BASE_URL}")
    wait_for_ollama()

    modelfile_content = MODELFILE_PATH.read_text(encoding="utf-8")
    print(f"Registering model '{MODEL_NAME}'")
    print("=== Modelfile ===")
    print(modelfile_content)
    print("=================")

    # The /api/create endpoint streams progress as JSONL.
    with httpx.stream(
        "POST",
        f"{BASE_URL}/api/create",
        json={
            "name": MODEL_NAME,
            "modelfile": modelfile_content,
            "stream": True,
        },
        timeout=600,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                print(line)

    # Verify
    r = httpx.get(f"{BASE_URL}/api/tags", timeout=10)
    r.raise_for_status()
    names = [m["name"] for m in r.json().get("models", [])]
    if any(MODEL_NAME in n for n in names):
        print(f"\n✅ Model '{MODEL_NAME}' is registered with Ollama")
        print(f"   Try it: curl -X POST {BASE_URL}/api/generate -d '{{\"model\":\"{MODEL_NAME}\",\"prompt\":\"// hello\"}}'")
    else:
        print(f"\n⚠️  Model '{MODEL_NAME}' not found in /api/tags after create; got: {names}")
        sys.exit(2)


if __name__ == "__main__":
    main()
