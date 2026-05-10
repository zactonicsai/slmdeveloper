import argparse
import json
import os
import sys
import time

import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "java-dto-assistant")

def wait_for_ollama(base_url: str, model: str, timeout_seconds: int = 180):
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            if r.ok:
                models = [m.get("name", "") for m in r.json().get("models", [])]
                if any(name == model or name.startswith(model + ":") for name in models):
                    return
                print(f"Ollama is up, but model '{model}' is not ready yet. Available models: {models}")
            else:
                print(f"Ollama returned HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as exc:
            print(f"Waiting for Ollama at {base_url}: {exc}")

        time.sleep(5)

    raise RuntimeError(
        f"Ollama/model not ready after {timeout_seconds}s. "
        f"Check logs with: docker compose logs -f ollama ollama-setup"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--wait", action="store_true", default=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    try:
        wait_for_ollama(OLLAMA_BASE_URL, args.model, args.timeout)

        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "stream": False,
            "options": {
                "temperature": args.temperature,
                "top_p": 0.9,
            },
        }

        resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        print(data.get("response", json.dumps(data, indent=2)))

    except requests.exceptions.ConnectionError as exc:
        print(f"Could not connect to Ollama at {OLLAMA_BASE_URL}.", file=sys.stderr)
        print("Fix options:", file=sys.stderr)
        print("  1. Start services: docker compose up --build -d", file=sys.stderr)
        print("  2. Check status: docker compose ps", file=sys.stderr)
        print("  3. Check logs: docker compose logs -f ollama ollama-setup", file=sys.stderr)
        print("  4. If running from host, use OLLAMA_BASE_URL=http://localhost:11434", file=sys.stderr)
        print("  5. If running inside Docker, use OLLAMA_BASE_URL=http://ollama:11434", file=sys.stderr)
        print(f"Original error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Ollama test failed: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
