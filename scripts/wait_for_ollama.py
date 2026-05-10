import os
import sys
import time
import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "java-dto-assistant")
TIMEOUT_SECONDS = int(os.getenv("OLLAMA_WAIT_TIMEOUT", "300"))

def main():
    deadline = time.time() + TIMEOUT_SECONDS
    print(f"Waiting for Ollama at {BASE_URL}...")

    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
            if r.ok:
                data = r.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                print(f"Ollama is reachable. Models: {models}")

                if any(name == MODEL or name.startswith(MODEL + ":") for name in models):
                    print(f"Model is ready: {MODEL}")
                    return

                print(f"Ollama is running, but model '{MODEL}' is not ready yet.")
                print("The ollama-setup container may still be pulling/creating the model.")
        except requests.RequestException as exc:
            print(f"Not ready yet: {exc}")

        time.sleep(5)

    print(f"Timed out waiting for Ollama model '{MODEL}' at {BASE_URL}.", file=sys.stderr)
    print("Try: docker compose logs -f ollama ollama-setup", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
