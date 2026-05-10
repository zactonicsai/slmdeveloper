#!/usr/bin/env sh
set -eu

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
MODEL_NAME="${OLLAMA_MODEL:-java-dto-assistant}"
CREATE_JSON="${CREATE_JSON:-ollama/create-java-dto-assistant.json}"

echo "Checking Ollama at ${OLLAMA_BASE_URL}..."
until curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null; do
  echo "Waiting for Ollama..."
  sleep 2
done

echo "Creating Ollama model: ${MODEL_NAME}"
curl -fsS "${OLLAMA_BASE_URL}/api/create" \
  -H "Content-Type: application/json" \
  -d @"${CREATE_JSON}"

echo
echo "Installed models:"
curl -fsS "${OLLAMA_BASE_URL}/api/tags"
echo
