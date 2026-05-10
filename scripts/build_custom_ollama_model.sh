#!/usr/bin/env sh
set -eu

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
BASE_MODEL="${OLLAMA_BASE_MODEL:-qwen2.5-coder:0.5b}"
CUSTOM_MODEL="${OLLAMA_MODEL:-java-dto-assistant}"
CREATE_JSON="${CREATE_JSON:-ollama/create-java-dto-assistant.json}"

echo "=== Java DTO Custom Ollama Model Builder ==="
echo "Ollama URL:    ${OLLAMA_BASE_URL}"
echo "Base model:    ${BASE_MODEL}"
echo "Custom model:  ${CUSTOM_MODEL}"
echo

echo "1) Waiting for Ollama..."
until curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null; do
  echo "   Ollama not ready yet..."
  sleep 2
done

echo "2) Pulling base model: ${BASE_MODEL}"
curl -fsS "${OLLAMA_BASE_URL}/api/pull" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${BASE_MODEL}\",\"stream\":false}" >/tmp/ollama-pull-response.json

cat /tmp/ollama-pull-response.json
echo

echo "3) Creating custom model: ${CUSTOM_MODEL}"
curl -fsS "${OLLAMA_BASE_URL}/api/create" \
  -H "Content-Type: application/json" \
  -d @"${CREATE_JSON}" >/tmp/ollama-create-response.json

cat /tmp/ollama-create-response.json
echo

echo "4) Installed models:"
curl -fsS "${OLLAMA_BASE_URL}/api/tags"
echo

echo "5) Test prompt:"
curl -fsS "${OLLAMA_BASE_URL}/api/generate" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${CUSTOM_MODEL}\",\"prompt\":\"Create a Java DTO named OrderRequest with Lombok Builder and Jakarta Validation.\",\"stream\":false}" \
  > /tmp/ollama-test-response.json

cat /tmp/ollama-test-response.json
echo
echo "Custom model build complete: ${CUSTOM_MODEL}"
