#!/usr/bin/env bash
# 30-second smoke test you can run after `docker compose up -d`.
# Verifies: ollama up, model registered, model produces a reply, UI serves.

set -euo pipefail

OLLAMA="${OLLAMA_URL:-http://localhost:11434}"
WEBUI="${WEBUI_URL:-http://localhost:8080}"
MODEL="${MODEL_NAME:-grocery-slm}"

pass() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
fail() { printf '  \033[31m✗\033[0m %s\n' "$1"; exit 1; }

echo "smoke test → ollama=$OLLAMA, webui=$WEBUI, model=$MODEL"

# 1. ollama responds
curl -fsS "$OLLAMA/api/tags" >/dev/null || fail "ollama /api/tags unreachable"
pass "ollama /api/tags is reachable"

# 2. model registered
if curl -fsS "$OLLAMA/api/tags" | grep -q "\"$MODEL"; then
  pass "model '$MODEL' is registered"
else
  fail "model '$MODEL' not found in ollama tags"
fi

# 3. inference works (ask a known-good question)
RESP=$(curl -fsS "$OLLAMA/api/chat" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"How do I pick a ripe avocado?\"}],\"options\":{\"temperature\":0.2,\"num_predict\":120}}")

CONTENT=$(printf '%s' "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["message"]["content"])')
[ ${#CONTENT} -gt 20 ] || fail "reply too short: $CONTENT"
pass "model returned a reply (${#CONTENT} chars)"
echo "    └─ ${CONTENT:0:120}..."

# 4. webui serves
curl -fsS "$WEBUI/" | grep -qi "grocerygpt" || fail "webui did not serve index"
pass "webui serves index.html"

curl -fsS "$WEBUI/api/tags" >/dev/null || fail "webui /api proxy broken"
pass "webui proxies /api → ollama"

echo
echo "all smoke checks passed."
