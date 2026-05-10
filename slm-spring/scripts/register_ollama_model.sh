#!/usr/bin/env bash
# One-shot: GGUF export + Ollama registration. Run from trainer container.
set -euo pipefail
bash scripts/export_to_gguf.sh
python scripts/register_ollama_model.py
