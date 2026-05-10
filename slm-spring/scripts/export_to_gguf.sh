#!/usr/bin/env bash
# Convert the merged HF model to a quantized GGUF that Ollama can serve.
#
# Pipeline:
#   1. (optional) merge LoRA into base via scripts/merge_lora.py
#   2. Run llama.cpp's convert_hf_to_gguf.py -> fp16 GGUF
#   3. Run llama-quantize -> Q4_K_M GGUF (configurable)
#
# Run from inside the trainer container:
#   docker compose run --rm trainer bash scripts/export_to_gguf.sh

set -euo pipefail

MERGED_DIR="${MERGED_DIR:-./adapters/merged}"
GGUF_DIR="${GGUF_DIR:-./gguf}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-/opt/llama.cpp}"
QUANT="${QUANT:-Q4_K_M}"
MODEL_NAME="${OLLAMA_MODEL:-spring-coder}"

mkdir -p "$GGUF_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "ERROR: llama.cpp not found at $LLAMA_CPP_DIR. Are you inside the trainer container?"
  exit 1
fi

# Step 1 — merge if needed
if [ ! -d "$MERGED_DIR" ] || [ -z "$(ls -A "$MERGED_DIR" 2>/dev/null)" ]; then
  echo ">>> Step 1/3 — merging LoRA adapter into base"
  python scripts/merge_lora.py
else
  echo ">>> Step 1/3 — using existing merged model at $MERGED_DIR"
fi

# Step 2 — HF -> GGUF (fp16)
F16_GGUF="$GGUF_DIR/${MODEL_NAME}.f16.gguf"
echo ">>> Step 2/3 — converting to GGUF (fp16) -> $F16_GGUF"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$F16_GGUF" \
  --outtype f16

# Step 3 — quantize
QUANT_GGUF="$GGUF_DIR/${MODEL_NAME}.${QUANT}.gguf"
echo ">>> Step 3/3 — quantizing to $QUANT -> $QUANT_GGUF"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$F16_GGUF" "$QUANT_GGUF" "$QUANT"

# Optional: drop the fp16 to save disk
if [ "${KEEP_F16:-false}" != "true" ]; then
  echo "Removing intermediate fp16 GGUF (set KEEP_F16=true to retain)"
  rm -f "$F16_GGUF"
fi

echo
echo "GGUF artifacts:"
ls -lh "$GGUF_DIR"
echo
echo "Next step: bash scripts/register_ollama_model.sh"
