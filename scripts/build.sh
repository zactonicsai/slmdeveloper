#!/usr/bin/env bash
# End-to-end build pipeline run inside the trainer container.
#
#   1. LoRA fine-tune SmolLM2-135M on the grocery dataset (CPU).
#   2. Merge the LoRA adapter back into the base model.
#   3. Convert the merged HF model to GGUF (F16 — small enough at 135M).
#   4. Stage GGUF + Modelfile in /workspace/output so the ollama-init service
#      can register the model with the running Ollama daemon.

set -euo pipefail

cd /workspace

OUT_DIR=/workspace/output
mkdir -p "$OUT_DIR"

echo "=========================================================="
echo " step 1/4: LoRA fine-tune (CPU)"
echo "=========================================================="
python training/train_lora.py

echo
echo "=========================================================="
echo " step 2/4: merge adapter into base"
echo "=========================================================="
python training/merge_adapter.py

echo
echo "=========================================================="
echo " step 3/4: convert merged HF model -> GGUF (f16)"
echo "=========================================================="
python /opt/llama.cpp/convert_hf_to_gguf.py "$OUT_DIR/merged" \
    --outfile "$OUT_DIR/grocery-slm.gguf" \
    --outtype f16

echo
echo "=========================================================="
echo " step 4/4: stage Modelfile for ollama"
echo "=========================================================="
cp /workspace/training/Modelfile "$OUT_DIR/Modelfile"

ls -lh "$OUT_DIR"
echo
echo "build complete -> $OUT_DIR/grocery-slm.gguf"
