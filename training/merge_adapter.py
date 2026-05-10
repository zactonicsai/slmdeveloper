"""
Merge the LoRA adapter into the base model and save a standalone HF model.

This is the prerequisite for GGUF conversion via llama.cpp's
`convert_hf_to_gguf.py`, which only accepts a full model, not a PEFT adapter.

After this script, the build pipeline runs llama.cpp conversion + quantization
to produce a single .gguf file that Ollama can ingest.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
ADAPTER_DIR = Path(os.environ.get("ADAPTER_DIR", "/workspace/output/lora-adapter"))
MERGED_DIR = Path(os.environ.get("MERGED_DIR", "/workspace/output/merged"))


def main() -> None:
    if not ADAPTER_DIR.exists():
        raise SystemExit(f"adapter not found: {ADAPTER_DIR}")

    print(f"[merge] base   : {BASE_MODEL}")
    print(f"[merge] adapter: {ADAPTER_DIR}")
    print(f"[merge] output : {MERGED_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    print("[merge] merging adapter into base weights ...")
    merged = peft_model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print("[merge] done.")


if __name__ == "__main__":
    main()
