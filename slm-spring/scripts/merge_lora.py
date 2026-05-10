"""Merge the LoRA adapter into the base model weights.

GGUF doesn't support PEFT adapters at runtime — we have to bake them in.
After this runs, ./adapters/merged/ contains a standalone HF model that
llama.cpp's convert_hf_to_gguf.py can ingest.
"""

import os
import shutil
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/phi-2")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "./adapters")) / "final"
MERGED_DIR = Path(os.getenv("MERGED_DIR", "./adapters/merged"))


def main() -> None:
    if not ADAPTER_DIR.exists():
        print(f"ERROR: adapter not found at {ADAPTER_DIR}. Run train_lora.py first.", file=sys.stderr)
        sys.exit(1)

    if MERGED_DIR.exists():
        print(f"Removing previous merge at {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge on CPU in float16 to keep memory predictable. GGUF conversion
    # will requantize anyway, so precision here doesn't matter much.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Applying LoRA from {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))

    print("Merging LoRA into base weights (this is the irreversible step)")
    model = model.merge_and_unload()

    print(f"Saving merged model to {MERGED_DIR}")
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    # Sanity: report the on-disk size
    total = sum(f.stat().st_size for f in MERGED_DIR.rglob("*") if f.is_file())
    print(f"Merged model size: {total / (1024**3):.2f} GiB")
    print("Done. Next step: scripts/export_to_gguf.sh")


if __name__ == "__main__":
    main()
