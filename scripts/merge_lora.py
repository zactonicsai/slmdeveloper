import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
ADAPTER_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs/java-dto-lora")
MERGED_DIR = os.getenv("MERGED_DIR", "/workspace/outputs/java-dto-merged-hf")

def main():
    if not Path(ADAPTER_DIR).exists():
        raise SystemExit(f"Adapter not found at {ADAPTER_DIR}. Run: make train")

    Path(MERGED_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from adapter: {ADAPTER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading base model: {MODEL_NAME}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    print(f"Loading adapter: {ADAPTER_DIR}")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    print("Merging LoRA adapter into base model...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged Hugging Face model to: {MERGED_DIR}")
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    print("""
Merged model saved.

To run in Ollama:
1. Convert this merged Hugging Face model to GGUF with llama.cpp.
2. Put the GGUF file in ./models.
3. Edit ollama/Modelfile.custom-gguf-template to point to it.
4. Run:
   ollama create java-dto-finetuned -f ollama/Modelfile.custom-gguf-template
""")

if __name__ == "__main__":
    main()
