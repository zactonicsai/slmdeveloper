import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
ADAPTER_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs/java-dto-lora")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    if not Path(ADAPTER_DIR).exists():
        raise SystemExit(f"Adapter not found at {ADAPTER_DIR}. Run: python scripts/train_lora.py")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    prompt = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
