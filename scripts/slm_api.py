import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
ADAPTER_DIR = os.getenv("OUTPUT_DIR", "/workspace/outputs/java-dto-lora")

app = FastAPI(title="Java DTO Custom SLM API", version="1.0.0")

model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=220, ge=1, le=1024)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

@app.on_event("startup")
def load_model():
    global model, tokenizer

    adapter_path = Path(ADAPTER_DIR)
    if not adapter_path.exists():
        print(f"Adapter not found at {ADAPTER_DIR}. Run scripts/train_lora.py first.")
        return

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

    if torch.cuda.is_available():
        model.cuda()

@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "adapter_not_loaded",
        "model_name": MODEL_NAME,
        "adapter_dir": ADAPTER_DIR,
        "cuda": torch.cuda.is_available(),
    }

@app.post("/generate")
def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail=f"LoRA adapter not loaded. Train first with: make train. Expected adapter at {ADAPTER_DIR}",
        )

    prompt = f"### Instruction:\n{req.prompt}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    return {
        "model": MODEL_NAME,
        "adapter": ADAPTER_DIR,
        "response": text,
    }
