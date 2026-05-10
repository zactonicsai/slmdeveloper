"""FastAPI inference server with pluggable backend.

Backends:
  - ollama  (default)  — calls a running Ollama daemon over HTTP
  - hf                 — runs the LoRA-tuned model in-process via transformers

Pipeline per request (same regardless of backend):
  1. Build instruction from request.
  2. Retrieve top-k canonical examples from ChromaDB.
  3. Generate Java with the chosen backend.
  4. Validate with javac. Retry on failure (up to MAX_RETRIES).
  5. Return code + validation report.
"""

import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .prompts import (
    SYSTEM_INSTRUCTION,
    build_inference_prompt,
)
from .rag import Retriever
from .validator import validate_java


load_dotenv()

INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "ollama").lower()
BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/phi-2")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "./adapters")) / "final"

GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.9"))
GEN_MAX_NEW_TOKENS = int(os.getenv("GEN_MAX_NEW_TOKENS", "768"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_RETRIES = 3


app = FastAPI(title="SLM Spring Generator", version="2.0.0")

_backend = None       # callable: (instruction, examples) -> str
_retriever = None
_backend_name = None


# =============================================================================
# Backend factories
# =============================================================================

def _make_ollama_backend():
    """Returns (callable, name). Calls Ollama via HTTP."""
    from .ollama_client import OllamaClient

    client = OllamaClient()
    options = {
        "temperature": GEN_TEMPERATURE,
        "top_p": GEN_TOP_P,
        "num_predict": GEN_MAX_NEW_TOKENS,
    }

    def generate(instruction: str, retrieved_examples: List[dict]) -> str:
        # The Modelfile already wraps the prompt in our trained TEMPLATE
        # and applies the SYSTEM message. So we send only the user content,
        # which is the instruction plus the RAG context.
        context_block = "\n".join(
            f"--- Reference example ({ex['metadata'].get('artifact_type', 'unknown')}: "
            f"{ex['metadata'].get('entity', '?')}) ---\n{ex['content']}"
            for ex in retrieved_examples
        )
        user_prompt = (
            f"### Reference patterns (treat as ground truth for style and imports):\n"
            f"{context_block}\n\n"
            f"{instruction}"
        )
        text = client.generate(user_prompt, options=options)
        return _trim(text)

    return generate, f"ollama:{client.model}"


def _make_hf_backend():
    """Returns (callable, name). Loads the base model + LoRA adapter in-process."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf backend] loading {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if ADAPTER_DIR.exists():
        print(f"[hf backend] applying LoRA from {ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    else:
        print(f"[hf backend] WARNING: no adapter at {ADAPTER_DIR}; using base model")
        model = base
    model.eval()

    def generate(instruction: str, retrieved_examples: List[dict]) -> str:
        prompt = build_inference_prompt(instruction, retrieved_examples)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=GEN_MAX_NEW_TOKENS,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                do_sample=GEN_TEMPERATURE > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):]
        return _trim(decoded)

    return generate, f"hf:{BASE_MODEL}"


def _load():
    global _backend, _retriever, _backend_name
    if _backend is not None:
        return

    _retriever = Retriever()

    if INFERENCE_BACKEND == "ollama":
        _backend, _backend_name = _make_ollama_backend()
    elif INFERENCE_BACKEND == "hf":
        _backend, _backend_name = _make_hf_backend()
    else:
        raise RuntimeError(f"Unknown INFERENCE_BACKEND={INFERENCE_BACKEND!r}; expected 'ollama' or 'hf'")
    print(f"[inference] backend ready: {_backend_name}")


# =============================================================================
# Output trimming
# =============================================================================

def _trim(text: str) -> str:
    """Stop at our prompt markers; trim to the last top-level closing brace."""
    for stop in ["\n### Instruction:", "\n### System:", "\n### Reference"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    matches = list(re.finditer(r"^\}\s*$", text, flags=re.MULTILINE))
    if matches:
        text = text[: matches[-1].end()]
    return text.strip()


# =============================================================================
# Request / response
# =============================================================================

class GenerateRequest(BaseModel):
    task: str = Field(..., description="'controller' or 'dto'")
    entity: str = Field(..., description="Entity name, e.g. 'Order'")
    fields: Optional[List[str]] = Field(
        default=None,
        description="DTO fields, e.g. ['id:Long', 'total:BigDecimal']",
    )
    base_path: Optional[str] = Field(default=None, description="Override request mapping path")
    notes: Optional[str] = Field(default=None, description="Free-form additional requirements")


class GenerateResponse(BaseModel):
    code: str
    classname: Optional[str]
    valid: bool
    syntax_ok: bool
    errors: list
    warnings: list
    attempts: int
    backend: str
    retrieved_sources: List[str]


def _build_instruction(req: GenerateRequest) -> str:
    if req.task == "dto":
        fields = ", ".join(req.fields or ["id:Long"])
        instr = (
            f"Generate a Java 21 Spring Boot 3.5.13 DTO class for entity "
            f"'{req.entity}' with fields: {fields}. Use Lombok and Jakarta Validation. "
            f"Pick reasonable validation constraints based on each field's type and name."
        )
    elif req.task == "controller":
        path = req.base_path or f"/api/v1/{req.entity.lower()}s"
        instr = (
            f"Generate a Java 21 Spring Boot 3.5.13 REST controller for entity "
            f"'{req.entity}' with full CRUD endpoints under {path}. Use constructor "
            f"injection via Lombok @RequiredArgsConstructor, ResponseEntity for all "
            f"return types, and @Valid on request bodies."
        )
    else:
        raise HTTPException(400, f"Unsupported task '{req.task}', must be 'controller' or 'dto'")

    if req.notes:
        instr += f" Additional requirements: {req.notes}"
    return instr


# =============================================================================
# Routes
# =============================================================================

@app.get("/healthz")
def healthz():
    backend_ok = True
    if INFERENCE_BACKEND == "ollama":
        # Check Ollama without forcing a model load
        from .ollama_client import OllamaClient
        backend_ok = OllamaClient().healthy()
    return {
        "status": "ok",
        "backend": INFERENCE_BACKEND,
        "backend_healthy": backend_ok,
        "base_model": BASE_MODEL,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    _load()
    instruction = _build_instruction(req)

    examples = _retriever.retrieve(
        query=instruction,
        top_k=RAG_TOP_K,
        artifact_type=req.task,
    )
    sources = [ex["metadata"].get("source", "?") for ex in examples]

    last_result = None
    code = ""
    for attempt in range(1, MAX_RETRIES + 1):
        code = _backend(instruction, examples)
        result = validate_java(code)
        last_result = result
        if result.ok:
            return GenerateResponse(
                code=code,
                classname=result.classname,
                valid=True,
                syntax_ok=result.syntax_ok,
                errors=[],
                warnings=result.warnings,
                attempts=attempt,
                backend=_backend_name,
                retrieved_sources=sources,
            )

    return GenerateResponse(
        code=code,
        classname=last_result.classname if last_result else None,
        valid=False,
        syntax_ok=last_result.syntax_ok if last_result else False,
        errors=last_result.errors if last_result else ["Unknown failure"],
        warnings=last_result.warnings if last_result else [],
        attempts=MAX_RETRIES,
        backend=_backend_name or INFERENCE_BACKEND,
        retrieved_sources=sources,
    )
