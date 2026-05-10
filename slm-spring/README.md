# SLM for Java 21 + Spring Boot 3.5.13 — served via Ollama

A small, focused language model that generates **Spring Boot 3.5.13 Controllers and DTOs** in Java 21, fine-tuned with LoRA, **converted to GGUF, and served via Ollama**, grounded with ChromaDB RAG to limit hallucination.

## Stack

- **Python 3.11**, PyTorch, Hugging Face Transformers, PEFT (LoRA), TRL
- **Base model**: `microsoft/phi-2` (default) or `distilgpt2`
- **Vector store**: ChromaDB + `all-MiniLM-L6-v2` embeddings
- **Quantization & serving**: llama.cpp → GGUF → **Ollama**
- **Inference API**: FastAPI in front of Ollama, with RAG and javac validation
- **Orchestration**: Docker Compose

## Architecture

```
                      ┌──────────────────────────┐
   user prompt ──────▶│ FastAPI inference server │
                      └────────┬─────────────────┘
                               │ 1. embed prompt
                               ▼
                      ┌──────────────────────────┐
                      │   ChromaDB (RAG)         │  ← canonical Spring/DTO snippets
                      └────────┬─────────────────┘
                               │ 2. build augmented prompt
                               ▼
                      ┌──────────────────────────┐
                      │ Ollama (spring-coder)    │  ← Modelfile wraps prompt template
                      │ GGUF Q4_K_M quantized    │
                      └────────┬─────────────────┘
                               │ 3. javac validation
                               ▼
                            Java code
```

The training pipeline is run once up-front:

```
HF base model (Phi-2)
       │
       │  + dataset (data/dto_controller_dataset.jsonl)
       ▼
LoRA fine-tune (PEFT)              ── adapters/final/
       │
       ▼
Merge adapter into base            ── adapters/merged/    (scripts/merge_lora.py)
       │
       ▼
Convert HF → GGUF fp16             ── gguf/spring-coder.f16.gguf
       │
       ▼
Quantize to Q4_K_M                 ── gguf/spring-coder.Q4_K_M.gguf
       │
       ▼
ollama create -f Modelfile         ── Ollama tag: spring-coder
```

## Local Dev Environment — step-by-step

### Prerequisites
- Docker Desktop (≥ 16 GB RAM allocated for Phi-2)
- NVIDIA GPU + CUDA drivers (optional, ~10× faster training; Ollama uses GPU automatically when present)
- ~15 GB free disk

### Step 1 — Configure
```bash
cd slm-spring
cp .env.example .env
# Toggle BASE_MODEL, INFERENCE_BACKEND, etc.
```

### Step 2 — Start ChromaDB and Ollama
```bash
docker compose up -d chromadb ollama
```

### Step 3 — Build the trainer image (includes JDK 21 + llama.cpp)
```bash
docker compose build trainer
```
First build is slow (~6 min) because it compiles `llama-quantize`. Subsequent builds are cached.

### Step 4 — Seed ChromaDB
```bash
docker compose run --rm trainer python src/load_chromadb.py
```

### Step 5 — Fine-tune via LoRA
```bash
docker compose run --rm trainer python src/train_lora.py
```
Adapters land in `./adapters/final/`. ~20 min on GPU for distilgpt2, ~90 min for Phi-2.

### Step 6 — Convert to GGUF and register with Ollama
```bash
docker compose run --rm trainer bash scripts/register_ollama_model.sh
```
This runs the full pipeline: merge LoRA → convert to GGUF → quantize Q4_K_M → POST to Ollama's `/api/create`.

Verify:
```bash
curl http://localhost:11434/api/tags
# Should list "spring-coder:latest"

curl -X POST http://localhost:11434/api/generate \
  -d '{"model":"spring-coder","prompt":"Generate a DTO for entity Book with title and author","stream":false}'
```

### Step 7 — Start the FastAPI inference server
```bash
docker compose up -d inference
curl http://localhost:8080/healthz
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"task":"controller","entity":"Order","fields":["id:Long","total:BigDecimal","status:String"]}'
```

### Step 8 — Run tests
```bash
docker compose run --rm trainer pytest tests/ -v
```

## How Ollama integration changes the flow

| Stage | Before | After |
|---|---|---|
| Training | LoRA fine-tune | unchanged |
| Adapter format | PEFT `adapter_model.safetensors` | unchanged |
| **Conversion** | n/a | **merge → GGUF → quantize** (`scripts/export_to_gguf.sh`) |
| **Registration** | n/a | **`ollama create -f Modelfile`** (`register_ollama_model.py`) |
| Serving | in-process HF + PEFT | **Ollama daemon, GPU-accelerated, with KV-cache reuse** |
| Prompt template | applied in Python | **applied via `TEMPLATE` block in Modelfile** |
| System prompt | string in `prompts.py` | **`SYSTEM` block in Modelfile** |
| Stop tokens | applied in Python | **`PARAMETER stop` in Modelfile** |
| Inference container memory | ~6 GB (model loaded) | ~200 MB (just FastAPI + Chroma client) |

You can still run the in-process HF backend by setting `INFERENCE_BACKEND=hf` in `.env` — useful for A/B comparisons.

## Anti-hallucination layers (now five)

1. **Tight prompt template** — fixed input shape; only two task types
2. **LoRA fine-tuned weights** — learned Spring/Lombok structure on curated examples
3. **RAG retrieval** — top-3 canonical examples injected into every prompt, filtered by `artifact_type`
4. **Modelfile-enforced system prompt** — Ollama applies it on every request, can't be bypassed by the user-facing API
5. **`javac` validation + retry loop** — generated code that doesn't compile is rejected; up to 3 retries

## Project layout

```
slm-spring/
├── docker-compose.yml
├── docker/
│   ├── trainer.Dockerfile        # Python + PyTorch + HF + JDK 21 + llama.cpp
│   └── inference.Dockerfile      # FastAPI + JDK 21
├── Modelfile                     # Ollama model definition
├── requirements.txt
├── .env.example
├── src/
│   ├── train_lora.py             # LoRA fine-tuning
│   ├── load_chromadb.py          # seed vector store
│   ├── inference_server.py       # FastAPI + RAG + ollama/hf backends
│   ├── ollama_client.py          # thin Ollama HTTP wrapper
│   ├── rag.py
│   ├── validator.py              # javac compilation check
│   └── prompts.py
├── scripts/
│   ├── merge_lora.py             # merge_and_unload + save HF model
│   ├── export_to_gguf.sh         # convert + quantize
│   ├── register_ollama_model.py  # POST /api/create
│   └── register_ollama_model.sh  # one-shot wrapper
├── data/
│   └── dto_controller_dataset.jsonl
├── examples/                     # canonical Java reference code
├── tests/
│   ├── test_validator.py
│   ├── test_rag.py
│   ├── test_generation.py        # FastAPI end-to-end
│   └── test_ollama.py            # direct Ollama API
├── adapters/                     # LoRA + merged HF model (gitignored)
└── gguf/                         # GGUF artifacts (gitignored)
```

## Troubleshooting

**"Model not found in /api/tags"**
The `ollama create` step failed. Check `docker compose logs ollama` for the FROM-path error. The GGUF must exist at `/gguf/spring-coder.Q4_K_M.gguf` *inside* the ollama container — ensure `./gguf` on the host is populated before running registration.

**"unsupported architecture" during convert_hf_to_gguf.py**
Update llama.cpp: rebuild the trainer image with `--no-cache` to pull the latest. Phi-2 and GPT-2 family are both supported in current master.

**Generated code uses `javax.validation`**
Three things to check, in order:
1. Modelfile `SYSTEM` is being applied (`curl /api/show -d '{"name":"spring-coder"}'`).
2. RAG retrieval returns examples (look at `retrieved_sources` in the response).
3. Training data uses jakarta — check `data/dto_controller_dataset.jsonl`.

**Ollama OOMs**
Drop to a smaller quant: set `QUANT=Q5_K_S` or `Q4_K_S` in `.env` and re-run step 6. Or reduce `num_ctx` in the Modelfile.
