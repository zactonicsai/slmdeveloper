# SLM for Java 21 + Spring Boot 3.5.13 Controller Generation

A small, focused language model that generates **Spring Boot 3.5.13 Controllers and DTOs** in Java 21, grounded in a curated ChromaDB of best-practice examples to limit hallucination.

## Stack
- **Python 3.11**, PyTorch 2.4+, Hugging Face Transformers, PEFT (LoRA), Datasets
- **Base model**: `microsoft/phi-2` (default) or `distilgpt2` (toggle in `.env`)
- **Vector store**: ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **Serving**: FastAPI inference server
- **Orchestration**: Docker Compose
- **Validation**: Java compiler check via `javac` (Java 21)

## Architecture

```
                     ┌─────────────────────────┐
   user prompt ─────▶│ FastAPI inference server│
                     └────────────┬────────────┘
                                  │ 1. embed prompt
                                  ▼
                     ┌─────────────────────────┐
                     │   ChromaDB (RAG)        │  ← canonical Spring/DTO snippets
                     │   top-k examples        │
                     └────────────┬────────────┘
                                  │ 2. build augmented prompt
                                  ▼
                     ┌─────────────────────────┐
                     │ LoRA-tuned SLM (Phi-2)  │
                     │ generate Java code      │
                     └────────────┬────────────┘
                                  │ 3. javac validation
                                  ▼
                              Java code
```

## Local Dev Environment — Step-by-Step

### Prerequisites
- Docker Desktop (with at least 8 GB RAM allocated; 16 GB recommended for Phi-2)
- NVIDIA GPU + CUDA drivers (optional but ~10× faster than CPU)
- Python 3.11+ if running outside Docker
- ~10 GB free disk for model weights and Docker images

### Step 1 — Clone & configure
```bash
cd slm-spring
cp .env.example .env
# Edit .env: set BASE_MODEL=microsoft/phi-2 (or distilgpt2 if low RAM)
```

### Step 2 — Build & start services
```bash
docker compose build
docker compose up -d chromadb
```
ChromaDB will be running on `http://localhost:8000`.

### Step 3 — Seed ChromaDB with canonical examples
```bash
docker compose run --rm trainer python src/load_chromadb.py
```
This loads everything in `examples/*.java` plus the JSONL snippets in `data/` into the vector store, with metadata (controller vs DTO, validation type, etc.).

### Step 4 — Fine-tune via LoRA
```bash
docker compose run --rm trainer python src/train_lora.py
```
Trains LoRA adapters on `data/dto_controller_dataset.jsonl`. Adapters land in `./adapters/`. With a single 8 GB GPU this takes ~20 min for distilgpt2, ~90 min for Phi-2.

### Step 5 — Run the inference server
```bash
docker compose up -d inference
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"task":"controller","entity":"Order","fields":["id:Long","total:BigDecimal","status:String"]}'
```

### Step 6 — Run the test suite
```bash
docker compose run --rm trainer pytest tests/ -v
```

## Why this limits hallucination

1. **Narrow scope.** The model only ever generates two artifact types (Controller, DTO). The prompt template is fixed.
2. **RAG grounding.** Every generation is conditioned on 3 retrieved canonical examples — so Lombok annotations, `jakarta.validation` imports, and `ResponseEntity` patterns come from real reference code, not the model's priors.
3. **Compile-time validation.** Each generation is written to a temp file and run through `javac`. Failed compilations are rejected and regenerated (up to 3 retries).
4. **Constrained decoding.** Temperature 0.2, top-p 0.9, with stop tokens at `}\n\n` to prevent runaway generation.

## Project layout

```
slm-spring/
├── docker-compose.yml
├── docker/
│   ├── trainer.Dockerfile        # Python + PyTorch + HF
│   └── inference.Dockerfile      # FastAPI + JDK 21 for validation
├── requirements.txt
├── .env.example
├── src/
│   ├── train_lora.py             # LoRA fine-tuning
│   ├── load_chromadb.py          # seed vector store
│   ├── inference_server.py       # FastAPI + RAG + generate
│   ├── rag.py                    # ChromaDB retrieval
│   ├── validator.py              # javac compilation check
│   └── prompts.py                # prompt templates
├── data/
│   └── dto_controller_dataset.jsonl
├── examples/                     # canonical Java reference code
│   ├── UserDTO.java
│   ├── UserController.java
│   ├── ProductDTO.java
│   ├── ProductController.java
│   ├── OrderDTO.java
│   └── OrderController.java
├── tests/
│   ├── test_rag.py
│   ├── test_validator.py
│   └── test_generation.py
└── adapters/                     # LoRA output (gitignored)
```