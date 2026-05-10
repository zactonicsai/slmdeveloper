# GroceryGPT — a fine-tuned SLM for fruits & vegetables

A small-language-model demo end-to-end: HuggingFace LoRA fine-tune → merged model → GGUF → Ollama → Docker Compose → HTML chat UI → tests. **Runs entirely on CPU.**

---

## What's in here

```
grocery-slm/
├── docker-compose.yml          # 4 services: trainer, ollama, ollama-init, webui
├── data/
│   └── dataset.jsonl           # ~60 grocery Q&A pairs in OpenAI chat format
├── training/
│   ├── Dockerfile              # CPU torch + transformers + peft + llama.cpp converter
│   ├── requirements.txt
│   ├── train_lora.py           # HuggingFace + PEFT LoRA fine-tune (CPU)
│   ├── merge_adapter.py        # merge LoRA back into base for GGUF export
│   └── Modelfile               # Ollama Modelfile (template + system prompt)
├── webui/
│   ├── index.html              # single-page chat, NDJSON streaming
│   └── nginx.conf              # static + proxies /api/* to ollama
├── tests/
│   ├── test_model.py           # pytest: registration + behavioural checks
│   └── requirements.txt
├── scripts/
│   ├── build.sh                # train → merge → GGUF (runs inside trainer)
│   └── smoke.sh                # 30-sec curl-based health check
└── output/                     # produced by the trainer (GGUF + Modelfile)
```

## Architecture

```
┌────────────────────┐    one-shot     ┌──────────────────────┐
│      trainer       │ ──────────────▶ │  output/             │
│  (HF LoRA + GGUF)  │                 │   grocery-slm.gguf   │
└────────────────────┘                 │   Modelfile          │
                                       └──────────┬───────────┘
                                                  │ mounted ro
                       ┌──────────────────┐       │
   browser  ◀───/───── │  webui (nginx)   │       │
   :8080              │                  │       ▼
                       │   /api/*  ──────▶  ollama:11434  ◀── ollama-init
                       └──────────────────┘    (CPU)         (registers model)
```

1. **trainer** (one-shot, profile=`train`): LoRA fine-tunes `SmolLM2-135M-Instruct` on `data/dataset.jsonl`, merges the adapter, and converts the merged HF model to a `f16` GGUF. Outputs land in `./output/`.
2. **ollama**: standard `ollama/ollama:latest` daemon, CPU-only, model storage in a named volume.
3. **ollama-init**: depends on a healthy `ollama`, runs `ollama create grocery-slm -f Modelfile` once, then exits.
4. **webui**: nginx serving the single-page chat UI and proxying `/api/*` to the ollama container, sidestepping CORS and giving the browser a single origin.

### Why these choices

| Choice | Rationale |
|---|---|
| `SmolLM2-135M-Instruct` | Smallest decent modern instruct model. Llama-arch → llama.cpp/Ollama compatible. Trains in minutes on CPU. |
| LoRA r=8 on attn projections only | Tiny memory footprint, fast on CPU, sufficient for stylistic/topical adaptation on ~60 examples. |
| fp32 (not bf16/fp16) | Most reliable across CPU vendors. We're tiny enough that memory isn't the bottleneck. |
| f16 GGUF (no quant) | At 135M params the f16 file is ~270 MB. Quantization adds little and complicates the build. |
| nginx proxy in front of ollama | One origin for the browser, avoids `OLLAMA_ORIGINS` configuration, cleanly supports streaming NDJSON. |
| `ollama-init` as a separate sidecar | Idempotent, declarative, completes-then-exits — composes cleanly via `service_completed_successfully`. |

---

## Running it

### Prerequisites

- Docker + Docker Compose v2
- ~2 GB free disk (HF cache + GGUF + Ollama model)
- A few GB RAM during training
- ~5–15 minutes on a modern multi-core CPU for the train step

### Step 1 — Train and export the model (one-shot)

```bash
docker compose --profile train run --rm trainer
```

This runs `scripts/build.sh` inside the trainer container:
1. `train_lora.py` — LoRA fine-tune (CPU, ~5–15 min)
2. `merge_adapter.py` — merge LoRA into base
3. `convert_hf_to_gguf.py` — export to `output/grocery-slm.gguf`
4. Stage `Modelfile` next to it

When it's done you'll have:

```
output/
├── grocery-slm.gguf   (~270 MB)
└── Modelfile
```

### Step 2 — Bring up the demo

```bash
docker compose up -d
```

This starts `ollama`, `ollama-init` (registers `grocery-slm`), and `webui`. Watch them come online:

```bash
docker compose logs -f ollama-init   # should print "[init] done." then exit
```

### Step 3 — Use it

Open http://localhost:8080 and ask about avocados, herb storage, watermelon ripeness, etc.

---

## Testing

### Quick smoke test (no Python deps)

```bash
./scripts/smoke.sh
```

Hits ollama directly + the webui proxy, asks the model one question, prints a snippet.

### Full pytest suite

```bash
pip install -r tests/requirements.txt
pytest tests/ -v
```

Tests covered:

| Test | What it checks |
|---|---|
| `test_ollama_is_reachable` | `/api/tags` returns 200 |
| `test_grocery_model_is_registered` | `grocery-slm` appears in tags |
| `test_model_metadata_via_show` | `/api/show` returns metadata containing the system prompt |
| `test_chat_returns_nonempty_response` | `/api/chat` produces > 20 chars |
| `test_responses_are_topical` (parametrised) | Replies to known fruit/veg questions contain plausible domain terms |
| `test_persona_holds_on_offtopic_question` | Asking for Python code doesn't get Python code |
| `test_webui_serves_index` | nginx serves the SPA |
| `test_webui_proxies_ollama_api` | nginx `/api/*` proxy works end-to-end |

Override the targets if you've remapped ports or are running remotely:

```bash
OLLAMA_URL=http://hostname:11434 WEBUI_URL=http://hostname:8080 pytest tests/ -v
```

---

## Iterating on the dataset

1. Edit `data/dataset.jsonl` (one JSON per line, OpenAI chat-completions schema)
2. `docker compose --profile train run --rm trainer` — retrains and re-exports
3. `docker compose restart ollama-init` — re-registers the new GGUF
   - The `ollama-init` script skips if the tag already exists; bump the model name in `Modelfile` and `index.html` (`MODEL` const) if you want side-by-side versions.

## Common knobs

Set in your shell before invoking `docker compose ... trainer`:

```bash
EPOCHS=5 LR=1e-4 OMP_NUM_THREADS=8 docker compose --profile train run --rm trainer
```

The trainer reads these env vars from `train_lora.py` (`EPOCHS`, `LR`, `BASE_MODEL`, `MAX_LEN`).

## Troubleshooting

**Training is very slow** — Set `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to your CPU's physical core count. Avoid SMT/hyperthread counts, they often hurt throughput here.

**`ollama-init` exits with "grocery-slm.gguf not found"** — You haven't run the train profile yet. Run step 1.

**Browser shows "model unreachable"** — Check `docker compose ps`. If `ollama-init` is still running, wait 10–30s for the warm-up to complete.

**HuggingFace download is rate-limited** — The `hf-cache` named volume persists model weights across runs; only the first train pays the download cost.

**Model gives bland or off-topic answers** — Expected at 135M parameters. Bump `EPOCHS`, expand the dataset, or swap `BASE_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct` for noticeably better results (3× the training time).

## Cleaning up

```bash
docker compose down -v          # stops everything + removes named volumes
rm -rf output/                  # delete the built GGUF
```
