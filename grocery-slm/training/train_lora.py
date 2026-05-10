"""
CPU LoRA fine-tuning for a small instruct model on grocery fruit/veggie Q&A.

Design choices for CPU:
  * Base model: HuggingFaceTB/SmolLM2-135M-Instruct (~135M params, Llama arch -> Ollama compatible)
  * No bitsandbytes / 4-bit quant (CUDA only). Train in fp32 (most reliable on CPU).
  * LoRA r=8, alpha=16 on attention projections. Tiny memory + fast on CPU.
  * Batch size 1 with gradient_accumulation_steps=8 -> effective batch 8.
  * 3 epochs is enough to teach style + content for this dataset size.

Expected wall-clock: ~5-15 min on a modern multi-core CPU for 65 examples * 3 epochs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ---------- config ----------------------------------------------------------

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DATASET_PATH = Path(os.environ.get("DATASET_PATH", "/workspace/data/dataset.jsonl"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output/lora-adapter"))
MAX_LEN = int(os.environ.get("MAX_LEN", 512))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-4))
SEED = 42

SYSTEM_PROMPT = (
    "You are GroceryGPT, a friendly assistant focused on grocery fruits and "
    "vegetables. You help with picking ripe produce, storage, seasonality, "
    "nutrition, substitutions, and quick prep tips. Stay concise, practical, "
    "and warm. If asked something unrelated to fruits or vegetables, gently "
    "redirect to produce topics."
)


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records in {path}")
    return records


def build_dataset(records: list[dict], tokenizer) -> Dataset:
    """Apply the model's chat template and tokenize.

    We prepend a SYSTEM message so the LoRA learns the persona, not just the QA.
    Loss is computed over ALL tokens (simple causal LM); the tiny dataset and
    consistent format make this fine in practice without a response-only mask.
    """
    rendered: list[str] = []
    for r in records:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *r["messages"]]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
        rendered.append(text)

    ds = Dataset.from_dict({"text": rendered})

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            max_length=MAX_LEN,
            truncation=True,
            padding=False,
        )
        return out

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    return ds


def main() -> None:
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] base model        : {BASE_MODEL}")
    print(f"[train] dataset path      : {DATASET_PATH}")
    print(f"[train] output dir        : {OUTPUT_DIR}")
    print(f"[train] device            : cpu")
    print(f"[train] torch threads     : {torch.get_num_threads()}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    records = load_records(DATASET_PATH)
    print(f"[train] examples          : {len(records)}")
    train_ds = build_dataset(records, tokenizer)

    # model — fp32 on CPU is the safest path
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # required when training

    # LoRA — target the standard Llama-style projection names
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # collator does padding to longest in batch + builds labels for causal LM
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",      # we save adapter manually at the end
        report_to="none",
        seed=SEED,
        bf16=False,              # CPU bf16 is unreliable across hardware
        fp16=False,
        optim="adamw_torch",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    print("[train] starting training ...")
    trainer.train()

    # save the LoRA adapter (small) + tokenizer
    print(f"[train] saving adapter to {OUTPUT_DIR}")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("[train] done.")


if __name__ == "__main__":
    main()
