"""Fine-tune the SLM with LoRA on the DTO/Controller dataset.

Uses TRL's SFTTrainer for clean instruction tuning. 4-bit quantization (QLoRA)
is enabled by default to keep VRAM use under ~6 GB even for Phi-2.
"""

import os
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

from prompts import build_training_prompt


load_dotenv()

BASE_MODEL = os.getenv("BASE_MODEL", "microsoft/phi-2")
ADAPTER_DIR = Path(os.getenv("ADAPTER_DIR", "./adapters"))
DATASET_PATH = Path(os.getenv("DATASET_PATH", "./data/dto_controller_dataset.jsonl"))

LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-4"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM_STEPS", "4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))
USE_4BIT = os.getenv("USE_4BIT", "true").lower() == "true"


# Target modules differ per model family — these are the safe defaults.
TARGET_MODULES_PER_MODEL = {
    "microsoft/phi-2": ["q_proj", "k_proj", "v_proj", "dense"],
    "distilgpt2": ["c_attn"],
    "gpt2": ["c_attn"],
}


def get_target_modules(model_name: str) -> list:
    for key, modules in TARGET_MODULES_PER_MODEL.items():
        if key in model_name:
            return modules
    # fallback
    return ["q_proj", "v_proj"]


def format_example(example: dict) -> dict:
    """Map dataset row -> training-formatted prompt string."""
    return {"text": build_training_prompt(example["instruction"], example["output"])}


def main() -> None:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Base model: {BASE_MODEL}")
    print(f"4-bit QLoRA: {USE_4BIT}")
    print(f"Dataset: {DATASET_PATH}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- quantization config ----
    bnb_config = None
    if USE_4BIT and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=get_target_modules(BASE_MODEL),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- dataset ----
    raw_ds = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    train_ds = raw_ds.map(format_example, remove_columns=raw_ds.column_names)
    print(f"Training rows: {len(train_ds)}")

    # ---- trainer ----
    sft_config = SFTConfig(
        output_dir=str(ADAPTER_DIR / "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if USE_4BIT and torch.cuda.is_available() else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=sft_config,
    )

    trainer.train()

    # ---- save adapter only ----
    final_path = ADAPTER_DIR / "final"
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"LoRA adapter saved to {final_path}")


if __name__ == "__main__":
    main()
