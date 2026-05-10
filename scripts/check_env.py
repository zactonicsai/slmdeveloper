import os
import torch
import transformers
import peft

print("=== SLM Dev Environment Check ===")
print(f"MODEL_NAME={os.getenv('MODEL_NAME', 'distilgpt2')}")
print(f"PyTorch={torch.__version__}")
print(f"Transformers={transformers.__version__}")
print(f"PEFT={peft.__version__}")
print(f"CUDA available={torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU count={torch.cuda.device_count()}")
    print(f"GPU name={torch.cuda.get_device_name(0)}")
else:
    print("Running CPU mode. Good for distilgpt2 demos; larger SLM fine-tuning will be slow.")
