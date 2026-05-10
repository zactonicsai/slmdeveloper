# CPU-only training image: HF transformers + PEFT + llama.cpp GGUF converter.
# Stays small by using the python:slim base, CPU torch wheels, and skipping
# the llama.cpp C++ build (we only need the Python conversion script).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheel (much smaller than the default CUDA build)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<2.6"

# llama.cpp ships a Python converter (convert_hf_to_gguf.py) that handles
# Llama-family architectures including SmolLM2. We only need the Python side
# for f16 GGUF output, so we skip the C++ build entirely.
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && pip install -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

WORKDIR /workspace

# install training deps
COPY training/requirements.txt /workspace/training/requirements.txt
RUN pip install -r /workspace/training/requirements.txt

# project files
COPY training/ /workspace/training/
COPY data/ /workspace/data/
COPY scripts/ /workspace/scripts/
RUN chmod +x /workspace/scripts/*.sh

CMD ["/workspace/scripts/build.sh"]
