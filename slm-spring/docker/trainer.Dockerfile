FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Base tooling: JDK 21 (validator), build tools (llama.cpp), bellsoft repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates wget gnupg git build-essential cmake pkg-config \
 && wget -qO- https://download.bell-sw.com/pki/GPG-KEY-bellsoft | gpg --dearmor -o /usr/share/keyrings/bellsoft.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/bellsoft.gpg] https://apt.bell-sw.com/ stable main" > /etc/apt/sources.list.d/bellsoft.list \
 && apt-get update && apt-get install -y --no-install-recommends bellsoft-java21 \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/bellsoft-java21-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /workspace

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-warm embedding model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# llama.cpp for GGUF conversion + quantization.
# Only the llama-quantize binary is built; convert_hf_to_gguf.py is Python.
ARG LLAMA_CPP_REF=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
 && cd /opt/llama.cpp \
 && cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
 && cmake --build build --config Release -j$(nproc) --target llama-quantize \
 && pip install -r requirements.txt \
 && rm -rf build/CMakeFiles build/_deps

ENV LLAMA_CPP_DIR=/opt/llama.cpp \
    PATH="/opt/llama.cpp/build/bin:${PATH}"

CMD ["bash"]
