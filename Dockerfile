FROM intel/intel-extension-for-pytorch:2.6.10-xpu-pip-base

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    einops \
    pillow \
    tqdm \
    loguru \
    pyvips \
    fastapi \
    uvicorn \
    python-multipart \
    redis \
    prometheus-client \
    prometheus-fastapi-instrumentator \
    psutil \
    && pip cache purge

# dont move the above pip (pyvips build would fail)
RUN apt update && apt install -y --no-install-recommends \
    libvips-dev \
    && rm -rf /var/lib/apt/lists/*

# REQUEST_BATCH_SIZE = max connections that is allowed by the server
# MAX_BATCH_SIZE = max batch of requests processed together
ENV ONEAPI_DEVICE_SELECTOR="level_zero:0" \
    TOKENIZERS_PARALLELISM=true \
    TORCH_COMPILE=true \
    OMP_NUM_THREADS=28 \
    IPEX_COMPILER_CACHE_PATH=/tmp/ipex_cache \
    SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 \
    SYCL_CACHE_PERSISTENT=1 \
    ENABLE_SDP_FUSION=1 \
    HOST="0.0.0.0" \
    PORT=8000 \
    WORKERS=1 \
    REQUEST_BATCH_SIZE=32 \
    MAX_BATCH_SIZE=16 \
    DEVICE_ID=0 \
    BF16_MODE=true \
    MODEL_CACHE_DIR=/root/.cache/huggingface \
    MOONDREAM_ENABLE_OPTIMIZATION=1

RUN install -d -m 777 /tmp/ipex_cache /tmp/torch_cache /root/.cache/huggingface

COPY backend.py server.py ./

EXPOSE 8000
CMD ["python", "server.py"]