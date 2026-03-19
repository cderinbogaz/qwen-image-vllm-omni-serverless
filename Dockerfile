FROM vllm/vllm-omni:v0.16.0

USER root
WORKDIR /app

ENTRYPOINT []

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1 \
    MODEL_CACHE_DIR=/runpod-volume/model-cache \
    HF_HOME=/runpod-volume/model-cache/huggingface \
    HF_HUB_CACHE=/runpod-volume/model-cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/model-cache/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/model-cache/huggingface/transformers \
    VLLM_ASSETS_CACHE=/runpod-volume/model-cache/vllm-assets \
    VLLM_SERVER_HOST=127.0.0.1 \
    VLLM_SERVER_PORT=8091 \
    VLLM_STARTUP_TIMEOUT=3600 \
    VLLM_REQUEST_TIMEOUT=1800 \
    VLLM_MODEL=Qwen/Qwen-Image-2512 \
    VLLM_SERVER_ARGS="--num-gpus 1 --vae-use-slicing --vae-use-tiling" \
    DEFAULT_IMAGE_SIZE=1328x1328 \
    DEFAULT_NUM_INFERENCE_STEPS=50 \
    DEFAULT_TRUE_CFG_SCALE=4.0

CMD ["python3", "-u", "/app/handler.py"]
