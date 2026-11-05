# syntax=docker/dockerfile:1
#  openDox - Parakeet ASR server (Silero VAD + Parakeet TDT v3 ONNX)
#
#  Defaults to the NVIDIA GPU image (CUDA 12.4.1 + cuDNN) so the container
#  uses the GPU whenever the host provides one. To build a CPU-only image
#  (smaller, no CUDA base):
#    docker build --build-arg DEVICE=cpu -t open-knowledge-ui-parakeet:cpu .
#  The GPU build still works on CPU-only hosts — onnxruntime falls back to
#  CPUExecutionProvider automatically — but the image is larger.

# ─── Base image: CUDA+cuDNN (default) or plain Python for CPU ──────────
# The GPU base matches whisper_server/Dockerfile (CUDA 12.4.1 + cuDNN) so
# both containers can share the same host driver.
ARG DEVICE=gpu
# Basis-Images auf Manifest-Digest gepinnt (multi-arch, amd64+arm64); Dependabot aktualisiert den Digest zusammen mit dem Tag.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS base-cpu
# Dynamisch quantisierte int8-Gewichte (~0,7 statt ~2,5 GB) sind fuer CPU-Inferenz gebaut.
ENV PARAKEET_QUANTIZATION=int8
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6 AS base-gpu
# GPU: volle Gewichte; int8 laeuft auf dem CUDA-Provider teils auf der CPU und ist dort langsamer.
ENV PARAKEET_QUANTIZATION=
# Die CUDA-Basis hat kein Python: nur hier per apt installieren. Im CPU-Image bliebe sonst ein zweites
# (Debian-)Python neben dem des Images, und pip installierte in ein anderes Python als das, das laeuft.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# Final stage selects the right base via the DEVICE arg.
FROM base-${DEVICE} AS final

ARG DEVICE=cpu
ENV DEVICE=${DEVICE} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: ffmpeg for audio decoding, curl for healthcheck.
# The CUDA base is Ubuntu 22.04 (apt), the CPU base is Debian slim (apt) —
# both use the same package names.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install onnx-asr with the matching extra. [gpu] pulls onnxruntime-gpu
# (CUDA 12), [cpu] pulls plain onnxruntime. onnxruntime is imported
# directly in streaming.py for the raw Silero VAD session, so it must be
# installed explicitly regardless of the extra.
COPY requirements.cpu.txt ./requirements.cpu.txt
COPY requirements.gpu.txt ./requirements.gpu.txt
RUN python -m pip install --no-cache-dir -r requirements.${DEVICE}.txt

# Download models at build time for faster container starts.
# Parakeet ASR (~300MB) goes to the default HF cache; Silero VAD (~2MB) goes
# into a flat directory under /app/models/silero-vad so onnxruntime accepts it
# (the HF cache stores files as symlinks into blobs/, which onnxruntime rejects
# for the raw VAD session we drive directly in streaming.py).
# Nur die Gewichte zu PARAKEET_QUANTIZATION (je Basis-Stage gesetzt) landen im Image; zur Laufzeit
# umgestellt laedt onnx-asr die andere Variante beim ersten Start nach.
RUN python -c "import os, onnx_asr; onnx_asr.load_model('nemo-parakeet-tdt-0.6b-v3', quantization=os.environ.get('PARAKEET_QUANTIZATION') or None)" \
    && python -c "import onnx_asr; onnx_asr.load_vad('silero', path='/app/models/silero-vad')"

COPY streaming.py .
COPY main.py .

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/v1/models || exit 1

# WebSocket-Ping, damit Proxys lange Live-Aufnahmen nicht als untaetig trennen
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001", "--ws-ping-interval", "20", "--ws-ping-timeout", "60"]
