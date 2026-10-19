# Basis-Images auf Manifest-Digest gepinnt (multi-arch, amd64+arm64); Dependabot aktualisiert den Digest zusammen mit dem Tag.
# --- Stage 1: React-Frontend bauen -------------------------------------------
# Separater Node-Build-Layer; das Ergebnis (statische Assets) wird in das
# Runtime-Image kopiert. Node ist zur Laufzeit nicht noetig.
FROM node:26-slim@sha256:0d1e33586362bb9e608c4e751e7a9e7edb55c9ac11461ed5666af696cc461dd4 AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci

COPY frontend ./frontend
# vite schreibt direkt nach ../app/static (relativ zu frontend/) => /build/app/static
RUN cd frontend && npm run build


# --- Stage 2: Python-Runtime -------------------------------------------------
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

LABEL maintainer="open-knowledge-ui"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Modell-Caches im Datenbereich? Nein: eigene Ordnung unter /app/.cache (Volume-Volumes
# drehen sich in docker-compose.yml um diese Pfade). HF_HOME deckt HuggingFace-Modelle
# (Embedding-/Reranking-Gewichte, Docling-Layout/TableFormer) ab, FASTEMBED_CACHE_PATH
# die ONNX-Caches von fastembed.
ENV HF_HOME=/app/.cache/huggingface
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gosu \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (system-wide, direkt verfügbar)
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies system-wide (no .venv!)
RUN uv pip install --system -e .

# PyTorch mit CUDA: pyproject.toml pinnt torch fuer lokale Entwicklung auf das CPU-Wheel.
# Im Image ersetzen wir es durch das CUDA-Build von PyPI (bringt die CUDA-Runtime mit, auf
# dem Host genuegt der NVIDIA-Treiber + Container Toolkit). Beschleunigt Docling, das
# Embedding (HuggingFace-Modelle wie bge-m3) und den Reranker. Ohne GPU:
#   docker compose build --build-arg TORCH_BACKEND=cpu
ARG TORCH_BACKEND=cuda
RUN if [ "$TORCH_BACKEND" = "cuda" ]; then \
      uv pip install --system --reinstall-package torch --reinstall-package torchvision torch torchvision ; \
    fi

# Embedding- + Sparse-Modell zur Build-Zeit in den Cache laden,
# damit der Containerstart nicht auf einen Download wartet (Start-Timeout-Fix).
# Verwendet FastEmbed (ONNX) statt HuggingFace/Torch — schnellerer Start,
# kein PyTorch-Overhead. Wenn EMBEDDING_MODEL geaendert wird, mit
# --build-arg EMBEDDING_MODEL=... neu bauen, damit der Name mit dem
# runtime EMBEDDING_MODEL (DB/Env) uebereinstimmt.
# Hinweis: fastembed erwartet den HF-Org-Prefix (z. B. sentence-transformers/...).
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN python -c "from langchain_community.embeddings import FastEmbedEmbeddings; \
      FastEmbedEmbeddings(model_name='${EMBEDDING_MODEL}')" \
 && python -c "from langchain_qdrant import FastEmbedSparse; \
      FastEmbedSparse(model_name='Qdrant/bm25')"

# Docling-Modelle (Layout/DocLayNet + TableFormer) zur Build-Zeit in den
# HF-Cache laden, damit der Containerstart nicht auf einen Download wartet.
# Docling lädt die Gewichte erst beim convert(), nicht beim Instanziieren —
# deshalb wird hier eine Minimal-PDF (via Pillow, bereits Dep) erzeugt und
# konvertiert (nutzt die GPU, falls beim Build verfuegbar, sonst CPU).
# OCR-Engine zur Laufzeit wählbar (OCR_ENGINE, siehe app/components/ocr.py). Das Warmup
# lädt nur die engine-unabhängigen Layout/TableFormer-Gewichte — bei rapidocr/ocrmac/off
# ändert sich nichts, die Tesseract-Binary bleibt trotzdem installiert.
RUN python -c "\
from PIL import Image; \
Image.new('RGB',(100,100),'white').save('/tmp/_warm.pdf'); \
from docling.document_converter import DocumentConverter, PdfFormatOption; \
from docling.datamodel.base_models import InputFormat; \
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions; \
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=PdfPipelineOptions(do_ocr=True, do_table_structure=True, ocr_options=TesseractCliOcrOptions(lang=['deu','eng'])))}); \
conv.convert('/tmp/_warm.pdf'); \
import os; os.remove('/tmp/_warm.pdf')"

# Standardmodelle fuer Embedding und Reranking (HuggingFace) zur Build-Zeit laden, damit eine
# frische Installation ohne Download startet. Andere Modelle waehlt man unter
# Admin > Einlesen & Modelle (Download beim ersten Mal, Cache im Volume hf-cache).
ARG HF_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG HF_RERANKER_MODEL=BAAI/bge-reranker-base
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${HF_EMBEDDING_MODEL}'); \
CrossEncoder('${HF_RERANKER_MODEL}')"

# Copy rest of application
COPY . .

# Gebaute React-SPA aus dem Frontend-Stage uebernehmen. Flask serviert
# app/static/ und der 404-Catch-all liefert app/static/index.html.
COPY --from=frontend /build/app/static ./app/static

# Create directories. Die Daten-Subdirs (files/processed_files/images/
# summaries) legt ensure_directories zur Laufzeit unter /app/data an (das
# app-data-Volume ueberlagert /app/data leer).
RUN mkdir -p /app/data

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Laeuft nicht als root (uid 1000, gid 0 fuer Docker-Volume-Kompatibilitaet). Der
# entrypoint wechselt beim Start mit gosu auf dieses Konto und gibt vorher die
# Daten-/Cache-Verzeichnisse frei (bestehende Volumes sind root-gehoert).
ARG APP_UID=1000
RUN mkdir -p /app/data /app/.cache \
    && useradd --uid "${APP_UID}" --gid 0 --no-create-home --system --shell /usr/sbin/nologin app \
    && chown -R "${APP_UID}:0" /app/data /app/.cache \
    && chmod -R g+rwX /app/data /app/.cache

EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "run_linux.py"]
