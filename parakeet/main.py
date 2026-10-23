# Standard library
import asyncio
import os
import struct
import tempfile
from contextlib import asynccontextmanager

import numpy as np

# Third-party imports
import onnx_asr
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from streaming import PCM_SAMPLE_RATE, parakeet_stream_engine

DEFAULT_MODEL = os.getenv("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3")

_models: dict[str, "onnx_asr.adapters.TextResultsAsrAdapter"] = {}


def _resolve_providers():
    """Return the ONNX providers list, excluding TensorRT.

    ``onnxruntime-gpu`` advertises ``TensorrtExecutionProvider`` in
    ``get_available_providers()``, but the CUDA base image has no TensorRT
    libs installed. If the full list is passed, onnxruntime tries TensorRT
    first, fails, and silently falls back to **CPU** — bypassing CUDA
    entirely. We filter it out so ``CUDAExecutionProvider`` is used directly.

    Override via ``RTT_ONNX_PROVIDERS`` (comma-separated).

    Returns:
        Provider list (or ``None`` for onnxruntime defaults).
    """
    import onnxruntime as ort

    env = os.environ.get("RTT_ONNX_PROVIDERS", "")
    if env.strip():
        return [p.strip() for p in env.split(",") if p.strip()]
    available = ort.get_available_providers()
    filtered = [p for p in available if p != "TensorrtExecutionProvider"]
    return filtered or None


def get_model(name: str | None = None) -> "onnx_asr.adapters.TextResultsAsrAdapter":
    """Load and cache an ONNX ASR model by name.

    Models are loaded on first request and kept in memory for subsequent
    requests. The default model is pre-loaded during application startup
    via the lifespan handler.

    Args:
        name: Model identifier (e.g. 'nemo-parakeet-tdt-0.6b-v3').
              Falls back to the PARAKEET_MODEL environment variable,
              then to DEFAULT_MODEL.

    Returns:
        The loaded and cached model instance.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    model_name = name or DEFAULT_MODEL

    if model_name in _models:
        return _models[model_name]

    print(f"[PARAKEET] Loading model: {model_name} ...")
    try:
        _models[model_name] = onnx_asr.load_model(
            model_name, providers=_resolve_providers()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    print(f"[PARAKEET] Model loaded: {model_name}")
    return _models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler: load the default model on startup, unload on shutdown.

    Also pre-loads the Silero VAD + Parakeet ONNX sessions used by the
    streaming WebSocket endpoint so the first client doesn't pay the cold-load
    penalty.
    """
    default_model = get_model()
    parakeet_stream_engine.ensure_loaded(asr_model=default_model)
    yield
    _models.clear()
    print("[PARAKEET] Models unloaded")


app = FastAPI(title="openDox Parakeet ASR", lifespan=lifespan)


@app.get("/v1/models")
async def list_models():
    """List available models.

    Returns:
        OpenAI-compatible /v1/models response with the default model.
    """
    return {"data": [{"id": DEFAULT_MODEL, "object": "model"}]}


def _run_recognition(wav_bytes: bytes, model_name: str | None) -> str:
    """Run model loading (if needed) + inference off the event loop.

    Both ``get_model()`` (can trigger a cold ``onnx_asr.load_model()`` on an
    unseen model name) and ``m.recognize()`` are blocking calls; running them
    synchronously inside the ``async def`` handler would freeze the whole
    server for every other in-flight request until this one finishes.
    """
    m = get_model(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        try:
            result = m.recognize(tmp.name)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"[PARAKEET] Recognition error: {e}")
            return ""


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
):
    """Transcribe audio using the Parakeet ONNX ASR model.

    OpenAI-compatible endpoint that accepts WAV or raw PCM audio
    (float32, 16 kHz, mono) and returns transcribed text with
    punctuation and capitalization.

    Args:
        file: Audio file (WAV or raw PCM).
        model: Model name to use (optional, defaults to PARAKEET_MODEL).
        language: Language hint (optional, currently unused by Parakeet).

    Returns:
        JSON response with a "text" field containing the transcription.
    """
    content = await file.read()
    if not content:
        return JSONResponse({"text": ""})

    wav_bytes = content if content[:4] == b"RIFF" else _pcm_to_wav(content)

    text = await asyncio.to_thread(_run_recognition, wav_bytes, model or None)

    return {"text": text}


def _pcm_to_wav(pcm: bytes) -> bytes:
    """Convert raw float32 16kHz mono PCM bytes to a WAV file in memory.

    Takes raw PCM sample data as bytes (float32 little-endian, 16 kHz,
    mono) and wraps it in a standard WAV file header.

    Args:
        pcm: Raw PCM audio data as float32 little-endian bytes.

    Returns:
        Complete WAV file as bytes (header + PCM int16 data).
    """
    audio = np.frombuffer(pcm, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (audio * 32767).astype(np.int16)

    num_channels = 1
    sample_rate = 16000
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm_int16) * 2

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )

    return header + pcm_int16.tobytes()


@app.websocket("/v1/audio/stream")
async def audio_stream(ws: WebSocket):
    """VAD-segmented streaming transcription over a WebSocket.

    The client sends raw **float32 16 kHz mono** PCM as binary frames. The
    server runs incremental Silero VAD + Parakeet ONNX ASR (see
    ``streaming.StreamingSession``) and pushes back JSON messages:

    * ``{"partial": "<committed + open partial>"}`` — live, growing text;
      emitted on every ingested frame where the transcript changed,
    * ``{"committed": "<frozen text>", "delta": "<newly committed chunk>"}``
      — sent whenever a speech segment is committed (trailing silence or
      auto-split), so the caller can freeze the corresponding text,
    * ``{"final": "<full transcript>"}`` — once when the client sends the
      ``{"event": "stop"}`` control message and the session is flushed.

    The client signals end-of-stream with a text frame ``{"event": "stop"}``.
    """
    await ws.accept()
    try:
        session = await asyncio.to_thread(parakeet_stream_engine.new_session)
    except RuntimeError as e:
        await ws.send_json({"error": f"stream engine unavailable: {e}"})
        await ws.close(code=1011)
        return

    last_committed = ""
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            # Text control frame
            if msg.get("text"):
                try:
                    ctrl = msg["text"]
                    if "stop" in ctrl.lower():
                        final = await asyncio.to_thread(session.finish)
                        await ws.send_json({"final": final})
                        break
                except Exception:
                    pass
                continue

            if msg.get("bytes"):
                pcm = msg["bytes"]
                if not pcm:
                    continue
                # ingest() runs VAD + (throttled) Parakeet inference — both are
                # blocking ONNX calls, so offload them to a worker thread to
                # keep the event loop responsive for other clients.
                await asyncio.to_thread(session.ingest, pcm, PCM_SAMPLE_RATE)

                # Emit a committed delta whenever a new segment was frozen.
                committed = await asyncio.to_thread(session.committed_text)
                if committed != last_committed:
                    delta = committed[len(last_committed) :].strip()
                    last_committed = committed
                    if delta:
                        await ws.send_json({"committed": committed, "delta": delta})

                # Emit the live partial — only the open segment's text, not the
                # committed portion, so the client can render it alongside its
                # own accumulated committed text without duplication.
                open_partial = await asyncio.to_thread(session.open_partial_text)
                if open_partial:
                    await ws.send_json({"partial": open_partial})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[PARAKEET-STREAM] error: {e}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass
