import sys
import types

from app.utils import device


def fake_torch(cuda=False, mps=False):
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )


def pick(monkeypatch, torch_mod, env=None):
    device.torch_device.cache_clear()
    monkeypatch.delenv("RAG_DEVICE", raising=False)
    if env:
        monkeypatch.setenv("RAG_DEVICE", env)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    result = device.torch_device()
    device.torch_device.cache_clear()
    return result


def test_prefers_cuda_then_mps_then_cpu(monkeypatch):
    assert pick(monkeypatch, fake_torch(cuda=True, mps=True)) == "cuda"
    assert pick(monkeypatch, fake_torch(mps=True)) == "mps"
    assert pick(monkeypatch, fake_torch()) == "cpu"


def test_env_overrides(monkeypatch):
    assert pick(monkeypatch, fake_torch(cuda=True), env="cpu") == "cpu"


def test_coreml_only_on_mac_when_available(monkeypatch):
    ort = types.SimpleNamespace(
        get_available_providers=lambda: [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.delenv("RAG_DEVICE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert device.onnx_providers() == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(sys, "platform", "linux")
    assert device.onnx_providers() is None
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    assert device.onnx_providers() == ["CPUExecutionProvider"]
