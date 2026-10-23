"""OCR-Auswahl per env (app/ingest/loaders/ocr.py): Mapping, Fail-soft-Fallback, Sprach-Parsing."""

from types import SimpleNamespace

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    OcrMacOptions,
    RapidOcrOptions,
    TesseractCliOcrOptions,
)

import app.ingest.loaders.preprocess_pdf as pp
from app.ingest.loaders import ocr
from app.ingest.loaders.ocr import DEFAULT_LANGUAGES, build_ocr_options

KEYS = ("OCR_ENGINE", "OCR_LANGUAGES", "OCR_TESSERACT_CMD")


def _clean_env(monkeypatch):
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)


class FakeConverter:
    def __init__(self, format_options):
        self.format_options = format_options


def _built_pipeline_options(monkeypatch):
    """Instantiiert PreprocessPDF ohne __init__ (kein VisionLLM) und faengt die
    PdfFormatOption über einen gefakten DocumentConverter ab."""
    captured = {}

    class FakeConverter:
        def __init__(self, format_options):
            self.format_options = format_options
            captured[InputFormat.PDF] = format_options[InputFormat.PDF]

    monkeypatch.setattr(pp, "DocumentConverter", FakeConverter)
    pdf = object.__new__(pp.PreprocessPDF)
    pdf._build_converter()
    return captured[InputFormat.PDF].pipeline_options


def test_default_is_tesseract_cli(monkeypatch):
    _clean_env(monkeypatch)
    do_ocr, options = build_ocr_options()
    assert do_ocr is True
    assert isinstance(options, TesseractCliOcrOptions)
    assert options.lang == ["deu", "eng"]
    assert options.tesseract_cmd == "tesseract"


def test_off_disables_ocr(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "off")
    assert build_ocr_options() == (False, None)


def test_rapidocr_uses_first_language_codes(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "rapidocr")
    monkeypatch.setenv("OCR_LANGUAGES", "en,de")
    do_ocr, options = build_ocr_options()
    assert do_ocr is True
    assert isinstance(options, RapidOcrOptions)
    assert options.lang == ["en", "de"]


def test_ocrmac_outside_macos_falls_back_to_tesseract(monkeypatch, caplog):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "ocrmac")
    monkeypatch.setattr(ocr, "sys", SimpleNamespace(platform="linux"))
    do_ocr, options = build_ocr_options()
    assert do_ocr is True
    assert isinstance(options, TesseractCliOcrOptions)
    assert "ocrmac" in caplog.text


def test_ocrmac_on_darwin_builds_vision_options(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "ocrmac")
    monkeypatch.setattr(ocr, "sys", SimpleNamespace(platform="darwin"))
    do_ocr, options = build_ocr_options()
    assert do_ocr is True
    assert isinstance(options, OcrMacOptions)


def test_unknown_engine_falls_back_and_warns(monkeypatch, caplog):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "whisper-ocr")
    do_ocr, options = build_ocr_options()
    assert do_ocr is True
    assert isinstance(options, TesseractCliOcrOptions)
    assert "whisper-ocr" in caplog.text


def test_tesseract_cmd_is_passed_through(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv(
        "OCR_TESSERACT_CMD", "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
    )
    _, options = build_ocr_options()
    assert options.tesseract_cmd == "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"


def test_language_parsing_strips_gaps_and_keeps_default(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_LANGUAGES", "  deu ,  eng ,  ")
    _, options = build_ocr_options()
    assert options.lang == ["deu", "eng"]
    monkeypatch.setenv("OCR_LANGUAGES", ",,")
    _, options = build_ocr_options()
    assert options.lang == ["deu", "eng"]
    assert DEFAULT_LANGUAGES == "deu,eng"


def test_converter_off(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("OCR_ENGINE", "off")
    options = _built_pipeline_options(monkeypatch)
    assert options.do_ocr is False


def test_converter_default_carries_tesseract(monkeypatch):
    _clean_env(monkeypatch)
    options = _built_pipeline_options(monkeypatch)
    assert options.do_ocr is True
    assert isinstance(options.ocr_options, TesseractCliOcrOptions)
