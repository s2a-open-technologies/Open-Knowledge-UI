"""
Minimaler Sanitizer-Dienst nach docs/sanitizer-api.md (Vorlage für eigene Adapter).

Erkennt hier nur E-Mail-Adressen, Telefonnummern und "Vorname Nachname"-Paare mit Regeln. Für den
echten Einsatz ersetzt du ``detect`` durch deinen Erkenner, z. B. Microsoft Presidio:

    from presidio_analyzer import AnalyzerEngine
    analyzer = AnalyzerEngine()
    def detect(text):
        return [(r.start, r.end, r.entity_type) for r in analyzer.analyze(text=text, language="de")]

Start:  uv run --with fastapi --with uvicorn uvicorn sanitizer_adapter:app --port 9500
"""

import re

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

API_KEY: str | None = None  # z. B. "geheim"; None = keine Anmeldung

PATTERNS = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "PHONE": re.compile(r"\+?\d[\d /-]{7,}\d"),
    "NAME": re.compile(r"\b[A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+\b"),
}

app = FastAPI(title="Sanitizer")


class Request(BaseModel):
    text: str
    language: str | None = None


def detect(text: str) -> list[tuple[int, int, str]]:
    found = []
    for kind, pattern in PATTERNS.items():
        found += [(m.start(), m.end(), kind) for m in pattern.finditer(text)]
    found.sort()
    result, last_end = [], -1
    for start, end, kind in found:  # Überlappungen verwerfen
        if start >= last_end:
            result.append((start, end, kind))
            last_end = end
    return result


@app.post("/sanitize")
def sanitize(body: Request, authorization: str | None = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="API-Key fehlt oder ist falsch")
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}
    seen: dict[str, str] = {}  # gleicher Wert -> gleicher Platzhalter
    parts, pos = [], 0
    for start, end, kind in detect(body.text):
        value = body.text[start:end]
        if value not in seen:
            counters[kind] = counters.get(kind, 0) + 1
            seen[value] = f"[{kind}_{counters[kind]}]"
            mapping[seen[value]] = value
        parts += [body.text[pos:start], seen[value]]
        pos = end
    parts.append(body.text[pos:])
    return {"filtered": "".join(parts), "mapping": mapping}
