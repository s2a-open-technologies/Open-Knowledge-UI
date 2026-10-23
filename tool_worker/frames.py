"""CSV/Excel in einen DataFrame laden (wie die App bisher: Encoding-Erkennung, defekte Zeilen ueberspringen)."""

from pathlib import Path

import pandas as pd
from charset_normalizer import from_bytes

MAX_FILE_BYTES = 200 * 1024 * 1024
SUFFIXES = {".csv", ".xlsx", ".xls"}


class FrameError(Exception):
    """Datei nicht ladbar (Text ist fuer das Modell/den Nutzer gedacht)."""


def detect_encoding(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            best = from_bytes(fh.read(50000)).best()
        return best.encoding if best and best.encoding else "cp1252"
    except UnicodeEncodeError:
        return "cp1252"


def load_frame(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() not in SUFFIXES:
        raise FrameError("Dateityp nicht unterstützt")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise FrameError("Datei ist zu groß für die Auswertung")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p, encoding=detect_encoding(str(p)), on_bad_lines="skip")
