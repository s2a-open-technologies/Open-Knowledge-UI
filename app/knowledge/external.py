# Open Knowledge UI - A privacy-focused, self-hosted RAG system
# SPDX-FileCopyrightText: 2026 S2A Open Technologies gGmbH
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 S2A Open Technologies gGmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Externe Vektorspeicher (nur lesen).

Eine externe Quelle erscheint als Knowledge der Art ``external``. Ihre Konfiguration liegt in
``knowledge.source_config``. Nichts wird geraten: URL, Collection, Vektornamen, Feldnamen und
das Embedding-Modell gibt der Anleger an; ``probe_source`` zeigt nur Fakten zum Abgleich.
Es werden ausschliesslich lesende Aufrufe benutzt (query_points, scroll, count, get_collection).
"""

import json
from dataclasses import dataclass, field
from logging import getLogger

from qdrant_client import QdrantClient

logger = getLogger(__name__)

TIMEOUT_S = 20
DEFAULT_FIELDS = {
    "text": "page_content",
    "filename": "filename",
    "title": "title",
    "headers": "headers",
    "page_start": "page_start",
    "page_end": "page_end",
    "chunk_index": "chunk_index",
    "group": "file_group_id",
}
KINDS = ("qdrant",)
DEFAULT_ACCESS = {
    "enabled": False,
    "user_field": "",  # Payload-Feld, das den Benutzer enthaelt
    "user_attr": "username",  # username | display_name
    "group_field": "",  # Payload-Feld mit Gruppennamen
    "groups_from": "local",  # local | external | both
    "static": [],  # feste Bedingungen fuer alle: [{"field": "mandant", "values": ["A"]}]
    "admins_unfiltered": False,
}


class SourceError(Exception):
    """Fachlicher Fehler mit lesbarer Meldung (400/502 in der API)."""


@dataclass(frozen=True)
class SourceConfig:
    kind: str = "qdrant"
    url: str = ""
    api_key: str = field(default="", repr=False)
    collection: str = ""
    dense_vector: str = ""  # leer = unbenannter Vektor
    sparse_vector: str = ""  # leer = keine Stichwortsuche (nur dense)
    fields: dict = field(default_factory=lambda: dict(DEFAULT_FIELDS))
    access: dict = field(default_factory=lambda: dict(DEFAULT_ACCESS))

    def to_dict(self, with_key: bool = True) -> dict:
        d = {
            "kind": self.kind,
            "url": self.url,
            "collection": self.collection,
            "dense_vector": self.dense_vector,
            "sparse_vector": self.sparse_vector,
            "fields": dict(self.fields),
            "access": dict(self.access),
        }
        if with_key:
            d["api_key"] = self.api_key
        return d

    def public(self) -> dict:
        """Fuer Antworten an die Oberflaeche: nie der Key."""
        return {**self.to_dict(with_key=False), "has_key": bool(self.api_key)}

    @property
    def uses_default_neighbor_fields(self) -> bool:
        return (
            self.fields.get("chunk_index") == DEFAULT_FIELDS["chunk_index"]
            and self.fields.get("group") == DEFAULT_FIELDS["group"]
        )


def parse_config(raw: dict | str | None) -> SourceConfig:
    if isinstance(raw, str):
        raw = json.loads(raw or "{}")
    raw = raw or {}
    fields = {
        **DEFAULT_FIELDS,
        **{
            k: (v or DEFAULT_FIELDS[k])
            for k, v in (raw.get("fields") or {}).items()
            if k in DEFAULT_FIELDS
        },
    }
    access = {
        **DEFAULT_ACCESS,
        **{k: v for k, v in (raw.get("access") or {}).items() if k in DEFAULT_ACCESS},
    }
    access["static"] = [
        {
            "field": str(c.get("field", "")).strip(),
            "values": [str(v) for v in (c.get("values") or [])],
        }
        for c in (access.get("static") or [])
        if isinstance(c, dict) and str(c.get("field", "")).strip()
    ]
    return SourceConfig(
        access=access,
        kind=raw.get("kind") or "qdrant",
        url=(raw.get("url") or "").strip().rstrip("/"),
        api_key=raw.get("api_key") or "",
        collection=(raw.get("collection") or "").strip(),
        dense_vector=(raw.get("dense_vector") or "").strip(),
        sparse_vector=(raw.get("sparse_vector") or "").strip(),
        fields=fields,
    )


def _check_url(url: str) -> None:
    from app.llm.registry import RemoteModelsError, _reject_link_local

    if not url.startswith(("http://", "https://")):
        raise SourceError("Die Adresse muss mit http:// oder https:// beginnen")
    try:
        _reject_link_local(url)
    except RemoteModelsError as e:
        raise SourceError(str(e)) from None


def connect(cfg: SourceConfig) -> QdrantClient:
    """Lese-Client fuer die Quelle (Zeitlimit, Link-Local gesperrt)."""
    if cfg.kind not in KINDS:
        raise SourceError(f"Unbekannte Art: {cfg.kind}")
    _check_url(cfg.url)
    return QdrantClient(
        url=cfg.url, api_key=cfg.api_key or None, timeout=TIMEOUT_S, prefer_grpc=False
    )


def _vector_info(collection_info) -> tuple[dict[str, int], list[str]]:
    """({Vektorname: Groesse}, [Sparse-Namen]) aus get_collection."""
    params = collection_info.config.params
    vectors = params.vectors
    dense: dict[str, int] = {}
    if hasattr(vectors, "size"):
        dense[""] = int(vectors.size)
    elif isinstance(vectors, dict):
        dense = {name: int(v.size) for name, v in vectors.items()}
    sparse = list((params.sparse_vectors or {}).keys())
    return dense, sparse


def probe_source(cfg: SourceConfig, client=None, sample_vector: bool = False) -> dict:
    """Fakten zur Quelle fuer den Abgleich durch den Anleger (rechnet nichts aus, rät nichts).

    Raises:
        SourceError: nicht erreichbar oder Collection fehlt.
    """
    own = client is None
    client = client or connect(cfg)
    try:
        try:
            names = [c.name for c in client.get_collections().collections]
        except Exception as e:
            raise SourceError(
                f"Nicht erreichbar: {type(e).__name__}: {e}"[:300]
            ) from None
        result: dict = {"collections": names, "reachable": True}
        if not cfg.collection:
            return result
        if cfg.collection not in names:
            raise SourceError(
                f"Collection „{cfg.collection}“ gibt es auf dem Server nicht"
            )
        info = client.get_collection(cfg.collection)
        dense, sparse = _vector_info(info)
        wanted = [cfg.dense_vector] if sample_vector else False
        points, _ = client.scroll(
            cfg.collection, limit=1, with_payload=True, with_vectors=wanted
        )
        result.update(
            {
                "points": int(client.count(cfg.collection).count),
                "dense_vectors": dense,  # Name ("" = unbenannt) -> Groesse
                "sparse_vectors": sparse,
                "sample_payload": (points[0].payload if points else None),
            }
        )
        if sample_vector and points:
            vec = points[0].vector
            vec = vec.get(cfg.dense_vector) if isinstance(vec, dict) else vec
            result["sample_vector"] = (
                list(vec) if isinstance(vec, (list, tuple)) else None
            )
        return result
    except SourceError:
        raise
    except Exception as e:
        raise SourceError(
            f"Abfrage fehlgeschlagen: {type(e).__name__}: {e}"[:300]
        ) from None
    finally:
        if own:
            client.close()


MIN_SIMILARITY = 0.8


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def validate_source(cfg: SourceConfig, dimension: int, client=None, embed=None) -> dict:
    """Harte Pruefung vor dem Speichern: Vektor und Dimension passen, Felder gefunden.

    Returns:
        Die Fakten von ``probe_source``.
    Raises:
        SourceError: mit der Ursache in lesbarer Form.
    """
    if not cfg.url:
        raise SourceError("Adresse fehlt")
    if not cfg.collection:
        raise SourceError("Collection fehlt")
    facts = probe_source(cfg, client, sample_vector=embed is not None)
    dense = facts["dense_vectors"]
    if cfg.dense_vector not in dense:
        known = ", ".join(repr(n) for n in dense) or "keine"
        raise SourceError(
            f"Vektor {cfg.dense_vector!r} gibt es in der Collection nicht (vorhanden: {known})"
        )
    if dense[cfg.dense_vector] != dimension:
        raise SourceError(
            f"Das Embedding-Modell liefert {dimension} Dimensionen, der Vektor der Quelle hat "
            f"{dense[cfg.dense_vector]}. Es muss das Modell sein, mit dem die Quelle aufgebaut wurde."
        )
    if cfg.sparse_vector and cfg.sparse_vector not in facts["sparse_vectors"]:
        raise SourceError(
            f"Sparse-Vektor {cfg.sparse_vector!r} gibt es nicht (vorhanden: {', '.join(facts['sparse_vectors']) or 'keine'})"
        )
    sample = facts.get("sample_payload")
    if sample is not None and cfg.fields["text"] not in sample:
        raise SourceError(
            f"Das Textfeld {cfg.fields['text']!r} steht nicht im Payload (vorhanden: {', '.join(sample) or 'keine'})"
        )
    if facts["points"] == 0:
        raise SourceError("Die Collection ist leer")
    # Gleiche Dimension heisst nicht gleiches Modell: den Text eines Beispielpunkts mit dem
    # angegebenen Modell einbetten und mit dem gespeicherten Vektor vergleichen.
    stored = facts.pop("sample_vector", None)
    if embed is not None and stored and sample:
        text = sample.get(cfg.fields["text"])
        if isinstance(text, str) and text.strip():
            similarity = _cosine(list(embed(text)), stored)
            facts["model_match"] = round(similarity, 3)
            if similarity < MIN_SIMILARITY:
                raise SourceError(
                    f"Das Embedding-Modell passt nicht zu den Vektoren der Quelle (Übereinstimmung "
                    f"{similarity:.2f} bei einem Beispieltext, erwartet über {MIN_SIMILARITY}). "
                    "Bitte das Modell angeben, mit dem die Quelle aufgebaut wurde."
                )
    return facts


def normalize_payload(payload: dict, cfg: SourceConfig) -> dict:
    """Uebersetzt die Felder der Quelle in die Namen, die die Pipeline erwartet."""
    out = dict(payload)
    f = cfg.fields
    for ours, theirs in (
        ("page_content", f["text"]),
        ("filename", f["filename"]),
        ("title", f["title"]),
        ("headers", f["headers"]),
        ("page_start", f["page_start"]),
        ("page_end", f["page_end"]),
        ("chunk_index", f["chunk_index"]),
        ("file_group_id", f["group"]),
    ):
        if theirs in payload:
            out[ours] = payload[theirs]
    return out


def layout_for(cfg: SourceConfig) -> dict:
    """Aufbau der Quelle fuer ``pipeline.run(layout=...)``."""
    return {
        "dense_name": cfg.dense_vector,
        "sparse_name": cfg.sparse_vector or None,
        "normalize": lambda payload: normalize_payload(payload, cfg),
        "neighbors": cfg.uses_default_neighbor_fields,
    }


def validate_access(cfg: SourceConfig) -> None:
    """Pruefung des Zugriffsfilters (nur wenn aktiviert)."""
    a = cfg.access
    if not a.get("enabled"):
        return
    if a["user_attr"] not in ("username", "display_name"):
        raise SourceError(
            "Zugriffsfilter: Benutzerattribut muss username oder display_name sein"
        )
    if a["groups_from"] not in ("local", "external", "both"):
        raise SourceError(
            "Zugriffsfilter: Gruppenquelle muss local, external oder both sein"
        )
    if not (a["user_field"] or a["group_field"] or a["static"]):
        raise SourceError(
            "Zugriffsfilter: mindestens ein Benutzerfeld, ein Gruppenfeld oder eine feste Bedingung angeben"
        )
    for cond in a["static"]:
        if not cond["values"]:
            raise SourceError(
                f"Zugriffsfilter: die feste Bedingung für „{cond['field']}“ braucht Werte"
            )


def _variants(value: str) -> list[str]:
    """Schreibweisen eines Namens (Qdrant vergleicht exakt): wie eingegeben, klein, gross,
    erster Buchstabe gross, jedes Wort gross. Exotische Mischschreibweisen muessen exakt passen."""
    return list(
        dict.fromkeys(
            [value, value.lower(), value.upper(), value.capitalize(), value.title()]
        )
    )


def build_filter(cfg: SourceConfig, identity: dict):
    """Qdrant-Filter fuer diesen Benutzer oder ``None`` (kein Filter konfiguriert).

    Rueckgabe ``(filter, blocked)``: ``blocked=True`` heisst, der Benutzer hat keine passende
    Identitaet, die Quelle liefert dann **nichts** (fail closed, nie ungefiltert).
    Semantik: alle festen Bedingungen UND mindestens eine Benutzer-/Gruppenbedingung.
    """
    from qdrant_client import models as qm

    a = cfg.access
    if not a.get("enabled"):
        return None, False
    if a.get("admins_unfiltered") and identity.get("is_admin"):
        return None, False

    must = [
        qm.FieldCondition(key=c["field"], match=qm.MatchAny(any=c["values"]))
        for c in a["static"]
    ]
    should = []
    if a["user_field"]:
        name = (
            identity.get(
                "display_name" if a["user_attr"] == "display_name" else "username"
            )
            or ""
        )
        if name:
            should.append(
                qm.FieldCondition(
                    key=a["user_field"], match=qm.MatchAny(any=_variants(name))
                )
            )
    if a["group_field"]:
        groups: list[str] = []
        if a["groups_from"] in ("local", "both"):
            groups += identity.get("groups") or []
        if a["groups_from"] in ("external", "both"):
            groups += identity.get("external_groups") or []
        names = list(
            dict.fromkeys(
                v for g in groups for v in _variants(str(g)) if str(g).strip()
            )
        )
        if names:
            should.append(
                qm.FieldCondition(key=a["group_field"], match=qm.MatchAny(any=names))
            )
    wants_identity = bool(a["user_field"] or a["group_field"])
    if wants_identity and not should:
        return None, True  # Filter verlangt eine Identitaet, der Benutzer hat keine
    flt = qm.Filter(
        must=must or None,
        min_should=qm.MinShould(conditions=should, min_count=1) if should else None,
    )
    return flt, False
