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
Persistenz fuer LLM-Connections und Modell-Konfigurationen.

Modell-Referenzen (``ref``), mit denen Chat und API ein Modell ansprechen:

- ``base:<connection_id>:<modell-id>``  Modell eines Endpunkts (ggf. mit Anpassung)
- ``clone:<config_id>``                 geklontes Modell mit eigenem Namen

Rueckgaben sind einfache dicts (keine ORM-Objekte), damit nach dem Schliessen
der Session kein DetachedInstanceError auftritt. Der API-Key verlaesst dieses
Modul nur ueber ``get_connection(..., with_key=True)``.
"""

import json
from logging import getLogger
from typing import Any

from app.core.database_service import get_db_service
from app.core.models import Connection, ModelConfig
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

CONNECTION_KINDS = ("llm", "tei", "infinity")
THINKING_LEVELS = ("none", "low", "medium", "high", "max")

# key -> (Typ, Minimum, Maximum)
PARAM_SPECS: dict[str, tuple[type, float | None, float | None]] = {
    "temperature": (float, 0.0, 2.0),
    "top_p": (float, 0.0, 1.0),
    "top_k": (int, 0, None),
    "max_tokens": (int, 1, None),
    "num_ctx": (int, 1, None),
}


def normalize_base_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def sanitize_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validiert Modell-Parameter. Leere Werte entfallen (= globaler Default).

    Raises:
        ValueError: bei unbekanntem Schluessel oder ungueltigem Wert.
    """
    clean: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if key == "thinking":
            if value not in THINKING_LEVELS:
                raise ValueError(
                    f"thinking muss eines von {', '.join(THINKING_LEVELS)} sein"
                )
            clean[key] = value
            continue
        if key not in PARAM_SPECS:
            raise ValueError(f"Unbekannter Parameter: {key}")
        typ, lo, hi = PARAM_SPECS[key]
        try:
            num = typ(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} ist keine gültige Zahl") from None
        if (lo is not None and num < lo) or (hi is not None and num > hi):
            raise ValueError(
                f"{key} muss zwischen {lo} und {hi or 'unbegrenzt'} liegen"
            )
        clean[key] = num
    return clean


# --------------------------------------------------------------------------
# Connections
# --------------------------------------------------------------------------


def _connection_dict(c: Connection, with_key: bool = False) -> dict:
    data = {
        "id": c.id,
        "name": c.name,
        "base_url": c.base_url,
        "enabled": bool(c.enabled),
        "kind": c.kind or "llm",
        "has_key": bool(c.api_key),
    }
    if with_key:
        data["api_key"] = c.api_key or ""
    return data


def ensure_default_connection(db_session) -> None:
    """Legt aus den bisherigen CHAT_BASEURL/API_KEY-Settings die Connection
    "Standard" an, falls noch keine Connection existiert (idempotent)."""
    if db_session.query(Connection).count() > 0:
        return
    settings = SettingsService(db_session)
    base_url = normalize_base_url(
        settings.get("CHAT_BASEURL") or "http://localhost:11434/v1"
    )
    db_session.add(
        Connection(
            name="Standard",
            base_url=base_url,
            api_key=settings.get("API_KEY") or "",
            enabled=True,
        )
    )


def list_connections() -> list[dict]:
    db = get_db_service().get_session()
    try:
        return [
            _connection_dict(c) for c in db.query(Connection).order_by(Connection.id)
        ]
    finally:
        db.close()


def get_connection(connection_id: int, with_key: bool = False) -> dict | None:
    db = get_db_service().get_session()
    try:
        c = db.query(Connection).filter_by(id=connection_id).first()
        return _connection_dict(c, with_key) if c else None
    finally:
        db.close()


def check_kind(kind: str) -> str:
    if kind not in CONNECTION_KINDS:
        raise ValueError(f"Typ muss eines von {', '.join(CONNECTION_KINDS)} sein")
    return kind


def create_connection(
    name: str, base_url: str, api_key: str, enabled: bool = True, kind: str = "llm"
) -> dict:
    check_kind(kind)
    db = get_db_service().get_session()
    try:
        c = Connection(
            kind=kind,
            name=name.strip(),
            base_url=normalize_base_url(base_url),
            api_key=(api_key or "").strip(),
            enabled=enabled,
        )
        db.add(c)
        db.commit()
        return _connection_dict(c)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def update_connection(
    connection_id: int,
    name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    enabled: bool | None = None,
    kind: str | None = None,
) -> dict | None:
    """Aktualisiert Felder; ``api_key=None`` oder leer laesst den Key unveraendert."""
    db = get_db_service().get_session()
    try:
        c = db.query(Connection).filter_by(id=connection_id).first()
        if not c:
            return None
        if name is not None and name.strip():
            c.name = name.strip()
        if base_url is not None and base_url.strip():
            c.base_url = normalize_base_url(base_url)
        if api_key:
            c.api_key = api_key.strip()
        if enabled is not None:
            c.enabled = enabled
        if kind is not None:
            c.kind = check_kind(kind)
        db.commit()
        return _connection_dict(c)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_connection(connection_id: int) -> bool:
    """Loescht die Connection samt Modell-Konfigurationen."""
    db = get_db_service().get_session()
    try:
        c = db.query(Connection).filter_by(id=connection_id).first()
        if not c:
            return False
        config_ids = [
            m.id for m in db.query(ModelConfig).filter_by(connection_id=connection_id)
        ]
        db.query(ModelConfig).filter_by(connection_id=connection_id).delete()
        db.delete(c)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    from app.auth.access import delete_grants_for_resource

    for config_id in config_ids:
        delete_grants_for_resource("model", config_id)

    default = get_default_model()
    if default and default.startswith(f"base:{connection_id}:"):
        set_default_model("")
    return True


# --------------------------------------------------------------------------
# Modell-Konfigurationen
# --------------------------------------------------------------------------


def make_base_ref(connection_id: int, model_id: str) -> str:
    return f"base:{connection_id}:{model_id}"


def make_clone_ref(config_id: int) -> str:
    return f"clone:{config_id}"


def parse_ref(ref: str) -> tuple[str, int, str | None]:
    """Zerlegt eine Referenz in (Typ, connection_id bzw. config_id, modell-id).

    Raises:
        ValueError: bei ungueltiger Referenz.
    """
    try:
        kind, _, rest = (ref or "").partition(":")
        if kind == "base":
            conn, _, model = rest.partition(":")
            if model:
                return "base", int(conn), model
        elif kind == "clone":
            return "clone", int(rest), None
    except ValueError:
        pass
    raise ValueError(f"Ungültige Modell-Referenz: {ref!r}")


def _config_dict(m: ModelConfig) -> dict:
    return {
        "id": m.id,
        "connection_id": m.connection_id,
        "base_model": m.base_model,
        "name": m.name or "",
        "is_clone": bool(m.is_clone),
        "visible": bool(m.visible),
        "is_public": bool(m.is_public),
        "system_prompt": m.system_prompt or "",
        "web_search": bool(m.web_search),
        "vision": bool(m.vision),
        "params": json.loads(m.params) if m.params else {},
        "knowledge_ids": json.loads(m.knowledge_ids) if m.knowledge_ids else None,
        "allow_extra_knowledge": bool(m.allow_extra_knowledge)
        if m.allow_extra_knowledge is not None
        else True,
        "ref": make_clone_ref(m.id)
        if m.is_clone
        else make_base_ref(m.connection_id, m.base_model),
    }


def list_model_configs() -> list[dict]:
    db = get_db_service().get_session()
    try:
        return [_config_dict(m) for m in db.query(ModelConfig).order_by(ModelConfig.id)]
    finally:
        db.close()


def get_model_config(ref: str) -> dict | None:
    """Konfiguration zu einer Referenz; ``None`` = unkonfiguriertes Basismodell."""
    kind, key, model = parse_ref(ref)
    db = get_db_service().get_session()
    try:
        q = db.query(ModelConfig)
        if kind == "clone":
            m = q.filter_by(id=key, is_clone=True).first()
        else:
            m = q.filter_by(connection_id=key, base_model=model, is_clone=False).first()
        return _config_dict(m) if m else None
    finally:
        db.close()


def _apply(m: ModelConfig, data: dict) -> None:
    if "name" in data:
        name = (data["name"] or "").strip()
        if m.is_clone:
            if name:  # ein Klon braucht immer einen Namen
                m.name = name
        else:
            m.name = name or None  # Alias fuer ein Basismodell; leer = Original-ID
    if "visible" in data:
        m.visible = bool(data["visible"])
    if "is_public" in data:
        m.is_public = bool(data["is_public"])
    if "web_search" in data:
        m.web_search = bool(data["web_search"])
    if "vision" in data:
        m.vision = bool(data["vision"])
    if "system_prompt" in data:
        m.system_prompt = (data["system_prompt"] or "").strip()
    if "params" in data:
        m.params = json.dumps(sanitize_params(data["params"]))
    if "allow_extra_knowledge" in data:
        m.allow_extra_knowledge = bool(data["allow_extra_knowledge"])
    if "knowledge_ids" in data:
        ids = data["knowledge_ids"]
        m.knowledge_ids = json.dumps(sorted({int(i) for i in ids})) if ids else None


def save_model_config(ref: str, data: dict) -> dict | None:
    """Speichert die Konfiguration (Anlegen bei Basismodellen, Update bei Klonen).

    Returns:
        Die gespeicherte Konfiguration oder ``None``, wenn Connection bzw. Klon fehlt.

    Raises:
        ValueError: bei ungueltigen Parametern.
    """
    kind, key, model = parse_ref(ref)
    db = get_db_service().get_session()
    try:
        if kind == "clone":
            m = db.query(ModelConfig).filter_by(id=key, is_clone=True).first()
            if not m:
                return None
        else:
            if not db.query(Connection).filter_by(id=key).first():
                return None
            m = (
                db.query(ModelConfig)
                .filter_by(connection_id=key, base_model=model, is_clone=False)
                .first()
            )
            if not m:
                m = ModelConfig(connection_id=key, base_model=model, is_clone=False)
                db.add(m)
        _apply(m, data)
        db.commit()
        return _config_dict(m)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def clone_model(ref: str, name: str) -> dict | None:
    """Legt einen Klon an; Prompt, Parameter und Knowledge werden uebernommen."""
    kind, key, model = parse_ref(ref)
    source = get_model_config(ref)
    db = get_db_service().get_session()
    try:
        if kind == "clone":
            if not source:
                return None
            connection_id, base_model = source["connection_id"], source["base_model"]
        else:
            if not db.query(Connection).filter_by(id=key).first():
                return None
            connection_id, base_model = key, model
        m = ModelConfig(
            connection_id=connection_id,
            base_model=base_model,
            is_clone=True,
            name=name.strip(),
            visible=True,
            # Klone sind standardmaessig privat (nur Admin), bis sie freigegeben werden.
            is_public=False,
        )
        if source:
            m.system_prompt = source["system_prompt"]
            m.web_search = source.get("web_search", False)
            m.vision = source.get("vision", False)
            m.allow_extra_knowledge = source.get("allow_extra_knowledge", True)
            m.params = json.dumps(source["params"])
            m.knowledge_ids = (
                json.dumps(source["knowledge_ids"]) if source["knowledge_ids"] else None
            )
        db.add(m)
        db.commit()
        return _config_dict(m)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def delete_model_config(ref: str) -> bool:
    """Loescht einen Klon bzw. setzt die Anpassung eines Basismodells zurueck."""
    kind, key, model = parse_ref(ref)
    existing = get_model_config(ref)
    db = get_db_service().get_session()
    try:
        q = db.query(ModelConfig)
        if kind == "clone":
            deleted = q.filter_by(id=key, is_clone=True).delete()
        else:
            deleted = q.filter_by(
                connection_id=key, base_model=model, is_clone=False
            ).delete()
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    if deleted and existing:
        from app.auth.access import delete_grants_for_resource

        delete_grants_for_resource("model", existing["id"])
    if deleted and get_default_model() == ref:
        set_default_model("")
    return bool(deleted)


def remove_knowledge_from_models(knowledge_id: int) -> None:
    """Entfernt ein geloeschtes Knowledge aus allen Modell-Konfigurationen."""
    db = get_db_service().get_session()
    try:
        for m in db.query(ModelConfig).filter(ModelConfig.knowledge_ids.isnot(None)):
            ids = [i for i in json.loads(m.knowledge_ids) if i != knowledge_id]
            m.knowledge_ids = json.dumps(ids) if ids else None
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# --------------------------------------------------------------------------
# Standardmodell
# --------------------------------------------------------------------------


def get_default_model() -> str:
    db = get_db_service().get_session()
    try:
        return SettingsService(db).get("DEFAULT_MODEL", "") or ""
    finally:
        db.close()


def set_default_model(ref: str) -> None:
    db = get_db_service().get_session()
    try:
        SettingsService(db).set("DEFAULT_MODEL", ref)
    finally:
        db.close()
