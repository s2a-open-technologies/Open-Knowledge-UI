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

"""Wertverzeichnis einer SQL-Quelle aufbauen, lesen und durchsuchen (Persistenz, Job, Vorschlag).

Die Textsuche selbst steht in ``app/knowledge/sql/values.py``. Gespeichert werden nur die verschiedenen Werte
ausgewaehlter Textspalten mit Haeufigkeit (Tabelle ``sql_values``), keine Zeilen.
"""

import re
import threading
from datetime import datetime
from logging import getLogger

from app.agent.toolworker import config as worker_config
from app.agent.toolworker.client import ToolWorkerUnavailable, get_client
from app.core.database_service import get_db_service
from app.core.models import SqlValue, SqlValueIndex
from app.knowledge.service import get_knowledge
from app.knowledge.sql import values as text
from app.knowledge.sql.config import parse_config

logger = getLogger(__name__)

# Spalten, die eher Schluessel, Kontakt- oder Personendaten sind: nicht vorgeschlagen (der Admin kann sie aufnehmen)
SENSITIVE_NAME = re.compile(
    r"(mail|phone|telefon|mobil|fax|iban|bic|passw|token|secret|geburt|birth|steuer|ssn|konto|kredit|"
    r"vorname|nachname|firstname|first_name|lastname|last_name|surname|adress|address|strasse|street|"
    r"(^|_)(id|uuid|guid|hash|key)$|_(id|uuid|guid|hash)$)",
    re.IGNORECASE,
)
MOSTLY_UNIQUE = 0.5
MIN_ROWS_FOR_UNIQUE_RULE = 50

_running: set[int] = set()
_running_lock = threading.Lock()
_cache: dict[int, tuple[object, list[dict]]] = {}


def suggest(profile: list[dict], max_distinct: int) -> list[dict]:
    """Vorschlag je Textspalte: aufnehmen oder nicht, mit Begruendung."""
    out = []
    for c in profile:
        include, reason = True, "Kategoriefeld mit wenigen verschiedenen Werten"
        if c["non_null"] == 0:
            include, reason = False, "leer"
        elif SENSITIVE_NAME.search(c["column"]):
            include, reason = False, "Schlüssel-, Kontakt- oder Personenspalte"
        elif c["distinct"] > max_distinct:
            include, reason = (
                False,
                f"mehr als {max_distinct} verschiedene Werte (Freitext oder Schlüssel)",
            )
        elif (
            c["non_null"] >= MIN_ROWS_FOR_UNIQUE_RULE
            and c["distinct"] / c["non_null"] > MOSTLY_UNIQUE
        ):
            include, reason = (
                False,
                "fast jeder Wert ist eindeutig (Freitext oder Name)",
            )
        out.append({**c, "include": include, "reason": reason})
    return out


def clean_value_columns(cfg) -> dict:
    """Die gewaehlten Spalten ohne alles, was die aktuelle Konfiguration nicht (mehr) zulaesst."""
    allowed = {t.lower() for t in cfg.tables}
    blocked = {
        t.lower(): {c.lower() for c in cols} for t, cols in cfg.blocked_columns.items()
    }
    out = {}
    for table, columns in cfg.value_columns.items():
        if allowed and table.lower() not in allowed:
            continue
        kept = tuple(
            c for c in columns if c.lower() not in blocked.get(table.lower(), set())
        )
        if kept:
            out[table] = kept
    return out


def _permitted(cfg):
    """Prueffunktion (tabelle, spalte): gewaehlt, Tabelle freigegeben, Spalte nicht gesperrt."""
    allowed = {
        (t.lower(), c.lower())
        for t, cols in clean_value_columns(cfg).items()
        for c in cols
    }
    return lambda table, column: (table.lower(), column.lower()) in allowed


def _signature(cfg) -> tuple:
    return (
        tuple(sorted(cfg.tables)),
        tuple(sorted((t, tuple(sorted(c))) for t, c in cfg.blocked_columns.items())),
        tuple(sorted((t, tuple(sorted(c))) for t, c in cfg.value_columns.items())),
    )


def prune(knowledge_id: int, source_config) -> None:
    """Entfernt Werte von Spalten, die nicht mehr gewaehlt, gesperrt oder nicht mehr freigegeben sind."""
    permitted = _permitted(parse_config(source_config))
    db = get_db_service().get_session()
    try:
        rows = db.query(SqlValue).filter_by(knowledge_id=knowledge_id).all()
        for r in rows:
            if not permitted(r.table_name, r.column_name):
                db.delete(r)
        db.flush()
        left = db.query(SqlValue).filter_by(knowledge_id=knowledge_id).all()
        idx = db.query(SqlValueIndex).filter_by(knowledge_id=knowledge_id).first()
        if idx is not None:
            idx.value_count = len(left)
            idx.column_count = len({(r.table_name, r.column_name) for r in left})
        db.commit()
    finally:
        db.close()
    _cache.pop(knowledge_id, None)


def _payload(cfg, extra: dict) -> dict:
    limits = worker_config.load_sql()
    return {
        "connection": cfg.connection(),
        "allowed_tables": sorted(cfg.allowed_tables())
        if cfg.allowed_tables()
        else None,
        "blocked_columns": {t: sorted(c) for t, c in cfg.blocked().items()},
        "limits": {
            "timeout_s": max(limits.timeout_s, 30),
            "max_cell_chars": limits.max_cell_chars,
        },
        **extra,
    }


def profile(source_config) -> list[dict]:
    """Textspalten der freigegebenen Tabellen mit Vorschlag. Wirft ``ValueError`` (Text fuer den Admin)."""
    cfg = parse_config(source_config)
    max_distinct, _ = worker_config.values_settings()
    try:
        out = get_client("sql").sql("profile", _payload(cfg, {}), 120)
    except ToolWorkerUnavailable as e:
        raise ValueError(str(e)) from None
    if not out.get("ok"):
        raise ValueError(out.get("error") or "Profil nicht lesbar")
    return suggest(out.get("columns") or [], max_distinct)


# --------------------------------------------------------------------------------------
# Stand und Aufbau
# --------------------------------------------------------------------------------------


def status(knowledge_id: int) -> dict:
    db = get_db_service().get_session()
    try:
        row = db.query(SqlValueIndex).filter_by(knowledge_id=knowledge_id).first()
        st = row.status if row else "idle"
        with _running_lock:
            alive = knowledge_id in _running
        if st == "building" and not alive:
            st = "error"  # Prozess wurde waehrend des Aufbaus beendet
        return {
            "status": st,
            "built_at": row.built_at.isoformat() if row and row.built_at else None,
            "value_count": row.value_count if row else 0,
            "column_count": row.column_count if row else 0,
            "error": (
                row.error
                if row and row.error
                else ("Aufbau wurde unterbrochen" if st == "error" and row else "")
            ),
        }
    finally:
        db.close()


def _set_status(knowledge_id: int, **fields) -> None:
    db = get_db_service().get_session()
    try:
        row = db.query(SqlValueIndex).filter_by(knowledge_id=knowledge_id).first()
        if row is None:
            row = SqlValueIndex(knowledge_id=knowledge_id)
            db.add(row)
        for k, v in fields.items():
            setattr(row, k, v)
        db.commit()
    finally:
        db.close()


def build(knowledge_id: int, source_config) -> None:
    """Liest die verschiedenen Werte der gewaehlten Spalten aus der Quelle und ersetzt das Verzeichnis.
    Bei einem Fehler bleibt das alte Verzeichnis erhalten."""
    cfg = parse_config(source_config)
    max_distinct, _ = worker_config.values_settings()
    collected: list[tuple[str, str, str, int]] = []
    skipped: list[str] = []
    try:
        for table, columns in clean_value_columns(cfg).items():
            for column in columns:
                try:
                    out = get_client("sql").sql(
                        "distinct",
                        _payload(
                            cfg,
                            {
                                "table": table,
                                "column": column,
                                "max_distinct": max_distinct,
                            },
                        ),
                        180,
                    )
                except ToolWorkerUnavailable as e:
                    raise ValueError(str(e)) from None
                if not out.get("ok"):
                    raise ValueError(
                        f"{table}.{column}: {out.get('error') or 'nicht lesbar'}"
                    )
                if out.get("too_many"):
                    skipped.append(f"{table}.{column}")
                    continue
                collected += [
                    (table, column, value, count)
                    for value, count in out.get("values") or []
                ]
        db = get_db_service().get_session()
        try:
            db.query(SqlValue).filter_by(knowledge_id=knowledge_id).delete()
            db.bulk_insert_mappings(
                SqlValue,
                [
                    {
                        "knowledge_id": knowledge_id,
                        "table_name": t,
                        "column_name": c,
                        "value": v,
                        "count": n,
                    }
                    for t, c, v, n in collected
                ],
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        note = (
            f"Übersprungen (zu viele verschiedene Werte): {', '.join(skipped)}"
            if skipped
            else ""
        )
        _set_status(
            knowledge_id,
            status="ready",
            built_at=datetime.utcnow(),
            value_count=len(collected),
            column_count=len({(t, c) for t, c, _, _ in collected}),
            error=note,
        )
    except Exception as e:  # noqa: BLE001 - der Admin sieht den Grund, das alte Verzeichnis bleibt
        logger.warning(f"Wertverzeichnis {knowledge_id}: {type(e).__name__}: {e}")
        _set_status(
            knowledge_id,
            status="error",
            error=str(e)[:500]
            if isinstance(e, ValueError)
            else "Aufbau fehlgeschlagen",
        )
    finally:
        with _running_lock:
            _running.discard(knowledge_id)


def start_build(knowledge_id: int, source_config) -> bool:
    """Startet den Aufbau im Hintergrund; ``False``, wenn er schon laeuft."""
    with _running_lock:
        if knowledge_id in _running:
            return False
        _running.add(knowledge_id)
    _set_status(knowledge_id, status="building", error="")
    threading.Thread(
        target=build, args=(knowledge_id, source_config), daemon=True
    ).start()
    return True


# --------------------------------------------------------------------------------------
# Suche
# --------------------------------------------------------------------------------------


def entries(knowledge_id: int) -> list[dict]:
    st = status(knowledge_id)
    k = get_knowledge(knowledge_id)
    cfg = parse_config(k["source_config"]) if k else None
    if cfg is None:
        return []
    stamp = (st["built_at"], st["value_count"], _signature(cfg))
    cached = _cache.get(knowledge_id)
    if cached and cached[0] == stamp:
        return cached[1]
    permitted = _permitted(
        cfg
    )  # Verteidigung in der Tiefe: nie ausliefern, was die Konfiguration nicht erlaubt
    db = get_db_service().get_session()
    try:
        rows = [
            text.entry(r.table_name, r.column_name, r.value, r.count)
            for r in db.query(SqlValue).filter_by(knowledge_id=knowledge_id)
            if permitted(r.table_name, r.column_name)
        ]
    finally:
        db.close()
    _cache[knowledge_id] = (stamp, rows)
    return rows


def has_values(knowledge_id: int) -> bool:
    return status(knowledge_id)["value_count"] > 0


def search(
    knowledge_id: int,
    term: str,
    table: str | None = None,
    column: str | None = None,
    limit: int | None = None,
):
    _, max_results = worker_config.values_settings()
    return text.rank(
        term,
        entries(knowledge_id),
        limit=limit or max_results,
        table=table,
        column=column,
    )
