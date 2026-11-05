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

"""Chat-Kern: eine Frage vorbereiten und die Antwort des Agenten als Events streamen.

* :func:`prepare_chat` (synchron, im Threadpool): Session, Modell und Rechte, Knowledge-Scope,
  Hinweise, Compaction, PII-Maskierung des ganzen Kontexts, Bilder, Agent.
* :func:`stream_chat` (async, im Event-Loop des Servers): liefert Event-Dicts
  (``token``, ``reasoning``, ``reasoning_done``, ``tool_start``, ``tool_end``, ``tool_result``,
  ``source_list``, ``warning``, ``compaction``, ``done``, ``error``). Gespeichert wird nur bei ``done``.
  Wird der Task abgebrochen (Stopp) oder scheitert der Lauf, wird nichts gespeichert und eine in
  diesem Request neu angelegte Session ohne Nachrichten und Anhaenge wieder geloescht.

Der Transport (WebSocket) liegt in ``app/api/chat.py``.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app import resources_ready, wait_for_resources
from app.agent.document_agent import CONVERSATION_ONLY_HINT, DocumentAgent, select_tools
from app.agent.tools.web_access import build_web_search_tool, pii_client
from app.auth.access import accessible_knowledge_ids, can_use_model
from app.auth.users import role_at_least
from app.chat import compaction
from app.chat.pii_client import PIIFilterUnavailable
from app.chat.pii_session import (
    PIISession,
    StreamDemasker,
    mask_conversation,
    set_session,
    unavailable_message,
)
from app.core import runtime
from app.core.database_service import get_db_service
from app.core.phase_logger import Phase, phase_logger
from app.knowledge.index_doc import ensure_index
from app.knowledge.service import knowledge_paths, list_knowledge
from app.llm.registry import ModelUnavailableError, get_agent, resolve_model
from app.rag import model_settings
from app.rag import sparse as sparse_lang
from app.rag.retrieval import RetrievalScope, set_scope

logger = logging.getLogger(__name__)

# Werkzeug-Ergebnisse im gespeicherten Verlauf: hoechstens so viele Aufrufe, je so viele Zeichen
MAX_TOOL_BLOCKS = 20
MAX_TOOL_CHARS = 2000


class ChatError(Exception):
    """Fehler vor dem Agent-Lauf; die Meldung geht als ``error``-Event an den Nutzer."""


def _tool_text(output) -> str:
    """Text eines Werkzeug-Ergebnisses (nicht die ``repr`` der ToolMessage mit name=/tool_call_id=)."""
    content = getattr(output, "content", output)
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content if content is not None else "")


def _tool_input_text(tool_input) -> str:
    if isinstance(tool_input, dict):
        parts = [f"{k}: {v}" for k, v in tool_input.items() if v not in ("", None)]
        return "\n".join(parts)[:MAX_TOOL_CHARS]
    return str(tool_input or "")[:MAX_TOOL_CHARS]


def tools_block(tool_calls: list[dict]) -> str:
    """``<tools>[json]</tools>`` fuer den Verlauf: maskiert, gekuerzt, hoechstens ``MAX_TOOL_BLOCKS`` Aufrufe."""
    rows = [
        {
            "name": tc["name"],
            "input": _tool_input_text(tc.get("input")),
            "output": (tc.get("output") or "")[:MAX_TOOL_CHARS],
        }
        for tc in tool_calls[:MAX_TOOL_BLOCKS]
    ]
    if not rows:
        return ""
    return (
        "\n\n<tools>"
        + json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
        + "</tools>"
    )


@dataclass
class ChatContext:
    session_id: str
    created: bool  # Session in diesem Request neu angelegt (Abbruch -> ggf. wieder loeschen)
    user: dict
    user_message: str
    resolved: Any = None
    agent: Any = None
    input_data: dict = field(default_factory=dict)
    scope: Any = None
    pii_filter: Any = None
    pii_session: Any = None
    compaction_note: str | None = None
    started: float = field(default_factory=time.time)


def discard_if_empty(ctx_or_session: "ChatContext | tuple[str, bool]") -> bool:
    """Neu angelegte Session ohne Nachrichten und Anhaenge loeschen (Abbruch, Fehler)."""
    if isinstance(ctx_or_session, ChatContext):
        session_id, created = ctx_or_session.session_id, ctx_or_session.created
    else:
        session_id, created = ctx_or_session
    if not created:
        return False
    db = get_db_service()
    if db.get_messages(session_id, limit=1) or db.list_session_attachments(session_id):
        return False
    return db.delete_session(session_id)


def prepare_chat(user: dict, payload: dict) -> ChatContext:
    """Alles bis zum Agent-Aufruf. Fehler vor dem Lauf: :class:`ChatError` (Session aufgeraeumt)."""
    user_message = payload.get("message") or ""
    if not user_message:
        raise ChatError("Keine Nachricht")
    db = get_db_service()
    session_id = payload.get("session_id")
    if session_id:
        sess = db.get_chat_session(session_id, user_id=user["id"])
        if not sess:
            raise ChatError("Ungültige Session")
        created = False
    else:
        sess = db.create_session(user_id=user["id"])
        session_id, created = sess.id, True
    try:
        return _prepare(user, payload, user_message, sess, session_id, created)
    except BaseException:
        discard_if_empty((session_id, created))
        raise


def _prepare(user, payload, user_message, sess, session_id, created) -> ChatContext:
    db = get_db_service()
    # None = alle Knowledge-Bereiche; Liste = nur diese (leer = keiner).
    knowledge_ids = payload.get("knowledge_ids")
    attachment_ids = payload.get("attachment_ids") or []

    if not sess.title:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        db.update_session_title(session_id, title)

    ctx = ChatContext(session_id=session_id, created=created, user=user, user_message=user_message)
    phase_logger.log_phase(Phase.USER_INPUT, f"Chat-Anfrage | {len(user_message)} Zeichen")

    if not resources_ready():
        try:
            ready = wait_for_resources(timeout=180)
        except RuntimeError as e:
            raise ChatError(str(e)) from e
        if not ready:
            raise ChatError(
                "RAG-System wird noch initialisiert (Embedding-Modell lädt). "
                "Bitte in wenigen Sekunden erneut versuchen."
            )

    # Modell des Requests (oder Standardmodell) -> eigener Agent mit Prompt,
    # Parametern und ggf. angehaengtem Knowledge. Ohne Modell gibt es keinen Chat.
    model_ref = payload.get("model")
    # Knowledge, das das Modell selbst mitbringt (vom Admin kuratiert): wer das
    # Modell nutzen darf, darf es durchsuchen. Sonst gelten die Rechte des Nutzers.
    model_base: set[int] | None = None
    model_allows_extra = True
    try:
        resolved = resolve_model(model_ref)
        if not can_use_model(user, resolved.ref):
            raise ChatError("Kein Zugriff auf dieses Modell.")
        tools = list(runtime.state.agent_tools)
        # Websuche nur, wenn das Modell sie hat UND SearXNG samt PII-Filter eingerichtet sind.
        if resolved.web_search:
            web_tool = build_web_search_tool()
            if web_tool:
                tools.append(web_tool)
        if resolved.knowledge_ids:
            model_base = set(resolved.knowledge_ids)
            model_allows_extra = resolved.allow_extra_knowledge
    except ModelUnavailableError as e:
        raise ChatError(
            f"{e} Bitte in der Administration eine Verbindung anlegen und ein Standardmodell wählen."
            if not model_ref
            else str(e)
        ) from e
    ctx.resolved = resolved

    # Verlauf: gespeicherte Zusammenfassung (falls vorhanden) plus alle Nachrichten danach.
    # Passt das nicht ins Eingabe-Budget des Modells, fasst die Compaction weiter unten zusammen.
    summary_text, history_rows = db.get_history(session_id)
    message_list = compaction.history_messages(history_rows, summary_text)

    # PII-Filter: None, wenn nicht eingerichtet oder abgeschaltet. Maskiert wird weiter unten der
    # GANZE Kontext (Zusammenfassung, Verlauf, neue Frage) in einem Aufruf, nach der Compaction.
    pii_filter = pii_client()
    pii_mapping = []  # Mapping fuer die Demaskierung der Antwort

    # Hinweis auf Chat-Anhaenge (nur fuer den Agent, wird nicht gespeichert),
    # damit er die Dokumentsuche auch tatsaechlich darauf anwendet.
    attached = db.list_session_attachments(session_id, status="ready")
    hints = ""
    if attached:
        names = ", ".join(f"„{a.filename}“" for a in attached)
        hints += (
            f"\n\n[Hinweis: Der Nutzer hat in diesem Chat folgende Dateien "
            f"angehängt: {names}. Ihr Inhalt ist über document_search_tool "
            f"durchsuchbar.]"
        )

    message_list.append({"role": "user", "content": user_message})

    # Durchsuchte Collections dieses Requests: gewaehlte Knowledge-Bereiche plus
    # (falls vorhanden) die Anhaenge dieser Session.
    all_knowledge = list_knowledge()
    # Nie Collections aus dem Request ungeprueft uebernehmen.
    readable = accessible_knowledge_ids(user, "read")
    wanted = (
        {int(i) for i in knowledge_ids if str(i).lstrip("-").isdigit()}
        if knowledge_ids is not None
        else None
    )
    if model_base is not None:
        # Das Modell-Knowledge ist die Basis (wer das Modell nutzen darf, darf es durchsuchen, auch
        # ohne eigene Freigabe). Dazu kommt, was der Nutzer selbst waehlt und lesen darf.
        extra = (wanted & readable) if (wanted is not None and model_allows_extra) else set()
        allowed = model_base | extra
        all_knowledge = [k for k in all_knowledge if k["id"] in allowed]
    else:
        all_knowledge = [k for k in all_knowledge if k["id"] in readable]
        if wanted is not None:
            all_knowledge = [k for k in all_knowledge if k["id"] in wanted]
    # SQL-Quellen haben keine Collection und keine Dateien: eigener Weg ueber die SQL-Werkzeuge
    sql_knowledge = [k for k in all_knowledge if k.get("kind") == "sql"]
    all_knowledge = [k for k in all_knowledge if k.get("kind") != "sql"]
    from app.auth.access import user_group_names
    from app.knowledge.index_doc import inline_hint
    from app.rag.helper import make_helper
    from app.rag.profiles import config_for_knowledge

    scope = RetrievalScope(
        user_id=user["id"],
        identity={
            "username": user.get("username", ""),
            "display_name": user.get("display_name", ""),
            "groups": user_group_names(user["id"]),
            "external_groups": user.get("external_groups") or [],
            "is_admin": role_at_least(user["role"], "admin"),
        },
        collections=tuple(k["collection_name"] for k in all_knowledge),
        session_id=session_id,
        # Jedes Knowledge sucht mit seinem Retrieval-Profil (sonst Standardprofil).
        profiles=tuple((k["collection_name"], config_for_knowledge(k)) for k in all_knowledge),
        # BM25-Sprache je Collection: jede Collection wird in der Sprache gefragt, mit der sie indexiert wurde
        sparse_langs=tuple(
            (k["collection_name"], sparse_lang.knowledge_language(k)) for k in all_knowledge
        ),
        helper=make_helper(resolved, user),
        # pandas-Tools sehen nur die strukturierten Dateien dieser Knowledge-Bereiche
        data_dirs=tuple(
            (paths.name_prefix, str(paths.data_dir))
            for paths in (knowledge_paths(k) for k in all_knowledge)
        ),
        # Inhaltsverzeichnis hängt am Knowledge-Eintrag: genau das der gewählten Knowledge, nie ein fremdes
        indexes=tuple((k["name"], ensure_index(k) or "") for k in all_knowledge),
        knowledge_names=tuple((k["collection_name"], k["name"]) for k in all_knowledge),
        # fremde Vektorspeicher: Konfiguration (inkl. Key, bleibt im Prozess) je Knowledge
        externals=tuple(
            (k["collection_name"], k["source_config"])
            for k in all_knowledge
            if k.get("kind") == "external" and k.get("source_config")
        ),
        sql_sources=tuple(
            (k["id"], k["name"], k["source_config"])
            for k in sql_knowledge
            if k.get("source_config")
        ),
        # jedes Knowledge wird mit dem Modell abgefragt, mit dem es indexiert wurde
        specs=tuple(
            (k["collection_name"], json.dumps(model_settings.knowledge_spec(k).to_dict()))
            for k in all_knowledge
        ),
    )
    ctx.scope = scope
    # Kleines Inhaltsverzeichnis gleich mitgeben ("zuerst lesen"), grosses nur per Werkzeug.
    hint = inline_hint(scope.indexes)
    if hint:
        hints += hint
    has_sql = bool(scope.sql_sources)
    if has_sql:
        from app.agent.toolworker import config as worker_config

        names = ", ".join(f"„{name}“" for _id, name, _cfg in scope.sql_sources)
        hints += "\n\n" + worker_config.sql_prompt().replace("{sources}", names)
        from app.knowledge.sql import valueindex

        if any(valueindex.has_values(kid) for kid, _name, _cfg in scope.sql_sources):
            hints += "\n\n" + worker_config.values_hint()

    # Ohne Wissen und ohne Anhaenge gibt es nichts zu durchsuchen: dann bekommt der Agent keine
    # Wissens-Werkzeuge und beantwortet/fasst nur aus dem Gespraech zusammen.
    has_knowledge = bool(scope.collections or scope.externals or scope.data_dirs or attached)
    tools = select_tools(tools, has_knowledge, has_sql)
    if not has_knowledge and not has_sql:
        message_list[-1]["content"] += CONVERSATION_ONLY_HINT
    try:
        ctx.agent = get_agent(resolved, tools)
    except ModelUnavailableError as e:
        raise ChatError(str(e)) from e

    # Automatische Compaction: Eingabe-Budget = num_ctx - max_tokens, Zusammenfassen ab 95 %.
    try:
        cfg = compaction.load_settings()
        budget = compaction.compute_budget(compaction.effective_params(resolved.params), cfg)
        if budget:
            overhead = compaction.estimate_tokens(
                DocumentAgent.compose_prompt(resolved.system_prompt)
            ) + compaction.tools_tokens(tools)
            new_message = message_list[-1]
            history, ctx.compaction_note = compaction.fit_history(
                history_rows,
                summary_text,
                new_message["content"] + hints,
                overhead,
                budget,
                cfg,
                lambda: compaction.summary_llm(resolved, budget),
                lambda text, through: db.set_session_summary(session_id, text, through),
                # Mit PII-Filter sieht auch das Zusammenfassungs-Modell nur maskierten Text
                mask_many=(lambda texts: mask_conversation(pii_filter, texts))
                if pii_filter
                else None,
                unmask=(lambda text, mapping: pii_filter.demask(text, mapping))
                if pii_filter
                else None,
            )
            message_list = history + [new_message]
    except PIIFilterUnavailable:
        pass  # der Maskierungsschritt unten meldet es dem Nutzer
    except Exception as e:  # nie am Chat scheitern lassen: ohne Compaction weiter
        logger.warning(f"Compaction fehlgeschlagen: {e}")

    # PII-Filter, fail-closed: der GANZE Kontext (Zusammenfassung, Verlauf, neue Frage) wird in einem
    # Aufruf maskiert, der Dienst vergibt dabei eindeutige Platzhalter. Ist er nicht erreichbar oder
    # zerstoert er die Struktur, geht NICHTS ans Modell; der Nutzer bekommt eine Fehlermeldung.
    if pii_filter:
        try:
            masked, pii_mapping = mask_conversation(
                pii_filter, [m["content"] for m in message_list]
            )
            message_list = [
                {"role": m["role"], "content": t} for m, t in zip(message_list, masked)
            ]
            if pii_mapping:
                logger.info(f"PII-Filter: {len(pii_mapping)} Entitäten im Kontext maskiert")
        except PIIFilterUnavailable as e:
            logger.warning(f"PII-Filter nicht verfügbar ({e}) - Nachricht wird nicht gesendet")
            raise ChatError(
                f"{unavailable_message(e)}. Die Nachricht wurde nicht "
                "gesendet, damit keine personenbezogenen Daten ungeschützt das System "
                "verlassen. Bitte den Administrator informieren oder den PII-Filter "
                "unter Administration › Tools prüfen (der Test dort prüft auch einen langen Text)."
            ) from e
    ctx.pii_filter = pii_filter
    # Alle Platzhalter dieses Requests (Kontext, Werkzeug-Ergebnisse, Antwort) liegen in einer Sitzung
    ctx.pii_session = PIISession(pii_filter, pii_mapping)
    message_list[-1]["content"] += hints  # Hinweise (Anhaenge, Inhaltsverzeichnis) sind kein Nutzertext
    # Versteht das Modell Bilder, sieht es die in dieser Frage angehaengten Bilder selbst (zusaetzlich
    # zum gelesenen Text). Der PII-Filter kann Bilder nicht schwaerzen; die Modellwahl des Admins
    # entscheidet, wer sie bekommt. Erst hier angehaengt, damit Maskierung und Compaction nur Text sehen.
    if resolved.vision and attachment_ids:
        from app.chat.attachments import image_data_url

        urls = []
        for attachment_id in dict.fromkeys(str(i) for i in attachment_ids):
            row = db.get_attachment(attachment_id)
            if row and row.session_id == session_id and row.user_id == user["id"]:
                url = image_data_url(session_id, attachment_id)
                if url:
                    urls.append(url)
        if urls:
            message_list[-1]["content"] = [
                {"type": "text", "text": message_list[-1]["content"]}
            ] + [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    ctx.input_data = {"messages": message_list}
    return ctx


async def stream_chat(ctx: ChatContext):
    """Agent laufen lassen und Events liefern. Abbruch = Task canceln (nichts wird gespeichert)."""
    scope, pii_session = ctx.scope, ctx.pii_session
    full_response = ""
    full_reasoning = ""
    tool_calls: list[dict] = []
    sources_sent = 0
    # Live-Demaskierung der gestreamten Antwort und des Reasonings
    token_demasker = StreamDemasker(pii_session)
    reasoning_demasker = StreamDemasker(pii_session)
    unmasking = bool(ctx.pii_filter)
    finished = False
    try:
        if ctx.compaction_note:
            yield {"compaction": ctx.compaction_note}
        phase_logger.log_phase(Phase.AGENT_START, "Agent-Stream gestartet")
        # Scope im Kontext dieses Tasks setzen; asyncio und die LangChain-Executors reichen ihn
        # an die Tool-Aufrufe weiter.
        set_scope(scope)
        set_session(pii_session)
        async for event in ctx.agent.astream_events(ctx.input_data, version="v2"):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = chunk.content
                reasoning = None
                if getattr(chunk, "additional_kwargs", None):
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                if content:
                    full_response += content
                    out = token_demasker.feed(content)
                    if out:
                        yield {"token": out}
                if reasoning:
                    full_reasoning += reasoning
                    out = reasoning_demasker.feed(reasoning)
                    if out:
                        yield {"reasoning": out}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                phase_logger.log_phase(Phase.TOOL_EXECUTION, f"Tool gestartet: {tool_name}")
                tool_calls.append(
                    {
                        "name": tool_name,
                        "input": event.get("data", {}).get("input", {}),
                        "output": None,
                    }
                )
                yield {"tool_start": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                tool_output = event.get("data", {}).get("output", "")
                phase_logger.log_phase(Phase.TOOL_EXECUTION, f"Tool beendet: {tool_name}")
                yield {"tool_end": tool_name}
                fresh = scope.sources[sources_sent:]
                if fresh:
                    sources_sent += len(fresh)
                    yield {"source_list": list(fresh)}
                while scope.warnings:
                    yield {"warning": scope.warnings.pop(0)}
                text = _tool_text(tool_output)[:MAX_TOOL_CHARS]
                shown_input = ""
                for tc in tool_calls:
                    if tc["name"] == tool_name and tc["output"] is None:
                        tc["output"] = text  # maskiert, wie das Modell es sah (so landet es im Verlauf)
                        shown_input = _tool_input_text(tc.get("input"))
                        break
                if unmasking and pii_session.pairs():
                    # Der Nutzer sieht echte Werte wie in der Antwort; gespeichert wird die maskierte Fassung
                    text, shown_input = pii_session.unmask(text), pii_session.unmask(shown_input)
                yield {"tool_result": {"name": tool_name, "output": text, "input": shown_input}}

        used = ", ".join(sorted({tc["name"] for tc in tool_calls})) or "-"
        phase_logger.log_phase(
            Phase.AGENT_RESPONSE,
            f"Chat fertig | Modell: {ctx.resolved.ref} | Tools: {used} | Antwort: {len(full_response)} Zeichen",
            duration=time.time() - ctx.started,
            level=logging.INFO,
        )
        # Signal an Frontend: Thinking abgeschlossen (zum Zuklappen des offenen Blocks).
        rest = reasoning_demasker.flush()
        if rest:
            yield {"reasoning": rest}
        yield {"reasoning_done": True}
        rest = token_demasker.flush()
        if rest:
            yield {"token": rest}
        await asyncio.to_thread(_save, ctx, full_response, full_reasoning, tool_calls)
        finished = True
        yield {"done": True, "session_id": ctx.session_id}
    except asyncio.CancelledError:
        phase_logger.log_phase(Phase.AGENT_RESPONSE, "Antwort vom Benutzer abgebrochen")
        raise
    except Exception as e:
        logger.exception("Chat-Lauf fehlgeschlagen")
        phase_logger.log_phase(
            Phase.AGENT_RESPONSE,
            f"Fehler: {type(e).__name__}",
            duration=0.0,
            level=logging.ERROR,
        )
        yield {"error": str(e)}
    finally:
        if not finished:
            try:
                discard_if_empty(ctx)
            except Exception as e:
                logger.warning(f"Leere Session konnte nicht entfernt werden: {e}")


def _save(ctx: ChatContext, full_response: str, full_reasoning: str, tool_calls: list) -> None:
    """Frage und Antwort speichern (nur nach ``done``); die Antwort demaskiert, Werkzeuge maskiert."""
    pii_session = ctx.pii_session
    unmasking = bool(ctx.pii_filter) and bool(pii_session.pairs())
    demasked_response = full_response
    if unmasking:
        try:
            demasked_response = pii_session.unmask(full_response)
            if demasked_response != full_response:
                logger.info(f"PII-Demaskierung: {len(pii_session.pairs())} Platzhalter ersetzt")
        except Exception as e:
            logger.warning(f"PII-Demaskierung fehlgeschlagen: {e}")
    db = get_db_service()
    db.save_message(ctx.session_id, "user", ctx.user_message)
    # Reasoning als <think>-Tags vor die Antwort speichern, damit der Denk-Block beim Neuladen
    # der History wieder als ausklappbarer Block erscheint.
    saved = demasked_response
    if full_reasoning:
        reasoning_text = full_reasoning.strip()
        if unmasking:
            try:
                reasoning_text = pii_session.unmask(reasoning_text)
            except Exception as e:
                logger.warning(f"PII-Demaskierung (Reasoning) fehlgeschlagen: {e}")
        saved = f"<think>{reasoning_text}</think>\n" + demasked_response
    saved += tools_block(tool_calls)  # Werkzeug-Aufrufe (maskiert) fuer den Verlauf
    if ctx.scope.sources:  # Quellen mit Score fuer die Anzeige im Verlauf
        saved += (
            "\n\n<sources>" + json.dumps(ctx.scope.sources, ensure_ascii=False) + "</sources>"
        )
    db.save_message(ctx.session_id, "assistant", saved)
