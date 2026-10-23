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

"""Chat: Stream (/chat), Abbruch, Sitzungen, Verlauf, Anhaenge und Spracheingabe."""

import asyncio
import json
import logging
import queue
import threading
import time
from logging import getLogger

from flask import (
    Blueprint,
    Response,
    current_app,
    jsonify,
    request,
)

from app import resources_ready, wait_for_resources
from app.agent.document_agent import CONVERSATION_ONLY_HINT, DocumentAgent, select_tools
from app.agent.tools.web_access import build_web_search_tool, pii_client
from app.auth.access import (
    accessible_knowledge_ids,
    can_use_model,
)
from app.auth.auth import auth_required_api, current_user
from app.auth.users import (
    role_at_least,
)
from app.chat import compaction
from app.chat.pii_client import PIIFilterUnavailable
from app.chat.pii_session import (
    PIISession,
    StreamDemasker,
    mask_conversation,
    set_session,
    unavailable_message,
)
from app.core.database_service import get_db_service
from app.core.phase_logger import Phase, phase_logger
from app.knowledge.index_doc import ensure_index
from app.knowledge.service import (
    knowledge_paths,
    list_knowledge,
)
from app.llm.registry import ModelUnavailableError, get_agent, resolve_model
from app.rag import model_settings
from app.rag import sparse as sparse_lang
from app.rag.retrieval import RetrievalScope, set_scope

bp = Blueprint("chat", __name__)
logger = getLogger(__name__)


# Globale Cancel-Events für laufende Chat-Streams, keyed by session_id.
_cancel_events = {}


_cancel_events_lock = threading.Lock()


def _get_cancel_event(session_id):
    with _cancel_events_lock:
        if session_id not in _cancel_events:
            _cancel_events[session_id] = threading.Event()
        return _cancel_events[session_id]


def _set_cancelled(session_id):
    with _cancel_events_lock:
        event = _cancel_events.get(session_id)
        if event is None:
            event = threading.Event()
            event.set()
            _cancel_events[session_id] = event
        else:
            event.set()


def _clear_cancelled(session_id):
    with _cancel_events_lock:
        event = _cancel_events.get(session_id)
        if event is not None:
            event.clear()


@bp.route("/api/sessions", methods=["GET"])
@auth_required_api
def get_sessions():
    """List all sessions"""
    db = get_db_service()
    sessions = db.list_sessions(user_id=current_user()["id"])

    result = []
    for sess in sessions:
        result.append(
            {
                "id": sess.id,
                "title": sess.title,
                "created_at": sess.created_at.isoformat() if sess.created_at else None,
                "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
            }
        )

    return jsonify(result)


@bp.route("/api/sessions", methods=["POST"])
@auth_required_api
def new_session():
    """Create a new session"""
    db = get_db_service()
    new_sess = db.create_session(user_id=current_user()["id"])

    return jsonify(
        {
            "id": new_sess.id,
            "title": new_sess.title,
            "created_at": new_sess.created_at.isoformat()
            if new_sess.created_at
            else None,
            "updated_at": new_sess.updated_at.isoformat()
            if new_sess.updated_at
            else None,
        }
    )


@bp.route("/api/sessions/<session_id>", methods=["GET"])
@auth_required_api
def get_session_messages(session_id):
    """Get session with messages"""
    db = get_db_service()
    sess = db.get_chat_session(session_id, user_id=current_user()["id"])

    if not sess:
        return jsonify({"error": "Session nicht gefunden"}), 404

    messages = db.get_messages(session_id)

    message_dicts = []
    for msg in messages:
        message_dicts.append(
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        )

    return jsonify(
        {
            "id": sess.id,
            "title": sess.title,
            "created_at": sess.created_at.isoformat() if sess.created_at else None,
            "updated_at": sess.updated_at.isoformat() if sess.updated_at else None,
            "messages": message_dicts,
        }
    )


@bp.route("/api/sessions/<session_id>", methods=["DELETE"])
@auth_required_api
def delete_session_route(session_id):
    """Delete a session"""
    db = get_db_service()
    deleted = db.delete_session(session_id, user_id=current_user()["id"])

    if not deleted:
        return jsonify({"error": "Session nicht gefunden"}), 404

    from app.chat.attachments import cleanup_session_attachments

    cleanup_session_attachments(session_id, current_app.extensions.get("client"))

    return jsonify({"success": True})


@bp.route("/history/<session_id>", methods=["GET"])
@auth_required_api
def history(session_id):
    db = get_db_service()
    sess = db.get_chat_session(session_id, user_id=current_user()["id"])

    if not sess:
        return {"error": "Session nicht gefunden"}, 404

    messages = db.get_messages(session_id)

    message_dicts = []
    for msg in messages:
        message_dicts.append(
            {
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if msg.created_at else None,
            }
        )

    return {"history": message_dicts}


MAX_TOOL_BLOCKS = 20


MAX_TOOL_CHARS = 2000


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


@bp.route("/chat", methods=["POST"])
@auth_required_api
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id")
    # None = alle Knowledge-Bereiche; Liste = nur diese (leer = keiner).
    knowledge_ids = data.get("knowledge_ids")
    attachment_ids = (
        data.get("attachment_ids")
        if isinstance(data.get("attachment_ids"), list)
        else []
    )

    if not user_message:
        return Response(
            'data: {"error": "Keine Nachricht"}\n\n', mimetype="text/event-stream"
        )

    db = get_db_service()
    final_session_id = session_id
    sess = None

    if session_id:
        sess = db.get_chat_session(session_id, user_id=current_user()["id"])
        if not sess:
            return Response(
                'data: {"error": "Ungültige Session"}\n\n', mimetype="text/event-stream"
            )
    else:
        new_sess = db.create_session(user_id=current_user()["id"])
        final_session_id = new_sess.id
        sess = new_sess

    if not sess.title:
        title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        db.update_session_title(final_session_id, title)

    chat_started = time.time()
    phase_logger.log_phase(
        Phase.USER_INPUT, f"Chat-Anfrage | {len(user_message)} Zeichen"
    )

    if not resources_ready():
        try:
            ready = wait_for_resources(timeout=180)
        except RuntimeError as e:
            return Response(
                f'data: {{"error": "{e}"}}\n\n', mimetype="text/event-stream"
            )
        if not ready:
            return Response(
                'data: {"error": "RAG-System wird noch initialisiert '
                "(Embedding-Modell lädt). Bitte in wenigen Sekunden "
                'erneut versuchen."}\n\n',
                mimetype="text/event-stream",
            )

    # Modell des Requests (oder Standardmodell) -> eigener Agent mit Prompt,
    # Parametern und ggf. angehaengtem Knowledge. Ohne Modell gibt es keinen Chat.
    model_ref = data.get("model")
    user = current_user()
    # Knowledge, das das Modell selbst mitbringt (vom Admin kuratiert): wer das
    # Modell nutzen darf, darf es durchsuchen. Sonst gelten die Rechte des Nutzers.
    model_base: set[int] | None = (
        None  # Knowledge, das das Modell mitbringt (Basis jeder Suche)
    )
    model_allows_extra = True
    resolved = None
    try:
        resolved = resolve_model(model_ref)
        if not can_use_model(user, resolved.ref):
            return Response(
                'data: {"error": "Kein Zugriff auf dieses Modell."}\n\n',
                mimetype="text/event-stream",
            )
        tools = list(current_app.extensions["agent_tools"])
        # Websuche nur, wenn das Modell sie hat UND SearXNG samt PII-Filter eingerichtet sind.
        if resolved.web_search:
            web_tool = build_web_search_tool()
            if web_tool:
                tools.append(web_tool)
        if resolved.knowledge_ids:
            model_base = set(resolved.knowledge_ids)
            model_allows_extra = resolved.allow_extra_knowledge
    except ModelUnavailableError as e:
        message = (
            f"{e} Bitte in der Administration eine Verbindung anlegen und ein Standardmodell wählen."
            if not model_ref
            else str(e)
        )
        return Response(
            f"data: {json.dumps({'error': message})}\n\n",
            mimetype="text/event-stream",
        )

    # Verlauf: gespeicherte Zusammenfassung (falls vorhanden) plus alle Nachrichten danach.
    # Passt das nicht ins Eingabe-Budget des Modells, fasst die Compaction weiter unten zusammen.
    summary_text, history_rows = db.get_history(final_session_id)
    message_list = compaction.history_messages(history_rows, summary_text)

    # PII-Filter: None, wenn nicht eingerichtet oder abgeschaltet. Maskiert wird weiter unten der
    # GANZE Kontext (Zusammenfassung, Verlauf, neue Frage) in einem Aufruf, nach der Compaction.
    pii_filter = pii_client()
    pii_mapping = []  # Mapping fuer die Demaskierung der Antwort

    # Hinweis auf Chat-Anhaenge (nur fuer den Agent, wird nicht gespeichert),
    # damit er die Dokumentsuche auch tatsaechlich darauf anwendet.
    attached = db.list_session_attachments(final_session_id, status="ready")
    hints = ""
    if attached:
        names = ", ".join(f"„{a.filename}“" for a in attached)
        hints += (
            f"\n\n[Hinweis: Der Nutzer hat in diesem Chat folgende Dateien "
            f"angehängt: {names}. Ihr Inhalt ist über document_search_tool "
            f"durchsuchbar.]"
        )

    message_list.append({"role": "user", "content": user_message})

    logger.info(f"[DEBUG] Messages an Agent: {len(message_list)} Nachrichten")

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
        extra = (
            (wanted & readable)
            if (wanted is not None and model_allows_extra)
            else set()
        )
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
        session_id=final_session_id,
        # Jedes Knowledge sucht mit seinem Retrieval-Profil (sonst Standardprofil).
        profiles=tuple(
            (k["collection_name"], config_for_knowledge(k)) for k in all_knowledge
        ),
        # BM25-Sprache je Collection: jede Collection wird in der Sprache gefragt, mit der sie indexiert wurde
        sparse_langs=tuple(
            (k["collection_name"], sparse_lang.knowledge_language(k))
            for k in all_knowledge
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
            (
                k["collection_name"],
                json.dumps(model_settings.knowledge_spec(k).to_dict()),
            )
            for k in all_knowledge
        ),
    )
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
    has_knowledge = bool(
        scope.collections or scope.externals or scope.data_dirs or attached
    )
    tools = select_tools(tools, has_knowledge, has_sql)
    if not has_knowledge and not has_sql:
        message_list[-1]["content"] += CONVERSATION_ONLY_HINT
    try:
        agent = get_agent(resolved, tools)
    except ModelUnavailableError as e:
        return Response(
            f"data: {json.dumps({'error': str(e)})}\n\n",
            mimetype="text/event-stream",
        )

    # Automatische Compaction: Eingabe-Budget = num_ctx - max_tokens, Zusammenfassen ab 95 %.
    compaction_note = None
    try:
        cfg = compaction.load_settings()
        budget = compaction.compute_budget(
            compaction.effective_params(resolved.params), cfg
        )
        if budget:
            overhead = compaction.estimate_tokens(
                DocumentAgent.compose_prompt(resolved.system_prompt)
            ) + compaction.tools_tokens(tools)
            new_message = message_list[-1]
            history, compaction_note = compaction.fit_history(
                history_rows,
                summary_text,
                new_message["content"] + hints,
                overhead,
                budget,
                cfg,
                lambda: compaction.summary_llm(resolved, budget),
                lambda text, through: db.set_session_summary(
                    final_session_id, text, through
                ),
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
                logger.info(
                    f"PII-Filter: {len(pii_mapping)} Entitäten im Kontext maskiert"
                )
        except PIIFilterUnavailable as e:
            logger.warning(
                f"PII-Filter nicht verfügbar ({e}) - Nachricht wird nicht gesendet"
            )
            return Response(
                "data: "
                + json.dumps(
                    {
                        "error": f"{unavailable_message(e)}. Die Nachricht wurde nicht "
                        "gesendet, damit keine personenbezogenen Daten ungeschützt das System "
                        "verlassen. Bitte den Administrator informieren oder den PII-Filter "
                        "unter Administration › Tools prüfen (der Test dort prüft auch einen langen Text)."
                    }
                )
                + "\n\n",
                mimetype="text/event-stream",
            )
    # Alle Platzhalter dieses Requests (Kontext, Werkzeug-Ergebnisse, Antwort) liegen in einer Sitzung
    pii_session = PIISession(pii_filter, pii_mapping)
    message_list[-1]["content"] += (
        hints  # Hinweise (Anhaenge, Inhaltsverzeichnis) sind kein Nutzertext
    )
    # Versteht das Modell Bilder, sieht es die in dieser Frage angehaengten Bilder selbst (zusaetzlich
    # zum gelesenen Text). Der PII-Filter kann Bilder nicht schwaerzen; die Modellwahl des Admins
    # entscheidet, wer sie bekommt. Erst hier angehaengt, damit Maskierung und Compaction nur Text sehen.
    if resolved.vision and attachment_ids:
        from app.chat.attachments import image_data_url

        user_id = user["id"]
        urls = []
        for attachment_id in dict.fromkeys(str(i) for i in attachment_ids):
            row = db.get_attachment(attachment_id)
            if row and row.session_id == final_session_id and row.user_id == user_id:
                url = image_data_url(final_session_id, attachment_id)
                if url:
                    urls.append(url)
        if urls:
            message_list[-1]["content"] = [
                {"type": "text", "text": message_list[-1]["content"]}
            ] + [{"type": "image_url", "image_url": {"url": url}} for url in urls]
    input_data = {"messages": message_list}

    def generate():
        result_queue = queue.Queue()
        full_response = ""
        full_reasoning = ""
        tool_calls = []
        cancel_event = _get_cancel_event(final_session_id)
        cancel_event.clear()

        # Mapping für Demaskierung in Closure verfügbar machen
        # Live-Demaskierung der gestreamten Antwort und des Reasonings
        token_demasker = StreamDemasker(pii_session)
        reasoning_demasker = StreamDemasker(pii_session)

        async def run_agent_async():
            nonlocal full_response
            nonlocal full_reasoning
            sources_sent = 0
            try:
                phase_logger.log_phase(Phase.AGENT_START, "Agent-Stream gestartet")
                # Scope im Kontext der Agent-Loop setzen; asyncio und die
                # LangChain-Executors reichen ihn an die Tool-Aufrufe weiter.
                set_scope(scope)
                set_session(pii_session)

                async for event in agent.astream_events(input_data, version="v2"):
                    if cancel_event.is_set():
                        result_queue.put(("cancelled", None))
                        break

                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        content = chunk.content

                        reasoning = None
                        if (
                            hasattr(chunk, "additional_kwargs")
                            and chunk.additional_kwargs
                        ):
                            reasoning = chunk.additional_kwargs.get("reasoning_content")

                        if content:
                            full_response += content
                            result_queue.put(("token", content))

                        if reasoning:
                            full_reasoning += reasoning
                            result_queue.put(("reasoning", reasoning))

                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", {})
                        phase_logger.log_phase(
                            Phase.TOOL_EXECUTION, f"Tool gestartet: {tool_name}"
                        )
                        result_queue.put(("tool_start", tool_name))
                        tool_calls.append(
                            {"name": tool_name, "input": tool_input, "output": None}
                        )

                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        phase_logger.log_phase(
                            Phase.TOOL_EXECUTION, f"Tool beendet: {tool_name}"
                        )
                        result_queue.put(("tool_end", tool_name))
                        fresh = scope.sources[sources_sent:]
                        if fresh:
                            sources_sent += len(fresh)
                            result_queue.put(("sources", list(fresh)))
                        while scope.warnings:
                            result_queue.put(("warning", scope.warnings.pop(0)))
                        text = _tool_text(tool_output)[:MAX_TOOL_CHARS]
                        shown_input = ""
                        if tool_calls:
                            for tc in tool_calls:
                                if tc["name"] == tool_name and tc["output"] is None:
                                    tc["output"] = (
                                        text  # maskiert, wie das Modell es sah (so landet es im Verlauf)
                                    )
                                    shown_input = _tool_input_text(tc.get("input"))
                                    break
                        result_queue.put(
                            ("tool_result", (tool_name, text, shown_input))
                        )

                if cancel_event.is_set():
                    phase_logger.log_phase(
                        Phase.AGENT_RESPONSE, "Antwort vom Benutzer abgebrochen"
                    )
                else:
                    used = ", ".join(sorted({tc["name"] for tc in tool_calls})) or "-"
                    phase_logger.log_phase(
                        Phase.AGENT_RESPONSE,
                        f"Chat fertig | Modell: {resolved.ref} | Tools: {used} | Antwort: {len(full_response)} Zeichen",
                        duration=time.time() - chat_started,
                        level=logging.INFO,
                    )
                    # Signal an Frontend: Thinking abgeschlossen (zum
                    # Zuklappen des offenen Blocks).
                    result_queue.put(("reasoning_done", None))
                    result_queue.put(("done", full_response))

            except Exception as e:
                import traceback

                traceback.print_exc()
                phase_logger.log_phase(
                    Phase.AGENT_RESPONSE,
                    f"Fehler: {type(e).__name__}",
                    duration=0.0,
                    level=logging.ERROR,
                )
                result_queue.put(("error", str(e)))

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_agent_async())
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

        if compaction_note:
            yield f"data: {json.dumps({'compaction': compaction_note})}\n\n"
        thread = threading.Thread(target=run_in_thread)
        thread.start()

        try:
            while True:
                msg_type, msg_data = result_queue.get()

                if msg_type == "done":
                    # Zurueckgehaltene Reste der Live-Demaskierung ausgeben
                    for rest, key in (
                        (reasoning_demasker.flush(), "reasoning"),
                        (token_demasker.flush(), "token"),
                    ):
                        if rest:
                            yield f"data: {json.dumps({key: rest})}\n\n"
                    # Antwort demaskieren bevor sie gespeichert wird
                    db = get_db_service()
                    demasked_response = msg_data
                    if pii_session.pairs() and pii_filter:
                        try:
                            demasked_response = pii_session.unmask(msg_data)
                            if demasked_response != msg_data:
                                logger.info(
                                    f"PII-Demaskierung: {len(pii_session.pairs())} Platzhalter ersetzt"
                                )
                        except Exception as e:
                            logger.warning(f"PII-Demaskierung fehlgeschlagen: {e}")

                    db.save_message(final_session_id, "user", user_message)

                    # Reasoning als <think>-Tags vor die Antwort speichern,
                    # damit der Denk-Block beim Neuladen der History wieder
                    # als ausklappbarer Block erscheint.
                    saved_assistant = demasked_response
                    if full_reasoning:
                        reasoning_text = full_reasoning.strip()
                        if pii_session.pairs() and pii_filter:
                            try:
                                reasoning_text = pii_session.unmask(reasoning_text)
                            except Exception as e:
                                logger.warning(
                                    f"PII-Demaskierung (Reasoning) fehlgeschlagen: {e}"
                                )
                        saved_assistant = (
                            f"<think>{reasoning_text}</think>\n" + demasked_response
                        )
                    saved_assistant += tools_block(
                        tool_calls
                    )  # Werkzeug-Aufrufe (maskiert) fuer den Verlauf
                    if scope.sources:  # Quellen mit Score fuer die Anzeige im Verlauf
                        saved_assistant += (
                            "\n\n<sources>"
                            + json.dumps(scope.sources, ensure_ascii=False)
                            + "</sources>"
                        )
                    db.save_message(final_session_id, "assistant", saved_assistant)
                    yield f"data: {json.dumps({'done': True, 'session_id': final_session_id})}\n\n"
                    break

                elif msg_type == "cancelled":
                    break

                elif msg_type == "reasoning_done":
                    rest = reasoning_demasker.flush()
                    if rest:
                        yield f"data: {json.dumps({'reasoning': rest})}\n\n"
                    yield f"data: {json.dumps({'reasoning_done': True})}\n\n"

                elif msg_type == "reasoning":
                    out = reasoning_demasker.feed(msg_data)
                    if out:
                        yield f"data: {json.dumps({'reasoning': out})}\n\n"

                elif msg_type == "token":
                    out = token_demasker.feed(msg_data)
                    if out:
                        yield f"data: {json.dumps({'token': out})}\n\n"

                elif msg_type == "tool_start":
                    yield f"data: {json.dumps({'tool_start': msg_data})}\n\n"

                elif msg_type == "tool_end":
                    yield f"data: {json.dumps({'tool_end': msg_data})}\n\n"

                elif msg_type == "warning":
                    yield f"data: {json.dumps({'warning': msg_data})}\n\n"

                elif msg_type == "sources":
                    yield f"data: {json.dumps({'source_list': msg_data})}\n\n"

                elif msg_type == "tool_result":
                    tool_name, tool_output, tool_input = msg_data
                    if pii_filter and pii_session.pairs():
                        # Der Nutzer sieht echte Werte wie in der Antwort; gespeichert wird die maskierte Fassung
                        tool_output, tool_input = (
                            pii_session.unmask(tool_output),
                            pii_session.unmask(tool_input),
                        )
                    payload = {
                        "tool_result": {
                            "name": tool_name,
                            "output": tool_output,
                            "input": tool_input,
                        }
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif msg_type == "error":
                    yield f"data: {json.dumps({'error': msg_data})}\n\n"
                    break
        except GeneratorExit:
            _set_cancelled(final_session_id)
        finally:
            thread.join(timeout=1)
            _clear_cancelled(final_session_id)

    return Response(generate(), mimetype="text/event-stream")


@bp.route("/chat/stop", methods=["POST"])
@auth_required_api
def stop_chat():
    """Bricht den laufenden Chat-Stream für eine Session ab."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    # Nur eigene Chats abbrechen (die Session-ID allein ist kein Berechtigungsnachweis).
    if session_id and get_db_service().get_chat_session(
        session_id, user_id=current_user()["id"]
    ):
        _set_cancelled(session_id)
    return jsonify({"success": True})


@bp.route("/api/stt/config", methods=["GET"])
@auth_required_api
def stt_config_route():
    """Ob und wie die Spracheingabe laeuft (``stream`` live, ``upload`` aufnehmen und senden)."""
    from app.chat import stt as stt_config

    return jsonify(stt_config.client_config())


@bp.route("/api/stt", methods=["POST"])
@auth_required_api
def stt():
    """Wandelt eine Aufnahme in Text um (OpenAI-kompatibler Standard ``/audio/transcriptions``)."""
    from app.chat import stt as stt_config

    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "Keine Audiodatei"}), 400
    audio = audio_file.read(stt_config.MAX_AUDIO_BYTES + 1)
    if not audio:
        return jsonify({"error": "Leere Audiodatei"}), 400
    if len(audio) > stt_config.MAX_AUDIO_BYTES:
        return jsonify({"error": "Die Aufnahme ist zu groß (höchstens 25 MB)"}), 413

    cfg = stt_config.load()
    try:
        text = stt_config.transcribe(cfg, audio, audio_file.filename)
    except stt_config.TranscriptionError as e:
        return jsonify({"error": str(e)}), 502
    return jsonify({"text": text})


@bp.route("/api/chat/attachments", methods=["POST"])
@auth_required_api
def upload_chat_attachment():
    from app import resources_ready
    from app.chat.attachments import is_image, is_supported, start_attachment_ingestion

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"success": False, "error": "Keine Datei"}), 400
    if not is_supported(file.filename):
        return jsonify(
            {
                "success": False,
                "error": "Dateityp im Chat nicht unterstützt "
                "(PDF, DOCX, DOC, ODT, MD, Bilder). CSV/Excel bitte über Knowledge hochladen.",
            }
        ), 415
    if is_image(file.filename):
        from app.ingest.image_text import enabled

        if not enabled():
            return jsonify(
                {
                    "success": False,
                    "error": "Bilder im Chat sind abgeschaltet (Administration › Einlesen & Modelle).",
                }
            ), 415

    sparse = current_app.extensions.get("sparse_embeddings")
    if not resources_ready() or not sparse:
        return jsonify(
            {
                "success": False,
                "error": "RAG-System wird noch initialisiert. Bitte kurz warten.",
            }
        ), 503

    db = get_db_service()
    session_id = request.form.get("session_id") or None
    user_id = current_user()["id"]
    if session_id and not db.get_chat_session(session_id, user_id=user_id):
        return jsonify({"success": False, "error": "Ungültige Session"}), 404
    if not session_id:
        session_id = db.create_session(user_id=user_id).id

    attachment = start_attachment_ingestion(
        file, session_id, sparse, current_user()["id"]
    )
    return jsonify({"success": True, "session_id": session_id, **attachment}), 202


@bp.route("/api/chat/attachments/<attachment_id>", methods=["GET"])
@auth_required_api
def chat_attachment_status(attachment_id):
    db = get_db_service()
    att = db.get_attachment(attachment_id)
    # Fremde Anhaenge sind nicht auffindbar (404 statt 403).
    if not att or not db.get_chat_session(att.session_id, user_id=current_user()["id"]):
        return jsonify({"error": "Anhang nicht gefunden"}), 404
    return jsonify(
        {
            "id": att.id,
            "session_id": att.session_id,
            "filename": att.filename,
            "status": att.status,
            "error": att.error,
        }
    )
