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
Automatische Compaction des Chat-Kontexts.

Eingabe-Budget = ``num_ctx`` minus ``max_tokens`` (die Ausgabe ist reserviert). Ab
``(100 - Luft) %`` dieses Budgets wird der aeltere Verlauf zusammengefasst, damit der Kontext
nie ueberlaeuft. Alle Werte und der Prompt liegen in der Datenbank (``settings``); hier stehen
nur die Samen. Token werden offline und bewusst vorsichtig geschaetzt (3 Zeichen je Token).
"""

import math
import re
from dataclasses import dataclass
from logging import getLogger

from app.core.database_service import get_db_service
from app.core.models import Setting
from app.core.settings_service import SettingsService

logger = getLogger(__name__)

CHARS_PER_TOKEN = 3
MIN_USABLE_TOKENS = 1024
MAX_PROMPT_CHARS = 8000

DEFAULT_PROMPT = (
    "Du fasst den bisherigen Verlauf eines Gesprächs zusammen, damit es mit weniger Platz "
    "fortgesetzt werden kann. Die Zusammenfassung ersetzt die unten stehenden Nachrichten.\n\n"
    "Behalte unbedingt:\n"
    "- Fragen, Ziele und Wünsche des Nutzers\n"
    "- Wichtige Fakten, Zahlen, Namen, Fristen, Dateinamen und Quellenangaben (mit Seite oder Abschnitt)\n"
    "- Getroffene Entscheidungen und Ergebnisse\n"
    "- Offene Punkte und Rückfragen\n\n"
    "Lass Floskeln, Wiederholungen und Zwischenschritte weg. Schreibe knapp, sachlich und auf "
    "Deutsch. Gib ausschließlich die Zusammenfassung aus, ohne Einleitung.\n\n"
    "{messages}"
)

# (Schluessel, Standardwert, Beschreibung)
SETTINGS = [
    (
        "COMPACTION_ENABLED",
        "true",
        "Chat-Verlauf automatisch zusammenfassen, bevor der Kontext ueberlaeuft",
    ),
    (
        "COMPACTION_HEADROOM_PERCENT",
        "5",
        "Luft in Prozent: Zusammenfassen ab (100 - Wert) % des Eingabe-Budgets",
    ),
    (
        "COMPACTION_KEEP_PERCENT",
        "30",
        "Anteil des Eingabe-Budgets in Prozent, der als juengster Verlauf wortwoertlich bleibt",
    ),
    (
        "COMPACTION_OUTPUT_RESERVE_PERCENT",
        "20",
        "Reserve fuer die Antwort in Prozent des Kontexts, wenn das Modell keine max. Tokens setzt",
    ),
    (
        "COMPACTION_PROMPT",
        DEFAULT_PROMPT,
        "Anweisung fuer die Zusammenfassung des Verlaufs (Platzhalter {messages})",
    ),
]

_LIMITS = {
    "headroom_percent": (1, 50),
    "keep_percent": (10, 80),
    "output_reserve_percent": (5, 60),
}


class CompactionConfigError(ValueError):
    """Ungueltige Einstellung (lesbare Meldung fuer die Oberflaeche)."""


def estimate_tokens(text: str) -> int:
    """Vorsichtige Schaetzung ohne Tokenizer (deutscher Text liegt bei ~3-4 Zeichen je Token)."""
    return math.ceil(len(text or "") / CHARS_PER_TOKEN)


def _int(raw, default: int) -> int:
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def load_settings() -> dict:
    """Aktuelle Einstellungen (frisch aus der Datenbank, Samen als Rueckfall)."""
    session = get_db_service().get_session()
    try:
        s = SettingsService(session)
        stored = session.query(Setting).filter_by(key="COMPACTION_PROMPT").first()
        prompt = (s.get("COMPACTION_PROMPT") or "").strip() or DEFAULT_PROMPT
        default_prompt = (
            (stored.default_value if stored else "") or ""
        ).strip() or DEFAULT_PROMPT
        return {
            "enabled": s.get_bool("COMPACTION_ENABLED", True),
            "headroom_percent": _int(s.get("COMPACTION_HEADROOM_PERCENT"), 5),
            "keep_percent": _int(s.get("COMPACTION_KEEP_PERCENT"), 30),
            "output_reserve_percent": _int(
                s.get("COMPACTION_OUTPUT_RESERVE_PERCENT"), 20
            ),
            "prompt": prompt,
            "default_prompt": default_prompt,
        }
    finally:
        session.close()


def save_settings(data: dict) -> None:
    """Prueft und speichert; nicht genannte Felder bleiben unveraendert."""
    values: dict[str, str] = {}
    if "enabled" in data:
        values["COMPACTION_ENABLED"] = "true" if data["enabled"] else "false"
    for field, (low, high) in _LIMITS.items():
        if field in data:
            try:
                number = int(data[field])
            except (TypeError, ValueError):
                raise CompactionConfigError(f"{field}: ganze Zahl erwartet") from None
            if not low <= number <= high:
                raise CompactionConfigError(
                    f"{field}: Wert zwischen {low} und {high} erwartet"
                )
            values[f"COMPACTION_{field.upper()}"] = str(number)
    if "prompt" in data:
        text = str(data["prompt"] or "").strip()
        if not text:
            raise CompactionConfigError("Der Prompt darf nicht leer sein")
        if len(text) > MAX_PROMPT_CHARS:
            raise CompactionConfigError(
                f"Der Prompt ist zu lang (höchstens {MAX_PROMPT_CHARS} Zeichen)"
            )
        if "{messages}" not in text:
            raise CompactionConfigError(
                "Der Prompt muss den Platzhalter {messages} enthalten"
            )
        values["COMPACTION_PROMPT"] = text
    if not values:
        return
    session = get_db_service().get_session()
    try:
        SettingsService(session).set_batch(values)
    finally:
        session.close()


@dataclass(frozen=True)
class Budget:
    window: int  # num_ctx
    output: int  # fuer die Antwort reserviert
    usable: int  # Eingabe-Budget (window - output)
    trigger: int  # ab hier wird zusammengefasst
    keep: int  # so viel juengster Verlauf bleibt wortwoertlich


def compute_budget(params: dict, cfg: dict | None = None) -> Budget | None:
    """Budget fuer die wirksamen Modellparameter oder ``None`` (aus / Fenster zu klein)."""
    cfg = cfg or load_settings()
    if not cfg["enabled"]:
        return None
    window = int(params.get("num_ctx") or 0)
    max_tokens = params.get("max_tokens")
    output = (
        int(max_tokens)
        if max_tokens
        else math.ceil(window * cfg["output_reserve_percent"] / 100)
    )
    usable = window - output
    if usable < MIN_USABLE_TOKENS:
        logger.warning(
            "Compaction aus: Eingabe-Budget %s Token (Kontext %s - Ausgabe %s) ist zu klein",
            usable,
            window,
            output,
        )
        return None
    return Budget(
        window=window,
        output=output,
        usable=usable,
        trigger=int(usable * (100 - cfg["headroom_percent"]) / 100),
        keep=int(usable * cfg["keep_percent"] / 100),
    )


def effective_params(model_params: dict | None) -> dict:
    """Modellwerte ueber den Standardwerten (wie ``build_chat_llm``)."""
    from app.llm.factory import global_chat_defaults

    return {**global_chat_defaults(), **(model_params or {})}


_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_SOURCES = re.compile(r"\n*<sources>.*?</sources>\s*$", re.DOTALL)
_TOOLS = re.compile(r"\n*<tools>.*?</tools>\s*$", re.DOTALL)


def clean_history_text(content: str) -> str:
    """Anzeigebestandteile (Quellen-, Werkzeug- und Denkblock) gehoeren nicht in den Verlauf fuers Modell."""
    return _THINK.sub("", _TOOLS.sub("", _SOURCES.sub("", content or ""))).strip()


def prompt_for_langchain(template: str) -> str:
    """Mit ``str.format`` benutzbar: nur ``{messages}`` bleibt Platzhalter."""
    return (
        template.replace("{", "{{")
        .replace("}", "}}")
        .replace("{{messages}}", "{messages}")
    )


def render_prompt(template: str, messages_text: str) -> str:
    return template.replace("{messages}", messages_text)


def _format(messages: list[dict]) -> str:
    names = {"user": "Nutzer", "assistant": "Assistent"}
    return "\n\n".join(
        f"{names.get(m['role'], m['role'])}: {m['content']}" for m in messages
    )


def summarize(llm, cfg: dict, messages: list[dict], previous: str | None = None) -> str:
    """Fasst Nachrichten (und eine fruehere Zusammenfassung) zusammen. ``llm`` hat ``invoke``."""
    parts = list(messages)
    if previous:
        parts.insert(
            0,
            {
                "role": "assistant",
                "content": f"[Bisherige Zusammenfassung]\n{previous}",
            },
        )
    reply = llm.invoke(render_prompt(cfg["prompt"], _format(parts)))
    text = reply.content if hasattr(reply, "content") else str(reply)
    if isinstance(text, list):  # inhaltsbloecke
        text = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in text
        )
    text = _THINK.sub("", text or "").strip()
    if not text:
        raise RuntimeError("Die Zusammenfassung ist leer")
    return text


def tools_tokens(tools) -> int:
    """Grobe Schaetzung, wie viel Platz die Werkzeug-Beschreibungen im Kontext brauchen."""
    import json

    total = 0
    for tool in tools or []:
        try:
            total += estimate_tokens(
                f"{tool.name} {tool.description} {json.dumps(tool.args, default=str)}"
            )
        except Exception:
            total += 200
    return total


def summary_llm(resolved, budget: Budget):
    """Chat-Modell fuer Zusammenfassungen: ohne Streaming, ohne Denken, begrenzte Antwortlaenge."""
    from app.llm.factory import build_chat_llm

    params = {
        **(resolved.params or {}),
        "thinking": "none",
        "temperature": 0.1,
        "max_tokens": max(256, min(2000, budget.usable // 4)),
    }
    return build_chat_llm(
        resolved.base_url,
        resolved.api_key,
        resolved.model_id,
        params,
        streaming=False,
        timeout=180,
    )


SUMMARY_INTRO = "[Zusammenfassung des bisherigen Gesprächs]\n"


def history_messages(rows, summary: str | None) -> list[dict]:
    """Verlauf fuers Modell: optionale Zusammenfassung zuerst, dann die Nachrichten."""
    out: list[dict] = []
    if summary:
        out.append({"role": "user", "content": SUMMARY_INTRO + summary})
        out.append(
            {
                "role": "assistant",
                "content": "Verstanden, ich berücksichtige den bisherigen Verlauf.",
            }
        )
    for row in rows:
        out.append(
            {
                "role": "user" if row.role == "user" else "assistant",
                "content": clean_history_text(row.content),
            }
        )
    return out


def clip_text(text: str, max_tokens: int) -> str:
    """Kuerzt auf ungefaehr ``max_tokens`` (Anfang und Ende bleiben, Mitte entfaellt)."""
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    return f"{text[:head]}\n\n[… gekürzt, um den Kontext nicht zu sprengen …]\n\n{text[-tail:]}"


def fit_history(
    rows,
    summary: str | None,
    new_message: str,
    overhead_tokens: int,
    budget: Budget,
    cfg: dict,
    llm_factory,
    persist,
    mask_many=None,
    unmask=None,
) -> tuple[list[dict], dict | None]:
    """Liefert den Verlauf fuers Modell (ohne die neue Frage) und ggf. Angaben zur Compaction.

    ``rows``: Nachrichten nach der gespeicherten Zusammenfassung (``role``, ``content``, ``id``).
    ``llm_factory()`` baut das Zusammenfassungs-Modell, ``persist(summary, through_id)`` speichert.
    Mit ``mask_many(texts) -> (maskierte Texte, Mapping)`` und ``unmask(text, Mapping)`` (PII-Filter)
    sieht das Zusammenfassungs-Modell nur maskierten Text; gespeichert wird der zurueckuebersetzte Klartext.
    """
    rows = list(rows)

    def total(msgs: list[dict]) -> int:
        return (
            overhead_tokens
            + estimate_tokens(new_message)
            + sum(estimate_tokens(m["content"]) for m in msgs)
        )

    messages = history_messages(rows, summary)
    before = total(messages)
    if before <= budget.trigger:
        return messages, None

    # Schnitt: rows[:cut] werden zusammengefasst, rows[cut:] bleiben wortwoertlich. Es bleibt so viel
    # juengster Verlauf, wie in ``keep`` passt (mindestens die letzten zwei Nachrichten).
    sizes = [estimate_tokens(clean_history_text(r.content)) for r in rows]
    acc, cut = 0, len(rows)
    for i in range(len(rows) - 1, -1, -1):
        if acc + sizes[i] > budget.keep and len(rows) - i > 2:
            break
        acc += sizes[i]
        cut = i
    while (
        cut < len(rows) and rows[cut].role != "user"
    ):  # der behaltene Teil beginnt mit einer Frage
        cut += 1
    if cut >= len(rows):
        users = [i for i, r in enumerate(rows) if r.role == "user"]
        cut = users[-1] if users else 0
    if cut <= 0:
        logger.info(
            "Compaction: nichts Aelteres zum Zusammenfassen, Nachrichten werden gekuerzt"
        )
        clipped = _clip_all(messages, budget)
        return clipped, {"before": before, "after": total(clipped), "clipped": True}

    old, kept = rows[:cut], rows[cut:]
    older = [
        {
            "role": "user" if r.role == "user" else "assistant",
            "content": clean_history_text(r.content),
        }
        for r in old
    ]
    if mask_many:
        texts = ([summary] if summary else []) + [m["content"] for m in older]
        masked, mapping = mask_many(texts)
        previous = masked[0] if summary else None
        older = [
            {**m, "content": t} for m, t in zip(older, masked[1 if summary else 0 :])
        ]
        new_summary = unmask(summarize(llm_factory(), cfg, older, previous), mapping)
    else:
        new_summary = summarize(llm_factory(), cfg, older, summary)
    persist(new_summary, old[-1].id)
    result = history_messages(kept, new_summary)
    if total(result) > budget.trigger:  # einzelne riesige Nachricht
        result = _clip_all(result, budget)
    return result, {"before": before, "after": total(result), "summarized": len(old)}


def _clip_all(messages: list[dict], budget: Budget) -> list[dict]:
    """Letzter Ausweg: jede einzelne Nachricht auf einen Anteil des Budgets kuerzen."""
    limit = max(256, int(budget.usable * 0.3))
    return [{**m, "content": clip_text(m["content"], limit)} for m in messages]
