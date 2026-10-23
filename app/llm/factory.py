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

"""Aufbau des Chat-LLMs (OpenAI-kompatibel) aus Endpunkt und Parametern."""

import os

import openai
from langchain_openai import ChatOpenAI


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI, das das Denken von Servern mit Reasoning-Parser weiterreicht.

    ``ChatOpenAI`` haelt sich an die OpenAI-Spezifikation und wirft Zusatzfelder
    anderer Anbieter weg. vLLM (und llama.cpp mit Reasoning-Parser) liefern das
    Denken getrennt vom Inhalt in ``delta.reasoning`` bzw. ``delta.reasoning_content``;
    ohne diese Unterklasse bliebe der Chat waehrend des Denkens leer. Der Chat liest
    es aus ``additional_kwargs["reasoning_content"]``.
    """

    def _convert_chunk_to_generation_chunk(
        self, chunk, default_chunk_class, base_generation_info
    ):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return None
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        delta = (choices[0].get("delta") if choices else None) or {}
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning:
            generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning
        return generation_chunk


_FALLBACK = {
    "temperature": "0.1",
    "top_p": "0.2",
    "top_k": "40",
    "max_tokens": "",
    "num_ctx": "4096",
    "thinking": "none",
}
_SETTING_KEYS = {
    "temperature": "CHAT_TEMPERATURE",
    "top_p": "CHAT_TOP_P",
    "top_k": "CHAT_TOP_K",
    "max_tokens": "CHAT_MAX_TOKENS",
    "num_ctx": "CHAT_NUM_CTX",
    "thinking": "CHAT_THINKING",
}


def global_chat_defaults() -> dict:
    """Globale Sampling-Einstellungen (Settings-DB, sonst env, sonst Default).

    Sie gelten fuer jedes Modell, das einen Parameter nicht selbst setzt.
    """
    raw = dict(_FALLBACK)
    try:
        from app.core.database_service import get_db_service
        from app.core.settings_service import SettingsService

        session = get_db_service().get_session()
        try:
            settings = SettingsService(session)
            for name, key in _SETTING_KEYS.items():
                raw[name] = settings.get(key, raw[name]) or raw[name]
            # leerer Wert ist bei max_tokens gueltig (= unbegrenzt)
            raw["max_tokens"] = settings.get("CHAT_MAX_TOKENS", "") or ""
        finally:
            session.close()
    except Exception:
        for name, key in _SETTING_KEYS.items():
            raw[name] = os.getenv(key, raw[name])

    return {
        "temperature": float(raw["temperature"] or 0.1),
        "top_p": float(raw["top_p"] or 0.2),
        "top_k": int(raw["top_k"] or 40),
        "max_tokens": int(raw["max_tokens"])
        if str(raw["max_tokens"]).strip()
        else None,
        "num_ctx": int(raw["num_ctx"] or 4096),
        "thinking": raw["thinking"] or "none",
    }


def build_chat_llm(
    base_url: str,
    api_key: str,
    model: str,
    params: dict | None = None,
    streaming: bool = True,
    timeout: float | None = None,
) -> ReasoningChatOpenAI:
    """Baut das Chat-LLM. ``params`` ueberschreibt die globalen Defaults."""
    p = {**global_chat_defaults(), **(params or {})}

    # extra_body wird vom OpenAI-SDK ungeprueft im Request mitgeschickt - ideal
    # fuer provider-spezifische Parameter wie top_k und num_ctx. (model_kwargs
    # wuerden als top-level kwargs validiert und vom SDK abgelehnt.)
    extra_body = {}
    if p.get("top_k"):
        extra_body["top_k"] = p["top_k"]
    if p.get("num_ctx"):
        extra_body["num_ctx"] = p["num_ctx"]

    kwargs = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key or "none",
        "streaming": streaming,
        "temperature": p["temperature"],
        "top_p": p["top_p"],
    }
    if p.get("max_tokens") is not None:
        kwargs["max_tokens"] = p["max_tokens"]
    if timeout is not None:
        kwargs["timeout"] = timeout

    # Thinking/Reasoning:
    # - Ollama nutzt reasoning_effort (none/low/medium/high/max).
    # - llama.cpp parst Think-Tags standardmaessig und liefert Thinking als
    #   reasoning_content, das langchain-openai (v1.x) nicht in additional_kwargs
    #   durchreicht. Daher reasoning_format=none: die Tags bleiben inline im
    #   content und das Frontend zeigt sie als ausklappbaren Block.
    #   reasoning_effort=none deaktiviert Thinking explizit.
    thinking = p.get("thinking") or "none"
    if thinking != "none":
        kwargs["reasoning_effort"] = thinking
        extra_body["reasoning_format"] = "none"
    else:
        kwargs["reasoning_effort"] = "none"

    if extra_body:
        kwargs["extra_body"] = extra_body
    # Eigener async-Client je Instanz: langchain-openai teilt sonst einen per lru_cache
    # gehaltenen Client samt Verbindungspool zwischen allen Modellen. Jeder Chat laeuft aber
    # in einem neuen Event-Loop; eine gepoolte Verbindung des vorigen (geschlossenen) Loops
    # endet in "RuntimeError: Event loop is closed". Der Client bindet sich erst bei der
    # ersten Anfrage an den Loop, daher ist das Anlegen hier unkritisch.
    kwargs["http_async_client"] = openai.DefaultAsyncHttpxClient()
    return ReasoningChatOpenAI(**kwargs)
