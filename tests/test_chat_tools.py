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


from types import SimpleNamespace

from app.agent.document_agent import (
    CONVERSATION_ONLY_HINT,
    KNOWLEDGE_TOOLS,
    DocumentAgent,
    select_tools,
)

ALL = [
    "document_search_tool",
    "read_knowledge_index",
    "list_files",
    "preview_data",
    "run_pandas",
    "list_summaries",
    "read_summary",
    "write_summary",
    "edit_summary",
    "web_search_tool",
]


def tools():
    return [SimpleNamespace(name=n) for n in ALL]


def names(ts):
    return [t.name for t in ts]


def test_without_knowledge_only_the_conversation_tools_remain():
    got = names(select_tools(tools(), has_knowledge=False))
    assert not set(got) & KNOWLEDGE_TOOLS
    assert got == [
        "list_summaries",
        "read_summary",
        "write_summary",
        "edit_summary",
        "web_search_tool",
    ]


def test_with_knowledge_or_attachments_all_tools_are_kept():
    assert names(select_tools(tools(), has_knowledge=True)) == ALL


def test_the_input_list_is_not_modified():
    original = tools()
    select_tools(original, has_knowledge=False)
    assert names(original) == ALL


def test_default_instructions_separate_conversation_summaries_from_document_summaries():
    from app.agent.behavior import DEFAULT_BEHAVIOR as prompt

    assert "Gespräch" in prompt and "keine Dokumentsuche" in prompt
    assert "FAZIT VORAB" not in prompt


def test_default_instructions_answer_targeted_questions_briefly_and_list_only_on_request():
    from app.agent.behavior import DEFAULT_BEHAVIOR as prompt

    assert (
        "Zähle nie den übrigen Inhalt auf" in prompt
        and "ausdrücklich eine Übersicht" in prompt
    )
    assert "nie vollständig wieder" in DocumentAgent.SYSTEM_PROMPT


def test_conversation_only_hint_forbids_inventing_documents():
    assert (
        "kein Wissen" in CONVERSATION_ONLY_HINT
        and "Erfinde keine Dokumentinhalte" in CONVERSATION_ONLY_HINT
    )


def test_write_summary_description_does_not_force_a_document_search():
    from app.agent.tools.custom_tools import WRITE_SUMMARY_DESCRIPTION

    assert (
        "Gespräch ODER" in WRITE_SUMMARY_DESCRIPTION
        and "Meine Ablagen" in WRITE_SUMMARY_DESCRIPTION
    )
