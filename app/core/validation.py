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
"""Zahlen aus Formularen pruefen (ein Ort statt je ein try/except pro Feld)."""


def parse_number(raw, label: str, low: float, high: float, kind=int):
    """``raw`` als ``kind`` (int/float) lesen und auf ``low..high`` pruefen.

    Wirft ``ValueError`` mit deutscher Meldung (``"<label>: ..."``), die der Aufrufer
    unveraendert an den Nutzer geben kann.
    """
    try:
        number = kind(raw)
    except (TypeError, ValueError):
        what = "ganze Zahl" if kind is int else "Zahl"
        raise ValueError(f"{label}: {what} erwartet") from None
    if not low <= number <= high:
        raise ValueError(f"{label}: erlaubt sind {_show(low)} bis {_show(high)}")
    return number


def _show(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)
