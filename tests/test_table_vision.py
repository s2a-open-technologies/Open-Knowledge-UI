"""Zweiter Durchgang fuer Tabellen aus PDFs per Vision-Modell (app/ingest/table_vision.py)."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from flask import Flask

from app.api import register_blueprints
from app.ingest import prompts, tasks
from app.ingest import table_vision as tv
from app.llm import connections as store
from app.llm import registry
from app.llm.local_llm import VisionLLM
from tests.test_access import add_user, client_for
from tests.test_web_and_vision import ADMIN, put_tasks


@pytest.fixture
def app(db, monkeypatch):
    monkeypatch.setattr(
        registry, "fetch_remote_models", lambda c, force=False: ["m1", "m2"]
    )
    app = Flask(__name__)
    app.secret_key = "t"
    app.extensions["ingestion_status"] = {"is_ingesting": False, "knowledge_id": None}
    register_blueprints(app)
    return app


class FakeModel:
    def __init__(self, answer="", error=None):
        self.answer, self.error, self.calls, self.last_instruction = (
            answer,
            error,
            0,
            None,
        )

    def generate_image_message(self, b64, instruction=None):
        self.last_instruction = instruction
        return b64

    def generate(self, message):
        self.calls += 1
        if self.error:
            raise self.error
        return self.answer


class FakeRenderer:
    def crop(self, page_no, bbox, page_height):
        from PIL import Image

        return Image.new("RGB", (40, 20), "white")


class FakeItem:
    class prov_item:
        page_no = 1
        bbox = object()

    prov = [prov_item]


class FakeDoc:
    class _Page:
        class size:
            height = 800.0

    pages = {1: _Page}


DRAFT = pd.DataFrame(
    {"Posten": ["Miete", "Kaution", "Strom"], "Betrag": ["950", "2850", "40"]}
)
GOOD = "| Posten | Betrag |\n|---|---|\n| Miete | 950,00 |\n| Kaution | 2.850,00 |\n| Strom | 40,00 |"


def refine(model, draft=DRAFT, **kw):
    return tv.refine_table(
        FakeItem(),
        FakeDoc(),
        draft,
        "| draft |",
        FakeRenderer(),
        model,
        "Pruefe.",
        **kw,
    )


def test_markdown_table_is_parsed_with_escaped_pipes_and_fences():
    df = tv.parse_markdown_table(
        "```markdown\n| a | b |\n| --- | :---: |\n| 1 | x\\|y |\n| 2 | 3 |\n```"
    )
    assert (
        list(df.columns) == ["a", "b"] and df.iat[0, 1] == "x|y" and df.shape == (2, 2)
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hier ist die Tabelle",
        "| a | b |\n|---|---|",
        "| a | b |\n|---|---|\n| 1 |",  # keine Datenzeile / ungleiche Spalten
        "| a | b |\n| 1 | 2 |\n| 3 | 4 |",  # ohne Trennzeile
        "| a |\n|---|\n| 1 |",  # nur eine Spalte
    ],
)
def test_unusable_answers_are_rejected(text):
    assert tv.parse_markdown_table(text) is None


def test_plausibility_rejects_dropped_or_invented_rows_and_columns():
    draft = pd.DataFrame([[1, 2]] * 10, columns=["a", "b"])
    ok = pd.DataFrame([[1, 2]] * 11, columns=["a", "b"])
    assert tv.is_plausible(ok, draft)
    assert not tv.is_plausible(
        pd.DataFrame([[1, 2]] * 5, columns=["a", "b"]), draft
    )  # Haelfte fehlt
    assert not tv.is_plausible(
        pd.DataFrame([[1, 2, 3]] * 10, columns=["a", "b", "c"]),
        pd.DataFrame([[1, 2]] * 10, columns=["a", "b"])
        .iloc[:, :1]
        .join(pd.DataFrame({"x": range(10)}))
        .iloc[:, :2]
        .assign(z=1, y=2, w=3),
    )
    assert tv.is_plausible(
        ok, draft, max_deviation_percent=10
    )  # eine Zeile mehr bei zehn ist genau die Grenze
    assert not tv.is_plausible(None, draft)


def test_a_good_answer_replaces_the_docling_table():
    model = FakeModel(GOOD)
    fixed = refine(model)
    assert fixed.iat[1, 1] == "2.850,00" and model.calls == 1
    assert (
        "Pruefe." in model.last_instruction and "Entwurf" in model.last_instruction
    )  # Bild + Entwurf gehen mit


@pytest.mark.parametrize(
    "answer",
    [
        "Das ist keine Tabelle",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "| a | b |\n|---|---|\n| 1 |",
    ],
)
def test_nonsense_keeps_the_docling_table(answer):
    assert refine(FakeModel(answer)) is None


def test_model_errors_are_retried_then_fall_back():
    model = FakeModel(error=RuntimeError("Timeout"))
    assert refine(model, retries=2) is None and model.calls == 2


def test_tables_without_page_info_are_left_alone():
    class NoProv:
        prov = []

    assert (
        tv.refine_table(
            NoProv(), FakeDoc(), DRAFT, "x", FakeRenderer(), FakeModel(GOOD), "p"
        )
        is None
    )


def test_the_renderer_crops_the_table_area_from_a_real_pdf(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    from docling_core.types.doc import BoundingBox, CoordOrigin

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(595, 842)
    path = tmp_path / "x.pdf"
    pdf.save(str(path))
    r = tv.PageRenderer(str(path))
    # Docling-Koordinaten: Ursprung unten links; Tabelle 100..300 breit, 600..700 von unten
    img = r.crop(
        1,
        BoundingBox(l=100, b=600, r=300, t=700, coord_origin=CoordOrigin.BOTTOMLEFT),
        842,
    )
    r.close()
    assert (
        img is not None
        and img.width == (200 + 2 * tv.MARGIN_PT) * tv.SCALE
        and img.height == (100 + 2 * tv.MARGIN_PT) * tv.SCALE
    )


# --- Aufgabe "tables" in den Einstellungen ------------------------------------------------------


def test_tables_task_is_configurable_resettable_and_admin_only(app, db):
    admin = client_for(app, ADMIN)
    over = admin.get("/api/admin/embedding").json["tasks"]["tables"]
    assert (
        over["enabled"] is True
        and over["prompt"] == over["default_prompt"] == prompts.TABLES_PROMPT
        and over["max_row_deviation"] == 30
    )
    r = put_tasks(
        admin,
        tables={
            "enabled": False,
            "prompt": "Nur Tabelle.",
            "max_row_deviation": 10,
            "temperature": 0.2,
        },
    )
    assert r.status_code == 200
    cfg = tasks.load_task("tables")
    assert (
        cfg["enabled"] is False
        and cfg["prompt"] == "Nur Tabelle."
        and cfg["max_row_deviation"] == 10
        and cfg["temperature"] == 0.2
    )
    assert put_tasks(admin, tables={"max_row_deviation": 500}).status_code == 400
    assert put_tasks(admin, tables={"prompt": "  "}).status_code == 400
    back = put_tasks(admin, tables={"prompt": over["default_prompt"]}).json["tasks"][
        "tables"
    ]
    assert back["prompt"] == prompts.TABLES_PROMPT
    assert (
        client_for(app, add_user(db, "ed", "editor"))
        .put("/api/admin/embedding", json={"tasks": {}})
        .status_code
        == 403
    )


def test_the_tables_model_defaults_to_the_vision_model(app, db):
    admin = client_for(app, ADMIN)
    conn = store.create_connection("C", "http://127.0.0.1:9/v1", "k")
    default_ref = store.make_base_ref(conn["id"], "m1")
    vision_ref = store.make_base_ref(conn["id"], "m2")
    store.set_default_model(default_ref)
    put_tasks(admin, vision={"model_ref": vision_ref})
    llm = VisionLLM("tables")
    llm.llm
    assert (
        llm.vision_model == "m2"
    )  # leer = Modell der Bildbeschreibung, nicht das Chat-Standardmodell
    put_tasks(admin, tables={"model_ref": default_ref})
    llm.llm
    assert llm.vision_model == "m1"


def test_migration_0031_matches_the_seed_and_runs_up_down_up(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "m0031", Path("alembic/versions/0031_table_vision.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = {k: v for k, v, _d in mod.ROWS}
    assert rows["TABLES_PROMPT"] == prompts.TABLES_PROMPT and rows[
        "TABLES_MAX_ROW_DEVIATION"
    ] == str(prompts.TABLES_MAX_ROW_DEVIATION)

    env = {**os.environ, "DATABASE_URL": f"sqlite:///{tmp_path / 'm.db'}"}
    run = lambda *a: subprocess.run(
        [sys.executable, "-m", "alembic", *a], env=env, check=True, capture_output=True
    )
    run("upgrade", "head")
    run("downgrade", "0030")
    run("upgrade", "head")


# --- Einbindung in die Tabellenextraktion --------------------------------------------------------


def _doc_with_table():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    doc = DoclingDocument(name="t")
    cells = [
        TableCell(
            text=t,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate(
            [["Posten", "Betrag"], ["Miete", "950"], ["Kaution", "2850"]]
        )
        for c, t in enumerate(row)
    ]
    doc.add_table(data=TableData(num_rows=3, num_cols=2, table_cells=cells))
    return doc


def _extract(base, doc, file_path):
    from app.ingest import tables

    seen = []

    def sink(df, number):
        seen.append(df)
        return {
            "name": f"k/t{number}.csv",
            "rows": int(df.shape[0]),
            "columns": list(df.columns),
            "frame": df,
        }

    markdown = base._extract_tables(doc, doc.export_to_markdown(), sink, file_path)
    return seen, markdown, tables


def test_extraction_uses_the_corrected_table_for_pdfs(db, monkeypatch):
    from app.ingest.loaders.preprocess_base import PreprocessBase

    base = PreprocessBase(model=FakeModel(), table_model=FakeModel(GOOD))
    monkeypatch.setattr(
        "app.ingest.loaders.preprocess_base._vision_enabled", lambda: True
    )
    monkeypatch.setattr(
        tv.PageRenderer, "crop", lambda self, *a: FakeRenderer().crop(*a)
    )
    doc = _doc_with_table()
    # ohne Seitenangabe im Dokument gibt es kein Bild: Docling-Tabelle bleibt
    seen, _md, _t = _extract(base, doc, "x.pdf")
    assert seen[0].iat[0, 1] == "950"
    # mit Seitenangabe wird die Antwort des Modells uebernommen
    monkeypatch.setattr(
        tv, "refine_table", lambda *a, **k: tv.parse_markdown_table(GOOD)
    )
    seen, md, _t = _extract(base, doc, "x.pdf")
    assert seen[0].iat[0, 1] == "950,00" and "[TABELLE 1" in md


def test_word_tables_and_a_disabled_task_never_call_the_model(db, monkeypatch):
    from app.ingest.loaders.preprocess_base import PreprocessBase

    model = FakeModel(GOOD)
    base = PreprocessBase(model=FakeModel(), table_model=model)
    monkeypatch.setattr(
        "app.ingest.loaders.preprocess_base._vision_enabled", lambda: True
    )
    assert base._table_refiner("vertrag.docx") is None  # kein Seitenbild bei Word
    s = db.get_session()
    from app.core.settings_service import SettingsService

    SettingsService(s).set("INGEST_TABLE_VISION", "false")
    s.close()
    assert base._table_refiner("vertrag.pdf") is None
    monkeypatch.setattr(
        "app.ingest.loaders.preprocess_base._vision_enabled", lambda: False
    )
    assert base._table_refiner("vertrag.pdf") is None and model.calls == 0


# --- Tabellen ueber Seitengrenzen ------------------------------------------------------------------

from app.ingest import table_merge as tm

HEAD = ["Nr", "Name", "Wert"]


def frame(rows, header=HEAD):
    return pd.DataFrame(rows, columns=header if header else range(len(rows[0])))


def geo(page, top, bottom, left=100.0, right=500.0):
    return tm.Geometry(page, top, bottom, left, right, 842.0, 595.0)


def test_continuation_needs_neighbour_pages_page_edges_and_matching_columns():
    a = frame([["1", "A", "1,5"], ["2", "B", "2,5"]])
    b = frame([["3", "C", "3,5"]])
    assert tm.is_continuation(a, geo(1, 160, 770), b, geo(2, 70, 300))
    assert not tm.is_continuation(
        a, geo(1, 160, 770), b, geo(3, 70, 300)
    )  # Seite uebersprungen
    assert not tm.is_continuation(
        a, geo(1, 160, 400), b, geo(2, 70, 300)
    )  # erste endet mitten auf der Seite
    assert not tm.is_continuation(
        a, geo(1, 160, 770), b, geo(2, 300, 500)
    )  # zweite beginnt mitten auf der Seite
    assert not tm.is_continuation(
        a, geo(1, 160, 770), frame([["3", "C"]], ["Nr", "Name"]), geo(2, 70, 300)
    )  # Spaltenzahl
    assert not tm.is_continuation(
        a, geo(1, 160, 770), b, geo(2, 70, 300, left=300, right=580)
    )  # woanders auf der Seite
    assert not tm.is_continuation(
        a, None, b, geo(2, 70, 300)
    )  # ohne Seitenangabe nie raten


def test_continuation_with_another_header_is_a_different_table():
    a = frame([["1", "A", "1,5"], ["2", "B", "2,5"]])
    other = frame([["Anna", "Bonn", "7,5"]], ["Kunde", "Stadt", "Betrag"])
    assert not tm.is_continuation(a, geo(1, 160, 770), other, geo(2, 70, 300))


def test_repeated_headers_and_header_less_continuations_are_merged_cleanly():
    a = frame([["1", "A", "1,5"], ["2", "B", "2,5"]])
    repeated = frame(
        [["3", "C", "3,5"]]
    )  # Kopfzeile wiederholt, von Docling als Kopf gelesen
    unnamed = frame(
        [HEAD, ["4", "D", "4,5"]], None
    )  # Kopfzeile als erste Datenzeile, Spalten 0..n
    no_header = frame(
        [["6", "F", "6,5"]], ["5", "E", "5,5"]
    )  # erste Datenzeile wurde zum Kopf
    for part, geometry in (
        (repeated, geo(2, 70, 300)),
        (unnamed, geo(2, 70, 300)),
        (no_header, geo(2, 70, 300)),
    ):
        assert tm.is_continuation(a, geo(1, 160, 770), part, geometry)
    merged = tm.merge_frames([a, repeated, unnamed, no_header])
    assert list(merged.columns) == HEAD
    assert merged["Nr"].tolist() == ["1", "2", "3", "4", "5", "6"]


def _doc_with_split_table(second_header=None, text_between=False):
    from docling_core.types.doc import (
        BoundingBox,
        CoordOrigin,
        DoclingDocument,
        ProvenanceItem,
        Size,
        TableCell,
        TableData,
    )

    doc = DoclingDocument(name="t")
    for n in (1, 2):
        doc.add_page(page_no=n, size=Size(width=595, height=842))

    def add(page, rows, bottom_left):
        cells = [
            TableCell(
                text=t,
                start_row_offset_idx=r,
                end_row_offset_idx=r + 1,
                start_col_offset_idx=c,
                end_col_offset_idx=c + 1,
                column_header=(r == 0),
            )
            for r, row in enumerate(rows)
            for c, t in enumerate(row)
        ]
        l_, b_, r_, t_ = bottom_left
        prov = ProvenanceItem(
            page_no=page,
            bbox=BoundingBox(
                l=l_, b=b_, r=r_, t=t_, coord_origin=CoordOrigin.BOTTOMLEFT
            ),
            charspan=(0, 0),
        )
        doc.add_table(
            data=TableData(
                num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells
            ),
            prov=prov,
        )

    add(
        1, [["Nr", "Name"], ["1", "A"], ["2", "B"]], (100, 60, 500, 680)
    )  # reicht bis fast an den unteren Seitenrand
    if text_between:
        doc.add_text(label="text", text="Zwischentext")
    add(
        2, [second_header or ["Nr", "Name"], ["3", "C"]], (100, 600, 500, 770)
    )  # beginnt oben auf Seite 2
    return doc


def test_extraction_merges_a_table_across_pages_into_one_csv_and_one_reference(db):
    from app.ingest.loaders.preprocess_base import PreprocessBase

    base = PreprocessBase(model=FakeModel(), table_model=FakeModel())
    base._table_refiner = lambda fp: None
    seen, md, _t = _extract(base, _doc_with_split_table(), "x.pdf")
    assert len(seen) == 1 and seen[0]["Nr"].tolist() == ["1", "2", "3"]
    assert (
        "Seiten 1–2" in md and md.count("[TABELLE") == 1 and "| 3" not in md
    )  # zweite Seite steht nur in der CSV


@pytest.mark.parametrize(
    "kwargs", [{"text_between": True}, {"second_header": ["Kunde", "Stadt"]}]
)
def test_text_between_or_another_header_keeps_the_tables_apart(db, kwargs):
    from app.ingest.loaders.preprocess_base import PreprocessBase

    base = PreprocessBase(model=FakeModel(), table_model=FakeModel())
    base._table_refiner = lambda fp: None
    seen, md, _t = _extract(base, _doc_with_split_table(**kwargs), "x.pdf")
    assert len(seen) == 2 and md.count("[TABELLE") == 2 and "Seiten" not in md
