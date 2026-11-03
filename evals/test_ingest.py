"""What breaks if ingestion regresses.

    .venv/bin/pytest evals/test_ingest.py -q

Asserts against data/fixtures/messy_sales.xlsx, whose every nasty trait is deliberate —
see data/fixtures/make_fixtures.py. The fixture is seeded, so the counts here are exact.

The last test is the one that matters most: M1 is defined by having no model call anywhere
in its path, and that is a property worth testing rather than promising.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest import documents, profile, router, store, tabular

FIXTURES = ROOT / "data" / "fixtures"
MESSY = FIXTURES / "messy_sales.xlsx"
REPS = FIXTURES / "reps.csv"

EXPECTED_COLUMNS = [
    "order_date", "region", "rep_id", "product", "units", "revenue", "discount_pct",
]
N_ROWS_AFTER_DEDUPE = 240
N_DUPES = 12


@pytest.fixture(scope="module")
def raw():
    return tabular.read_raw(MESSY)


@pytest.fixture(scope="module")
def clean(raw):
    sheet = raw["Q1-Q4"]
    header_row, _ = tabular.detect_header(sheet)
    return tabular.normalize(sheet, header_row)


# --- M1.2 read -------------------------------------------------------------------

def test_reads_every_sheet(raw):
    assert list(raw) == ["Q1-Q4", "Notes"]


def test_raw_keeps_the_preamble(raw):
    """header=None throughout: nothing is decided at read time."""
    assert raw["Q1-Q4"].iloc[0, 0] == "Acme Corp — Sales Export"


# --- M1.3 header detection -------------------------------------------------------

def test_finds_the_header_under_four_junk_rows(raw):
    header_row, confidence = tabular.detect_header(raw["Q1-Q4"])
    assert header_row == 4
    assert confidence > 0


def test_header_row_zero_when_the_file_is_clean():
    header_row, _ = tabular.detect_header(tabular.read_raw(REPS)["reps"])
    assert header_row == 0


def test_merged_header_cell_names_every_column_it_spans(tmp_path):
    """A merged header arrives as value-then-NaN. Scoring those gaps as missing made the real
    header look emptier than the data below it, and on a narrow sheet the data row won."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws["A1"] = "Regional breakdown"
    ws.merge_cells("A1:D1")
    ws["A3"], ws["B3"], ws["D3"] = "Region", "H1 2024", "Notes"
    ws.merge_cells("B3:C3")                       # one header cell spanning two columns
    for i, row in enumerate([["North", 10, 20, "ok"], ["South", 30, 40, "ok"]], start=4):
        for c, value in enumerate(row, start=1):
            ws.cell(row=i, column=c, value=value)
    wb.save(tmp_path / "merged.xlsx")

    raw = tabular.read_raw(tmp_path / "merged.xlsx")["Sheet"]
    header_row, confidence = tabular.detect_header(raw)
    assert header_row == 2, "picked a data row over the header"
    assert confidence > 0

    df, _ = tabular.normalize(raw, header_row)
    assert list(df.columns) == ["region", "h1_2024", "h1_2024_2", "notes"]
    assert len(df) == 2


# --- M1.4 normalization ----------------------------------------------------------

def test_columns_are_snake_case(clean):
    df, _ = clean
    assert list(df.columns) == EXPECTED_COLUMNS


def test_serial_integers_became_real_dates(clean):
    df, _ = clean
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
    assert df["order_date"].min().year == 2023
    assert df["order_date"].max().year == 2024


def test_currency_strings_became_floats(clean, raw):
    """'$94,897.17' -> 94897.17, checked against the exact cell it came from."""
    df, _ = clean
    assert pd.api.types.is_float_dtype(df["revenue"])

    written = raw["Q1-Q4"].iloc[5, 5]                        # first data row, Revenue ($)
    assert written.startswith("$") and "," in written        # the fixture is still messy
    assert df["revenue"].iloc[0] == float(written.lstrip("$").replace(",", ""))
    assert df["revenue"].between(12.40, 98_430.00).all()


def test_totals_row_is_gone(clean):
    df, _ = clean
    assert "TOTAL" not in df["region"].astype(str).values
    assert not df["product"].isna().all()
    assert df["order_date"].notna().all()


def test_exact_duplicates_dropped(clean):
    df, changelog = clean
    assert len(df) == N_ROWS_AFTER_DEDUPE
    dedupe = [c for c in changelog if c.step == "dedupe"]
    assert dedupe and dedupe[0].count == N_DUPES


def test_changelog_records_the_rename_trail(clean):
    _, changelog = clean
    renames = tabular.renames(changelog)
    assert renames["revenue"] == "Revenue ($)"
    assert renames["discount_pct"] == "Discount %"


def test_changelog_is_ordered_and_non_empty(clean):
    _, changelog = clean
    steps = [c.step for c in changelog]
    assert steps.index("rename") < steps.index("dedupe")


# --- M1.5 schema card ------------------------------------------------------------

def test_schema_card_reports_what_was_understood(clean):
    df, changelog = clean
    card = profile.build_schema_card(df, "sales", changelog, sheet="Q1-Q4", header_row=4)
    text = card.to_text()

    assert "sales" in text and "240 rows" in text
    assert "header detected row 4" in text
    assert '[was "Revenue ($)"]' in text          # the rename trail the model answers in
    assert "212" not in text or "dedupe" in text


def test_card_carries_full_precision(clean):
    """A rounded figure in the card becomes a wrong figure in an answer later."""
    df, changelog = clean
    card = profile.build_schema_card(df, "sales", changelog, sheet="Q1-Q4", header_row=4)
    revenue = next(c for c in card.columns if c.name == "revenue")
    assert abs(revenue.mean - df["revenue"].mean()) < 1e-6


def test_provenance_and_findings_stay_separate(clean):
    """The model reads them as one block; the UI shows them in different places."""
    df, changelog = clean
    card = profile.build_schema_card(df, "sales", changelog, sheet="Q1-Q4", header_row=4)
    assert any("dedupe" in n for n in card.notes)
    assert not any("dedupe" in f.text for f in card.findings)
    assert all(line in card.to_text()
               for line in card.notes + [f.text for f in card.findings])
    assert all(f.human and f.human != f.text for f in card.findings)


def test_card_notices_clustered_missingness(clean):
    df, changelog = clean
    card = profile.build_schema_card(df, "sales", changelog, sheet="Q1-Q4", header_row=4)
    findings = " ".join(f.text for f in card.findings)
    assert "discount_pct" in findings and "East" in findings


def test_card_finds_the_join_candidate(clean):
    df, changelog = clean
    reps, _ = tabular.normalize(tabular.read_raw(REPS)["reps"], 0)
    card = profile.build_schema_card(
        df, "sales", changelog, sheet="Q1-Q4", header_row=4, others={"reps": reps}
    )
    findings = " ".join(f.text for f in card.findings)
    assert "rep_id" in findings and "reps" in findings


# --- M1.6 registry ---------------------------------------------------------------

def test_registered_tables_are_queryable(clean):
    df, changelog = clean
    session = store.Session.new()
    card = profile.build_schema_card(df, "sales", changelog, sheet="Q1-Q4", header_row=4)
    session.add_table("sales", df, card, changelog, header_row=4)

    total = session.con.sql("SELECT count(*) FROM sales").fetchone()[0]
    assert total == N_ROWS_AFTER_DEDUPE
    assert session.table_names() == ["sales"]


# --- M1.7 router + documents -----------------------------------------------------

@pytest.mark.parametrize("name, lane", [
    ("a.csv", "tabular"), ("b.xlsx", "tabular"), ("c.xlsm", "tabular"), ("d.pdf", "document"),
])
def test_router_picks_the_lane(name, lane):
    assert router.lane_for(name) == lane


def test_router_rejects_what_it_cannot_read():
    with pytest.raises(ValueError):
        router.lane_for("notes.docx")


# --- the module's defining property ----------------------------------------------

FORBIDDEN = ("openai", "langchain", "langgraph", "fastmcp", "mcp", "anthropic")

PROBE = f"""
import sys
from ingest import documents, profile, router, store, tabular
print([m for m in sys.modules if m.split(".")[0] in {FORBIDDEN!r}])
"""


def test_ingestion_never_imports_a_model_client():
    """M1 runs with no API key. Importing every ingest module must not pull an LLM client.

    In a subprocess, because the thing being asserted is a property of `ingest/` and not of
    this process: once M2 arrived, evals/test_agent.py put `openai` in a shared sys.modules
    and an in-process check started passing or failing on test file order. Planting
    `import openai` in router.py must be what fails this, and nothing else.
    """
    probe = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT,
                           capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", f"ingest/ pulled a model client: {probe.stdout}"
