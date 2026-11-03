"""The session registry — one object owns everything that dies with the tab.

This is Seam 2 from plans/datasurf-plan.md §4a. The DuckDB connection, the raw frames, the
registered tables and the extracted documents all live here and nowhere else. Persisting later
means serializing one object; the discipline is refusing to stash session state anywhere but here.

Session lives in st.session_state, never @st.cache_data — it holds a live DuckDB connection,
which is neither hashable nor picklable.
"""

import re
from dataclasses import dataclass, field

import duckdb
import pandas as pd


@dataclass
class Table:
    """A registered, queryable table and everything we understood about it."""
    df: pd.DataFrame
    card: object              # profile.SchemaCard — untyped to keep store.py free of the import cycle
    changelog: list
    source: str               # filename it came from
    sheet: str | None = None
    header_row: int = 0


@dataclass
class Document:
    pages: dict[int, str]     # 1-indexed page number -> raw text
    source: str

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass
class Session:
    con: duckdb.DuckDBPyConnection
    raw: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)
    tables: dict[str, Table] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)
    # table_name -> header row the user chose. Lives here rather than in widget state, which
    # Streamlit discards for any widget that stops being rendered — and only the selected
    # table renders a header input.
    overrides: dict[str, int] = field(default_factory=dict)
    # M2 onward. Untyped on purpose: `ingest/` must stay importable with no model client in
    # sys.modules, which evals/test_ingest.py asserts.
    turns: list = field(default_factory=list)       # nodes.Answer per question asked
    traces: list = field(default_factory=list)      # trace.Trace per question, for the ledger
    # The artifact store behind the handle pattern (plans/datasurf-plan.md §9). A tool result
    # lands here and the graph state carries only its key, so a 200-row result costs four
    # lines of context. In memory, not parquet: §4a says user data never reaches disk.
    artifacts: dict[str, object] = field(default_factory=dict)   # handle -> tabular.Result
    # (table, column, stat) -> (value, provenance). Presets write here when they run, and
    # column_stat reads it first, so tier 1 gets richer as the session goes on rather than
    # staying frozen at whatever pandas can aggregate (§6, precompute budget).
    stats: dict = field(default_factory=dict)
    findings: dict = field(default_factory=dict)     # preset name -> analysis.Report

    @classmethod
    def new(cls) -> "Session":
        con = duckdb.connect()
        # Model-authored SQL runs on this connection (tools/tabular.py). The statement-type
        # check there refuses COPY and ATTACH, but a plain SELECT over read_csv('/etc/passwd')
        # parses as a SELECT — this is the layer that stops it, and it belongs on the
        # connection rather than in the caller, because every caller shares this one.
        con.execute("SET enable_external_access=false")
        return cls(con=con)

    def add_table(self, name: str, df: pd.DataFrame, card, changelog: list,
                  source: str = "", sheet: str | None = None, header_row: int = 0) -> None:
        """Register and record together — they must never drift apart."""
        self.con.register(name, df)
        self.tables[name] = Table(df, card, changelog, source, sheet, header_row)

    def put_artifact(self, result) -> str:
        """Store a tool result, return the handle the graph state carries instead of it."""
        handle = f"r{len(self.artifacts) + 1}"
        self.artifacts[handle] = result
        return handle

    def add_document(self, name: str, pages: dict[int, str], source: str = "") -> None:
        self.documents[name] = Document(pages, source or name)

    def table_names(self) -> list[str]:
        return list(self.tables)

    def frames(self, exclude: str = "") -> dict[str, pd.DataFrame]:
        """Every other registered frame — what the Schema Card scans for join candidates."""
        return {n: t.df for n, t in self.tables.items() if n != exclude}


def table_name(source: str, sheet: str | None = None) -> str:
    """A SQL-safe, collision-resistant name: 'messy_sales.xlsx' + 'Q1-Q4' -> messy_sales_q1_q4."""
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = [stem, sheet] if sheet else [stem]
    name = re.sub(r"[^0-9a-zA-Z]+", "_", "_".join(parts)).strip("_").lower()
    return f"t_{name}" if not name or name[0].isdigit() else name
