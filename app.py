"""DataSurf — upload a file, see what was understood.

    .venv/bin/streamlit run app.py

M1 is ingestion: no model call, no API key needed — that half still runs with neither.
M2 adds the graph on top of it: a question is resolved at the cheapest tier that can answer
it, and the screen says which tier that was and what it checked. M2.4's presets are pure
pandas and scipy — they cost nothing and run without a key. M4.1's ledger makes every call
visible from the first question rather than after the optimizations (plans/modules.md §6).

The Schema Card is written for a model — dense, monospace, token-counted. This file renders
the same facts a second time for a human: what was read, what was changed, what was found.
The model's view is still here, one expander down.

Streamlit re-executes this file top to bottom on every interaction, so the rule is: read each
file exactly once into Session.raw, and re-derive tables only when a header row changes.
"""

import pandas as pd
import streamlit as st

import config
from agent import graph
from analysis import anomalies, eda, health, matrices, segments
from harness import ledger
from harness.trace import Trace
from ingest import documents, profile, router, store, tabular

ACCEPTED = ["csv", "xlsx", "xlsm", "pdf"]
PREVIEW_ROWS = 12
CLEAR_MARGIN = 0.15          # below this the detector is guessing, and should say so
HEADER_TINT = "background-color: rgba(59, 130, 246, 0.28)"   # readable on light and dark

st.set_page_config(page_title="DataSurf", page_icon=":material/waves:", layout="wide")


def session() -> store.Session:
    if "session" not in st.session_state:
        st.session_state.session = store.Session.new()   # holds a DuckDB connection: never cache_data
    return st.session_state.session


# --- ingestion -------------------------------------------------------------------

def read_new_files(sess: store.Session, files) -> list[str]:
    """Each file is opened once. Everything downstream derives from the cached raw frames."""
    problems = []
    for file in files or []:
        if file.name in sess.raw or file.name in sess.documents:
            continue
        try:
            if router.lane_for(file.name) == "tabular":
                sess.raw[file.name] = tabular.read_raw(file)
            else:
                sess.add_document(file.name, documents.extract_pages(file), file.name)
        except Exception as exc:
            problems.append(f"{file.name}: {exc}")
    return problems


def wanted_tables(sess: store.Session) -> dict[str, tuple[str, str, int]]:
    """table_name -> (source file, sheet, header row), the header row overridden by the user."""
    wanted = {}
    for source, sheets in sess.raw.items():
        for sheet, raw in sheets.items():
            name = store.table_name(source, sheet if len(sheets) > 1 else None)
            detected, _ = tabular.detect_header(raw)
            wanted[name] = (source, sheet, int(sess.overrides.get(name, detected)))
    return wanted


def rebuild(sess: store.Session, wanted: dict) -> None:
    """Normalize every sheet, then card them — two passes, so every card sees every other table.

    Skipped entirely when nothing changed, which is most reruns.
    """
    # Session survives in session_state, but Streamlit re-imports a module when its file
    # changes — so an edit to profile.py leaves cards built by the *previous* SchemaCard class
    # sitting in sess.tables. isinstance is False across that reload, which is exactly the
    # signal we want: these were derived by code that no longer exists, so derive them again.
    fresh = all(isinstance(t.card, profile.SchemaCard) for t in sess.tables.values())
    current = {n: (t.source, t.sheet, t.header_row) for n, t in sess.tables.items()}
    if fresh and current == wanted:
        return

    for name in set(sess.tables) - set(wanted):
        sess.con.unregister(name)
        del sess.tables[name]

    derived = {name: tabular.normalize(sess.raw[source][sheet], header_row)
               for name, (source, sheet, header_row) in wanted.items()}

    for name, (df, changelog) in derived.items():
        source, sheet, header_row = wanted[name]
        others = {n: d for n, (d, _) in derived.items() if n != name}
        # a csv has no sheet — read_raw keys it by the filename stem, which is not a fact
        # worth putting in the card
        card = profile.build_schema_card(
            df, name, changelog, others=others, header_row=header_row,
            sheet=sheet if len(sess.raw[source]) > 1 else None,
            header_chosen=name in sess.overrides)
        sess.add_table(name, df, card, changelog, source, sheet, header_row)


# --- turning the changelog into sentences ----------------------------------------

def change_summary(changelog: list) -> list[tuple[str, str, str]]:
    """(icon, headline, detail) per kind of change.

    The changelog is one entry per column per step — fourteen rows for the demo workbook,
    which reads as noise. Grouping by step turns it into six sentences a person can scan.
    """
    groups: dict[str, list] = {}
    for change in changelog:
        groups.setdefault(change.step, []).append(change)

    out = []
    for step, items in groups.items():
        n, total = len(items), sum(i.count for i in items)
        cols = ", ".join(f"`{i.data['column']}`" for i in items if "column" in i.data)

        if step == "rename":
            pairs = " · ".join(f'`{i.data["from"]}` → `{i.data["to"]}`' for i in items)
            out.append((":material/edit:", f"Renamed {n} columns to snake_case", pairs))
        elif step == "to_numeric":
            out.append((":material/tag:", f"Parsed {n} columns as numbers",
                        f"Stripped currency symbols, thousands separators and units from {cols}"))
        elif step == "serial_dates":
            out.append((":material/calendar_month:",
                        f"Converted {n} column from Excel serial numbers to real dates",
                        f"{cols} held integers like `45543`, which Excel stores for `2024-09-08`"))
        elif step == "to_datetime":
            out.append((":material/calendar_month:", f"Parsed {n} columns as dates", cols))
        elif step == "drop_totals":
            out.append((":material/delete_sweep:", f"Removed {total} summary row",
                        "A `TOTAL` row was mixed in with the data — left in, it would have been "
                        "counted as a record and doubled every sum"))
        elif step == "dedupe":
            out.append((":material/content_copy:", f"Removed {total} duplicate rows",
                        "Identical in every column"))
        elif step == "drop_empty_rows":
            out.append((":material/backspace:", f"Removed {total} blank rows", ""))
        elif step == "drop_empty_columns":
            out.append((":material/backspace:", f"Removed {total} empty columns",
                        "No header and no values"))
    return out


def columns_frame(card) -> pd.DataFrame:
    """The Schema Card's columns, as something sortable instead of a wall of monospace."""
    return pd.DataFrame([{
        "Column": c.name,
        "Type": c.kind,
        "Values": c.summary,
        "Missing": c.null_pct * 100,
        "Distinct": c.distinct,
        "Originally called": c.original or "",
    } for c in card.columns])


COLUMN_CONFIG = {
    "Values": st.column_config.TextColumn(width="large"),
    "Missing": st.column_config.ProgressColumn(
        format="%.1f%%", min_value=0, max_value=100,
        help="Share of rows where this column is empty"),
    "Distinct": st.column_config.NumberColumn(format="%d"),
}


# --- panels ----------------------------------------------------------------------

def upload_panel():
    with st.sidebar:
        st.title("DataSurf")
        files = st.file_uploader("Upload data", type=ACCEPTED, accept_multiple_files=True,
                                 help="Spreadsheets become queryable tables. PDFs are stored as text.")
        st.caption(":material/lock: Your rows live in this session only and never reach disk. "
                   "Each question appends a trace to `data/traces/` — the SQL, the token "
                   "counts and the answer given, never the rows behind it.")
        return files


def tables_panel(sess: store.Session):
    """Rendered after rebuild, or it reports the state of the previous run."""
    with st.sidebar:
        if sess.tables:
            st.subheader("Queryable tables")
            st.dataframe(
                pd.DataFrame([{"Table": n, "Rows": len(t.df), "Cols": len(t.df.columns)}
                              for n, t in sess.tables.items()]),
                hide_index=True, width="stretch")
            st.caption("Registered in DuckDB. Query any of them by name once the agent lands.")
        if sess.documents:
            st.subheader("Documents")
            for name, doc in sess.documents.items():
                st.caption(f":material/description: {name} — {doc.page_count} pages")


def overview(table: store.Table):
    card = table.card
    removed = sum(c.count for c in table.changelog
                  if c.step in ("dedupe", "drop_totals", "drop_empty_rows"))
    flagged = sum(1 for c in card.columns if c.flags)

    with st.container(horizontal=True):
        st.metric("Rows kept", f"{len(table.df):,}", f"{-removed} removed" if removed else None,
                  delta_color="off", border=True,
                  help="Duplicates, blank rows and summary rows are not records")
        st.metric("Columns", len(table.df.columns), border=True)
        st.metric("Cleanups applied", len(table.changelog), border=True,
                  help="Every one is itemized below")
        st.metric("Columns to check", flagged, border=True,
                  help="Columns with more than 10% missing values")


def header_panel(sess: store.Session, name: str):
    """Detection is always visible, never silent — a confidently wrong guess is the danger."""
    table = sess.tables[name]
    raw = sess.raw[table.source][table.sheet]
    detected, margin = tabular.detect_header(raw)
    chosen = sess.overrides.get(name, detected)

    def remember():
        sess.overrides[name] = int(st.session_state[f"hdr::{name}"])

    with st.container(border=True):
        st.subheader("Where the column names are", divider=False)

        if chosen != detected:
            st.markdown(f"You set the column names to **row {chosen}**. "
                        f"DataSurf would have guessed row {detected}.")
        elif margin >= CLEAR_MARGIN:
            st.markdown(f"Column names found on **row {chosen}**. "
                        f"Everything above it was treated as a title block and skipped.")
        else:
            st.markdown(f"Column names are probably on **row {chosen}**, but it was a close call. "
                        f"Check the highlighted row below and change it if it looks wrong.")

        left, right = st.columns([1, 3])
        with left:
            st.number_input("Header row", min_value=0, max_value=max(len(raw) - 1, 0),
                            value=chosen, key=f"hdr::{name}", on_change=remember,
                            help="Change this and every column, type and stat re-derives instantly.")
            if chosen != detected:
                st.badge("Set by you", icon=":material/person:", color="violet")
            elif margin >= CLEAR_MARGIN:
                st.badge("Clear match", icon=":material/check:", color="green")
            else:
                st.badge("Worth a look", icon=":material/help:", color="orange")
        with right:
            st.caption(f"Your file exactly as it arrived — the highlighted row is the one being "
                       f"used for column names.")
            st.dataframe(highlight_header(raw.head(PREVIEW_ROWS), chosen),
                         width="stretch", height=260)


def highlight_header(raw: pd.DataFrame, header_row: int):
    """Raw cells mix ints and strings in one column, which Arrow cannot type. Display only."""
    shown = raw.map(lambda v: "" if pd.isna(v) else str(v))
    return shown.style.apply(
        lambda row: [HEADER_TINT if row.name == header_row else ""] * len(row), axis=1)


def changes_panel(table: store.Table):
    with st.container(border=True):
        st.subheader("What DataSurf changed")
        st.caption("Your file was not modified. These are the fixes applied to the copy in memory.")
        for icon, headline, detail in change_summary(table.changelog):
            st.markdown(f"{icon} **{headline}**")
            if detail:
                st.caption(detail)


def columns_panel(table: store.Table):
    with st.container(border=True):
        st.subheader("What is in each column")
        st.dataframe(columns_frame(table.card), hide_index=True, width="stretch",
                     column_config=COLUMN_CONFIG)

        if table.card.findings:
            st.markdown(":material/lightbulb: **Worth knowing**")
            for finding in table.card.findings:
                st.caption(finding.human)


def model_view_panel(table: store.Table):
    text = table.card.to_text()
    with st.expander(f"What the model will see — about {len(text) // 4} tokens",
                     icon=":material/smart_toy:"):
        st.caption("The rows themselves never enter a prompt. This card does, once per table, "
                   "and it is all the model gets before it decides what to query.")
        st.code(text, language=None)

    with st.expander(f"Preview the cleaned rows — {len(table.df):,} of them",
                     icon=":material/table_rows:"):
        st.dataframe(table.df.head(50), width="stretch")


def documents_panel(sess: store.Session):
    if not sess.documents:
        return
    st.subheader("Documents")
    st.caption("Text extraction only. Search, chunking and page citation arrive with the "
               "document agent.")
    for name, doc in sess.documents.items():
        with st.container(border=True):
            st.markdown(f":material/description: **{name}** — {doc.page_count} pages")
            page = st.number_input("Page", 1, max(doc.page_count, 1), 1, key=f"pg::{name}")
            st.text(documents.preview(doc.pages, int(page))
                    or "(no extractable text on this page — it may be a scan)")


PRESETS = {
    "Data health": (health, ":material/stethoscope:"),
    "Auto-EDA": (eda, ":material/insights:"),
    "Correlations": (matrices, ":material/grid_on:"),
    "Anomalies": (anomalies, ":material/warning:"),
    "Segments": (segments, ":material/pie_chart:"),
}


def run_preset(sess: store.Session, label: str, name: str):
    """Deterministic, so it carries a trace proving it spent nothing.

    The findings land in `sess.stats` on the way out, which is what makes tier 1 richer for
    the rest of the session: a question about outliers now costs one model call instead of a
    generated query (plans/datasurf-plan.md §6).
    """
    module, _ = PRESETS[label]
    trace = Trace.new(f"preset: {label}")
    with trace.span("preset", preset=label, table=name) as event:
        report = module.report(sess.tables[name].df, name)
        event.out = {"findings": len(report.findings), "frames": list(report.frames),
                     "stats_cached": len(report.stats)}

    sess.stats |= report.stats
    sess.findings[f"{label} · {name}"] = report
    answer = graph.Answer(f"{label} — {name}", trace, preset=label, report=report,
                          text=report.headline, frames=report.frames)
    sess.turns.append(answer)
    sess.traces.append(trace)


def preset_row(sess: store.Session, name: str) -> None:
    with st.container(border=True):
        st.markdown(":material/bolt: **Run an analysis** — no model, no key, same answer "
                    "every time")
        with st.container(horizontal=True):
            for label, (_, icon) in PRESETS.items():
                st.button(label, icon=icon, key=f"preset::{label}",
                          on_click=run_preset, args=(sess, label, name))


def render_report(report) -> None:
    for item in report.findings:
        st.markdown(f":material/check_circle: {item.human}")
    for title, frame in report.frames.items():
        with st.expander(title, icon=":material/table:"):
            st.dataframe(frame, width="stretch")


def render_answer(answer) -> None:
    """One turn, rendered the same way live and when replayed from history."""
    if answer.error:
        st.error(answer.error, icon=":material/error:")
    if answer.text:
        st.markdown(answer.text)
    if answer.report is not None:
        render_report(answer.report)
    if answer.tier:
        tier_badge(answer)
    if answer.sql:
        with st.expander("The SQL that produced it", icon=":material/code:"):
            st.code(answer.sql, language="sql")
            cost = ledger.usd(answer.trace.usage.usd)
            st.caption(f"{answer.trace.calls} model calls · {answer.trace.usage.tok_in:,} tokens "
                       f"in · {answer.trace.usage.tok_out:,} out · {cost} · "
                       f"{answer.trace.ms:,} ms")
    if answer.rows is not None and not answer.rows.empty:
        st.dataframe(answer.rows, hide_index=True, width="stretch")
        if answer.truncated:
            st.caption(f"Cut off at {len(answer.rows):,} rows.")


TIER_NOTES = {
    "P": ("matched a known question — answered from the Schema Card", "green"),
    "0": ("the answer was already in the Schema Card", "green"),
    "1": ("one precomputed statistic, computed by DataSurf not the model", "green"),
    "2": ("a generated query, then checked", "blue"),
}


def tier_badge(answer) -> None:
    note, colour = TIER_NOTES.get(answer.tier, ("", "grey"))
    with st.container(horizontal=True):
        st.badge(f"tier {answer.tier}", color=colour, icon=":material/route:")
        st.badge(f"{answer.trace.calls} model call{'s' if answer.trace.calls != 1 else ''}",
                 color="grey", icon=":material/token:")
        if answer.verified:
            st.badge("verified", color="green", icon=":material/verified:")
    st.caption(note + (f" · {answer.verified}" if answer.verified else "")
               + (f" · {answer.reason}" if answer.reason else ""))


def ask_panel(sess: store.Session) -> None:
    """The turns live on Session, so the ledger below reads them without being handed them."""
    if not sess.tables:
        st.info("Upload a spreadsheet to ask questions about it. The document lane arrives "
                "with M3.", icon=":material/upload_file:")
        return

    st.caption("Answered at the cheapest tier that can be right: a known question costs "
               "nothing, a card lookup one call, a generated query two and a check. The rows "
               "never enter the prompt — only the Schema Cards do.")
    preset_row(sess, st.session_state.get("preset_table", next(iter(sess.tables))))

    for turn in sess.turns:
        with st.chat_message("user"):
            st.write(turn.question)
        with st.chat_message("assistant"):
            render_answer(turn)

    if not config.API_KEY:
        st.warning("Set `OPENROUTER_API_KEY` in `.env` and restart to ask questions. "
                   "Everything else on this page runs without a key.",
                   icon=":material/key_off:")

    question = st.chat_input("Ask about your data", submit_mode="disable",
                             disabled=not config.API_KEY)
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Routing, then answering at the cheapest tier that works…"):
            answer = graph.ask(sess, question)
        render_answer(answer)

    sess.turns.append(answer)
    sess.traces.append(answer.trace)


# --- page ------------------------------------------------------------------------

sess = session()
files = upload_panel()

for problem in read_new_files(sess, files):
    st.error(problem, icon=":material/error:")

rebuild(sess, wanted_tables(sess))
tables_panel(sess)          # after rebuild, or it reports the previous run's state
ledger_slot = st.sidebar.container()

if not sess.tables and not sess.documents:
    st.title("Upload a spreadsheet to begin")
    st.markdown(
        "DataSurf reads messy files the way a person would: it finds the real header row, "
        "strips currency symbols, turns Excel's date integers back into dates, drops the "
        "stray `TOTAL` row, and tells you everything it did."
    )
    st.caption("Try `data/fixtures/messy_sales.xlsx` — its headers start on row 4. Add "
               "`data/fixtures/reps.csv` alongside it and DataSurf will spot the join between them.")
    st.stop()

understood, ask = st.tabs(["What was understood", "Ask"])

with understood:
    if sess.tables:
        labels = {n: f"{t.source}" + (f"  ·  {t.sheet}" if t.sheet and len(sess.raw[t.source]) > 1 else "")
                  for n, t in sess.tables.items()}
        name = st.selectbox("Showing", list(sess.tables), format_func=lambda n: labels[n],
                            key="preset_table",
                            help="Every sheet of every file is already registered and "
                                 "queryable. This picks which one you are looking at, and "
                                 "which one the Ask tab's preset buttons run against.")
        table = sess.tables[name]

        overview(table)
        header_panel(sess, name)

        left, right = st.columns([1, 1])
        with left:
            changes_panel(table)
        with right:
            columns_panel(table)

        model_view_panel(table)

    documents_panel(sess)

with ask:
    ask_panel(sess)

# Last, because it reports what the question above just spent — the slot was reserved before
# the model call so the sidebar renders in its usual place rather than below it.
with ledger_slot:
    ledger.panel(sess.traces)
