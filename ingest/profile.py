"""The Schema Card — what crosses the context boundary.

Raw rows stay on the left of the line and die with the session; only this card goes into a
prompt (plans/datasurf-plan.md §5). It is written once per table at ingest and becomes the
stable cached prefix of every prompt about that table, so it is built here and rendered
verbatim by the UI — never assembled in app.py.

Three details are load-bearing:

  the rename trail    `[was "Revenue ($)"]` lets the model answer in the user's vocabulary
                      while querying the canonical column
  the notes block     findings computed deterministically, so the model gets clustered
                      missingness and join candidates without spending a tool call
  full precision      μ=1,204.37, not μ=1,204. A rounded figure in the card becomes a wrong
                      figure in an answer.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

MAX_COLUMNS = 12          # beyond this the card says "… N more columns"
CATEGORY_MAX = 40         # distinct values above which a string column is text, not a category
LIST_ALL_BELOW = 7        # categories listed in full rather than sampled
HIGH_MISSING = 0.10
CLUSTER_SHARE = 0.70      # share of a column's nulls in one category before it is worth saying
OVERLAP = 0.80            # value overlap before two columns are a candidate join key
KEY_UNIQUE = 0.95         # how unique the *other* side must be to count as a join target

# An id is named like one, or is very nearly unique. Cardinality alone is not enough — `units`
# in a 240-row sheet has 195 distinct values and is still a measure.
ID_NAME = re.compile(r"(^|_)(id|ids|key|code|no|num|number|sku|ref|uuid)$", re.I)
ID_UNIQUE = 0.95      # public since M2.4: analysis/eda.py guesses column roles with the same
                      # rule, and two copies of a threshold drift apart


@dataclass
class Finding:
    """Something the profiler noticed, said twice.

    `text` goes in the card, where tokens are budgeted and the model wants it terse.
    `human` goes on screen, where the reader needs to know why it matters.
    """
    text: str
    human: str


@dataclass
class ColumnCard:
    name: str
    kind: str
    summary: str
    null_pct: float
    distinct: int
    mean: float | None = None
    original: str | None = None      # pre-normalization name, when it changed
    flags: list[str] = field(default_factory=list)


@dataclass
class SchemaCard:
    table: str
    rows: int
    columns: list[ColumnCard]
    # two kinds of note, kept apart because they have two audiences: the model reads them as
    # one block, the UI shows provenance beside the changelog and findings on their own
    notes: list[str] = field(default_factory=list)        # what ingestion did to the data
    findings: list[Finding] = field(default_factory=list)  # what the data itself says
    sheet: str | None = None
    header_row: int = 0
    header_chosen: bool = False      # the user picked this row; the detector did not

    def to_text(self, max_columns: int = MAX_COLUMNS) -> str:
        head = f"table {self.table}   {self.rows:,} rows × {len(self.columns)} cols"
        if self.sheet:
            head += f' · sheet "{self.sheet}"'
        # the distinction matters to whoever reads this card: a guess and a human's answer
        # deserve different amounts of trust
        head += (f" · header set by user to row {self.header_row}" if self.header_chosen
                 else f" · header detected row {self.header_row}")

        width = max((len(c.name) for c in self.columns[:max_columns]), default=4) + 2
        lines = [head, ""]
        for c in self.columns[:max_columns]:
            line = (f"{c.name:<{width}}{c.kind:<10}{c.summary:<43} "     # the trailing space is
                    f"null {c.null_pct * 100:>5.1f}%")                   # load-bearing when it overflows
            if c.original:
                line += f'   [was "{c.original}"]'
            for flag in c.flags:
                line += f"   {flag}"
            lines.append(line.rstrip())

        if len(self.columns) > max_columns:
            lines.append(f"… {len(self.columns) - max_columns} more columns")

        if self.notes or self.findings:
            lines += ["", "notes  " + f"\n{'':7}".join(
                self.notes + [f.text for f in self.findings])]

        text = "\n".join(lines)
        return text + f"\n\ncost   ≈ {len(text) // 4:,} tokens   · the same sheet as raw CSV " \
                      f"≈ {self._raw_tokens():,} tokens (est.)"

    def _raw_tokens(self) -> int:
        return int(self.rows * len(self.columns) * 3.2)   # ~13 chars a cell, ~4 chars a token


def build_schema_card(df: pd.DataFrame, table: str, changelog: list, *,
                      sheet: str | None = None, header_row: int = 0,
                      header_chosen: bool = False,
                      others: dict[str, pd.DataFrame] | None = None) -> SchemaCard:
    from ingest.tabular import renames

    trail = renames(changelog)
    columns = [_column_card(df[c], c, trail.get(c), len(df)) for c in df.columns]
    return SchemaCard(
        table, len(df), columns,
        notes=_provenance(changelog),
        findings=_clustered_missingness(df) + _candidate_keys(df, others or {}),
        sheet=sheet, header_row=header_row, header_chosen=header_chosen,
    )


# --- per column ------------------------------------------------------------------

def _fmt(x: float, sep: bool = True) -> str:
    """Enough precision that the number in the card is the number in the answer."""
    if x != x:
        return "—"
    if x == int(x) and abs(x) < 1e15:
        return f"{int(x):,}" if sep else f"{int(x)}"
    return f"{x:,.2f}" if abs(x) >= 1 else f"{x:.4g}"


def _column_card(s: pd.Series, name: str, original: str | None, rows: int) -> ColumnCard:
    null_pct = float(s.isna().mean()) if rows else 0.0
    distinct = int(s.nunique(dropna=True))
    values = s.dropna()
    mean = None

    if pd.api.types.is_datetime64_any_dtype(s):
        kind = "date"
        summary = (f"{values.min():%Y-%m-%d} → {values.max():%Y-%m-%d}") if len(values) else "—"

    elif pd.api.types.is_bool_dtype(s):
        kind, summary = "bool", f"{int(values.sum())} true / {int((~values).sum())} false"

    elif pd.api.types.is_numeric_dtype(s):
        integral = pd.api.types.is_integer_dtype(s)
        kind = "int" if integral else "float"
        mean = float(values.mean()) if len(values) else None
        identifier = integral and (bool(ID_NAME.search(name))
                                   or (rows and distinct / rows >= ID_UNIQUE))
        span = (f"{_fmt(values.min(), not identifier)} → {_fmt(values.max(), not identifier)}"
                if len(values) else "—")
        if identifier:
            summary = f"{span}  ({distinct:,} distinct)"        # an identifier, not a measure
        elif mean is not None:
            summary = f"{span}  μ={_fmt(mean)}  σ={_fmt(float(values.std()))}"
        else:
            summary = span

    elif distinct <= CATEGORY_MAX:
        kind = f"cat({distinct})"
        top = [str(v) for v in values.value_counts().index[:LIST_ALL_BELOW]]
        summary = _clip(" | ".join(top) if distinct <= LIST_ALL_BELOW
                        else ", ".join(f'"{v}"' for v in top[:2]) + f", +{distinct - 2}")

    else:
        kind = "text"
        sample = [str(v) for v in values.head(2)]
        summary = _clip(", ".join(f'"{v[:18]}"' for v in sample) + f"  ({distinct:,} distinct)")

    flags = ["⚠ high missingness"] if null_pct > HIGH_MISSING else []
    return ColumnCard(name, kind, summary, null_pct, distinct, mean, original, flags)


def _clip(text: str, width: int = 43) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


# --- the notes block -------------------------------------------------------------

def _provenance(changelog: list) -> list[str]:
    """What ingestion did. The UI shows this next to the changelog; the model sees it inline."""
    notes = []
    # dedupe leads: it is the one note that changes every count in the card above it
    for change in sorted(changelog, key=lambda c: c.step != "dedupe"):
        if change.step == "dedupe":
            notes.append(f"dedupe removed {change.count:,} exact-duplicate rows")
        elif change.step == "drop_totals":
            notes.append(f"{change.count} summary row(s) removed from the data — "
                         f"column totals here are computed, not read")
        elif change.step == "serial_dates":
            notes.append(f"{change.data['column']} arrived as Excel serial integers")
    return notes


def _clustered_missingness(df: pd.DataFrame) -> list[Finding]:
    """Nulls that hide in one segment are a data-collection story, not noise."""
    categoricals = [c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c]) and 1 < df[c].nunique() <= 20]
    notes = []

    for col in df.columns:
        missing = df[col].isna()
        total = int(missing.sum())
        if not total or missing.mean() <= 0.05:
            continue
        for group in categoricals:
            if group == col:
                continue
            share = df.loc[missing, group].value_counts(normalize=True)
            if len(share) and share.iloc[0] >= CLUSTER_SHARE:
                where, pct = share.index[0], share.iloc[0]
                notes.append(Finding(
                    f"{col} nulls cluster in {group}={where} ({pct:.0%} of them)",
                    f"`{col}` is empty in {missing.mean():.0%} of rows, and {pct:.0%} of those "
                    f"gaps are in `{group} = {where}`. That is a collection gap, not random "
                    f"noise — averages over `{col}` will under-represent {where}.",
                ))
                break
    return notes


def _candidate_keys(df: pd.DataFrame, others: dict[str, pd.DataFrame]) -> list[Finding]:
    """Value overlap, not name matching — a column called `id` in both tables proves nothing."""
    found = []
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        mine = set(df[col].dropna().unique())
        if len(mine) < 2:
            continue
        for name, other in others.items():
            for their_col in other.columns:
                values = other[their_col].dropna()
                theirs = set(values.unique())
                # a join target must be near-unique on its side; without this, `region` matches
                # `reps.territory` at 100% overlap and is not a key at all
                if not theirs or len(theirs) / len(values) < KEY_UNIQUE:
                    continue
                overlap = len(mine & theirs) / len(mine)
                if overlap >= OVERLAP:
                    found.append((f"{col} → {name}.{their_col} ({overlap:.1%} overlap)",
                                  f"`{col}` matches `{their_col}` in **{name}** for "
                                  f"{overlap:.0%} of its values — these two tables can be "
                                  f"joined on it."))
    if not found:
        return []
    return [Finding(f"candidate keys: {', '.join(t for t, _ in found)}",
                    " ".join(h for _, h in found))]
