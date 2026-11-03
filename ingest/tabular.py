"""csv/xlsx in → a clean DataFrame plus a record of everything that was done to it.

Three public functions, in the order they run:

    read_raw      file   -> {sheet_name: frame}      nothing decided yet, header=None throughout
    detect_header frame  -> (row_index, confidence)  the module's hard part
    normalize     frame, header_row -> (df, changelog)

Reading is separated from deciding on purpose: the raw frames are cached on the Session, so
changing the header row re-derives everything without touching the file again.

No LLM anywhere in this file — evals/test_ingest.py asserts that.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SCAN_ROWS = 20      # how far down to look for a header
LOOKAHEAD = 6       # data rows consulted when testing for a header→data type shift

# Excel's 1900 date system, as integers. 20000 ≈ 1954, 60000 ≈ 2064 — wide enough to be useful,
# narrow enough that a year column (2023) or a small id never falls inside it.
SERIAL_MIN, SERIAL_MAX = 20_000, 60_000
SERIAL_ORIGIN = "1899-12-30"

_DATE_NAME = re.compile(r"date|day|month|year|time|_at$|_dt$", re.I)
_TOTALS = re.compile(r"^\s*(grand\s+)?(total|subtotal|sum|totals)\s*:?\s*$", re.I)
_CURRENCY = re.compile(r"[$£€¥,\s%]")

NUMERIC_TOLERANCE = 0.95   # a column converts if this share of its non-null values are numbers


@dataclass
class Change:
    """One transformation, in the order it happened. The changelog is a list of these."""
    step: str
    detail: str
    count: int = 0
    data: dict = field(default_factory=dict)


# --- read ------------------------------------------------------------------------

def read_raw(file, name: str | None = None) -> dict[str, pd.DataFrame]:
    """Every sheet, unparsed. header=None so no column names are assumed yet.

    Accepts a path or a Streamlit UploadedFile. Values stay as written — CSVs are read as
    strings rather than type-inferred, so both lanes reach detect_header looking the same.
    """
    name = name or getattr(file, "name", str(file))
    suffix = Path(name).suffix.lower()

    if suffix == ".csv":
        return {Path(name).stem: pd.read_csv(file, header=None, dtype=str, keep_default_na=True)}
    return pd.read_excel(file, sheet_name=None, header=None)


# --- header detection ------------------------------------------------------------

def _numericish(value) -> bool:
    """True for 45543, '45543', '$94,897.17' — anything that is a number once punctuation goes."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, (pd.Timestamp,)):
        return False
    try:
        float(_CURRENCY.sub("", str(value)))
        return True
    except ValueError:
        return False


def _score(raw: pd.DataFrame, i: int) -> float:
    """How much row i looks like a header. Four signals, all scaled by how full the row is.

    Scaling by fill is what stops a one-cell title banner from scoring well on the other three,
    which it otherwise would — a banner is 100% string and 100% unique across its single cell.
    """
    # forward-filled, because that is how normalize will read it: a merged header cell arrives
    # as value-then-NaN, and scoring the gaps as missing makes a real header look emptier than
    # the data underneath it — on a narrow sheet that is enough to lose
    row = raw.iloc[i].ffill()
    filled = row[row.notna()]
    fill = len(filled) / len(row) if len(row) else 0.0
    if not len(filled):
        return 0.0

    stringy = sum(not _numericish(v) for v in filled) / len(filled)
    unique = filled.astype(str).nunique() / len(filled)

    # the header→data boundary: cells that are text here but numbers in the rows underneath
    below = raw.iloc[i + 1: i + 1 + LOOKAHEAD]
    shifted = 0
    for col in raw.columns:
        if _numericish(row[col]) or pd.isna(row[col]):
            continue
        values = below[col].dropna()
        if len(values) and sum(_numericish(v) for v in values) / len(values) > 0.6:
            shifted += 1
    shift = shifted / len(raw.columns) if len(raw.columns) else 0.0

    return 0.30 * fill + fill * (0.25 * stringy + 0.20 * unique + 0.25 * shift)


def detect_header(raw: pd.DataFrame) -> tuple[int, float]:
    """Best-scoring row in the first SCAN_ROWS, plus its margin over the runner-up.

    The margin is reported rather than the raw score because it is what the user needs: a
    clear winner means the guess is safe, a narrow one means look at the preview.
    """
    scores = [_score(raw, i) for i in range(min(SCAN_ROWS, len(raw)))]
    if not scores:
        return 0, 0.0

    best = int(np.argmax(scores))
    runner_up = max((s for i, s in enumerate(scores) if i != best), default=0.0)
    return best, round(max(0.0, scores[best] - runner_up), 3)


# --- normalization ---------------------------------------------------------------

def _snake(name) -> str:
    s = str(name).replace("%", " pct ").replace("#", " num ")
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)          # camelCase -> camel Case
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "unnamed"


def _unique_names(names: list[str]) -> list[str]:
    """Merged headers and blank columns collide once snake_cased. Suffix rather than silently drop."""
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def _to_numeric(s: pd.Series) -> tuple[pd.Series, bool] | None:
    """'$94,897.17' -> 94897.17, '(1,234)' -> -1234.

    Returns the column and whether punctuation was actually removed, so the changelog can say
    what happened instead of claiming a strip that never occurred. None when the column is not
    numeric at all — 'North' stays 'North'.
    """
    text = s.astype(str).str.strip()
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)     # accounting negatives
    bare = text.str.replace(_CURRENCY, "", regex=True)
    converted = pd.to_numeric(bare, errors="coerce")

    present = s.notna().sum()
    if not present or converted.notna().sum() / present < NUMERIC_TOLERANCE:
        return None
    return converted, bool((bare != text).any())


def normalize(raw: pd.DataFrame, header_row: int) -> tuple[pd.DataFrame, list[Change]]:
    """Everything below the header row, cleaned, with a record of each step."""
    log: list[Change] = []

    header = raw.iloc[header_row]
    df = raw.iloc[header_row + 1:].copy()

    # drop columns empty in both the header and the data, then forward-fill the header row so a
    # merged header cell (value, then None) names all the columns it spans
    keep = [c for c in raw.columns if df[c].notna().any() or pd.notna(header[c])]
    if len(keep) < len(raw.columns):
        log.append(Change("drop_empty_columns", "columns with no header and no data",
                          len(raw.columns) - len(keep)))
    df, header = df[keep], header[keep].ffill()

    blank_rows = int((~df.notna().any(axis=1)).sum())
    if blank_rows:
        df = df.dropna(how="all")
        log.append(Change("drop_empty_rows", "rows with no values at all", blank_rows))

    originals = [str(h) for h in header]
    df.columns = _unique_names([_snake(h) for h in header])
    for new, old in zip(df.columns, originals):
        if new != old:
            log.append(Change("rename", f'"{old}" → {new}', data={"from": old, "to": new}))

    df = _drop_totals(df, log)
    df = _coerce_types(df, log)

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.append(Change("dedupe", "exact duplicate rows removed", before - len(df)))

    return df.reset_index(drop=True), log


def _drop_totals(df: pd.DataFrame, log: list[Change]) -> pd.DataFrame:
    """A stray 'TOTAL' row mixed into the data. Only trailing rows, so a product named Total lives."""
    tail = df.tail(5)
    hits = [i for i, row in tail.iterrows()
            if any(_TOTALS.match(str(v)) for v in row.dropna())]
    if hits:
        df = df.drop(index=hits)
        log.append(Change("drop_totals", "summary rows mixed into the data", len(hits)))
    return df


def _untyped(s: pd.Series) -> bool:
    """Anything still carrying raw cell values. `== object` misses pandas 3's str dtype."""
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))


def _coerce_types(df: pd.DataFrame, log: list[Change]) -> pd.DataFrame:
    for col in df.columns:
        s = df[col]

        numeric_entry = None
        if _untyped(s):
            result = _to_numeric(s)
            if result is not None:
                numeric, stripped = result
                what = "stripped currency, separators and units" if stripped else "parsed as numbers"
                numeric_entry = Change("to_numeric", f"{col}: {what}",
                                       int(numeric.isna().sum() - s.isna().sum()),
                                       data={"column": col})
                df[col], s = numeric, numeric
                log.append(numeric_entry)

        # Excel serial integers. Name-gated on purpose: converting any integer that happens to
        # fall in the serial range would silently destroy a genuine id column.
        if _DATE_NAME.search(col) and pd.api.types.is_numeric_dtype(s):
            values = s.dropna()
            if len(values) and values.between(SERIAL_MIN, SERIAL_MAX).all() and (values % 1 == 0).all():
                df[col] = pd.to_datetime(s, unit="D", origin=SERIAL_ORIGIN)
                if numeric_entry is not None:
                    log.remove(numeric_entry)     # an intermediate step, not something that happened
                log.append(Change("serial_dates", f"{col}: Excel serial integers → dates",
                                  int(len(values)), data={"column": col}))
                continue

        if _DATE_NAME.search(col) and _untyped(df[col]):
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() / max(s.notna().sum(), 1) >= NUMERIC_TOLERANCE:
                df[col] = parsed
                log.append(Change("to_datetime", f"{col}: parsed as dates",
                                  int(parsed.notna().sum()), data={"column": col}))

    return df


def renames(changelog: list[Change]) -> dict[str, str]:
    """new_name -> original_name. The Schema Card's rename trail reads from this."""
    return {c.data["to"]: c.data["from"] for c in changelog if c.step == "rename"}
