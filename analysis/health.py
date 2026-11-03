"""Data health: what is wrong with this table, computed not guessed.

The first preset because it is the one whose answer a model must never invent. "7 duplicate
rows" is a fact or it is nothing.

Everything here is pandas and numpy. The model's only job, later, is to write prose around
numbers it did not produce (plans/datasurf-plan.md §10).
"""

import pandas as pd

from analysis.report import Report, finding, measures
from ingest.profile import ID_NAME

IQR_FACTOR = 1.5         # Tukey's fence: "unusual". ~0.7% of clean normal data lands outside
FAR_OUT = 3.0            # Tukey's far-out fence: ~4.7σ, which clean data essentially never
                         # reaches. The count goes to tier 1 either way; only a far-out value
                         # earns a finding, because a report that flags two in every clean
                         # table teaches the reader to ignore it.
HIGH_MISSING = 0.10
MOSTLY_DATES = 0.80      # share of a text column that parses as a date before we say so


def report(df: pd.DataFrame, table: str) -> Report:
    out = Report("Data health", table)
    rows = len(df)

    out.findings += _missing(df, rows)
    out.findings += _duplicates(df, rows)
    out.findings += _constant(df)
    out.findings += _outliers(df, out)
    out.findings += _unparsed_dates(df)
    out.findings += _lost_padding(df)

    out.headline = (f"{len(out.findings)} thing{'s' if len(out.findings) != 1 else ''} worth "
                    f"knowing across {rows:,} rows × {len(df.columns)} columns"
                    if out.findings else "nothing to flag")
    out.frames["Per column"] = _per_column(df, rows)
    return out


def _per_column(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return pd.DataFrame([{
        "Column": c,
        "Type": str(df[c].dtype),
        "Missing": df[c].isna().mean() * 100,
        "Distinct": int(df[c].nunique(dropna=True)),
        "Constant": df[c].nunique(dropna=True) <= 1,
    } for c in df.columns])


def _missing(df: pd.DataFrame, rows: int) -> list:
    found = []
    for col in df.columns:
        share = df[col].isna().mean()
        if share > HIGH_MISSING:
            found.append(finding(
                f"{col} is {share:.1%} null ({int(df[col].isna().sum()):,} rows)",
                f"`{col}` is empty in {share:.0%} of rows. Any average over it describes only "
                f"the {1 - share:.0%} that are filled — state that, or filter explicitly."))
    return found


def _duplicates(df: pd.DataFrame, rows: int) -> list:
    """Ingestion already dropped exact duplicates, so anything here is a *partial* duplicate:
    the same key appearing twice with different values, which is the harder problem."""
    found = []
    exact = int(df.duplicated().sum())
    if exact:
        found.append(finding(f"{exact:,} exact duplicate rows remain",
                             f"{exact:,} rows are identical in every column."))

    keys = [c for c in df.columns if ID_NAME.search(str(c))]
    for col in keys:
        repeated = df[col].dropna().duplicated().sum()
        if not repeated:
            continue
        worst = df[col].value_counts()
        found.append(finding(
            f"{col} repeats: {int(repeated):,} rows share a value with another "
            f"(max {int(worst.iloc[0])}×)",
            f"`{col}` looks like an identifier but is not unique — {int(repeated):,} rows "
            f"reuse a value, one of them {int(worst.iloc[0])} times. Joining on it will "
            f"multiply rows."))
    return found


def _constant(df: pd.DataFrame) -> list:
    dead = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if not dead:
        return []
    return [finding(
        f"constant columns: {', '.join(map(str, dead))}",
        f"{', '.join(f'`{c}`' for c in dead)} hold one value for every row, so they cannot "
        f"explain anything. They are safe to ignore.")]


def _outliers(df: pd.DataFrame, out: Report) -> list:
    """Tukey's rule, which needs no distributional assumption — unlike a z-score."""
    found = []
    for col in measures(df):
        values = df[col].dropna()
        if len(values) < 8:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        span = q3 - q1
        if span <= 0:
            continue
        low, high = q1 - IQR_FACTOR * span, q3 + IQR_FACTOR * span
        beyond = values[(values < low) | (values > high)]
        out.stats[(out.table, col, "outliers")] = (
            len(beyond), f"the data health preset, Tukey {IQR_FACTOR}×IQR rule")

        far_low, far_high = q1 - FAR_OUT * span, q3 + FAR_OUT * span
        far = values[(values < far_low) | (values > far_high)]
        if not len(far):
            continue
        worst = far.iloc[(far - values.median()).abs().argmax()]
        found.append(finding(
            f"{col} has {len(far):,} value(s) beyond {FAR_OUT}×IQR, worst {worst:,.10g} "
            f"(typical range {low:,.10g} – {high:,.10g})",
            f"{len(far):,} of {len(values):,} `{col}` values sit far outside the rest — the "
            f"worst is {worst:,.2f} against a median of {values.median():,.2f} and a typical "
            f"range of {low:,.2f} – {high:,.2f}. At that distance it is usually a unit "
            f"mix-up or a data-entry error rather than a real extreme, and it will drag any "
            f"mean you compute."))
    return found


def _unparsed_dates(df: pd.DataFrame) -> list:
    """A text column that is mostly dates means ingestion's parse did not fire."""
    found = []
    for col in df.columns:
        series = df[col].dropna()
        if (pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series) or not len(series)):
            continue
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        share = parsed.notna().mean()
        if share >= MOSTLY_DATES and share < 1.0:
            found.append(finding(
                f"{col} is {share:.0%} parseable dates, stored as text — mixed formats",
                f"`{col}` holds dates but stayed text, which means {1 - share:.0%} of its "
                f"values are in a different format. Sorting it sorts alphabetically."))
        elif share == 1.0:
            found.append(finding(
                f"{col} is entirely dates, stored as text",
                f"`{col}` is all dates but stayed text. Date arithmetic on it will fail."))
    return found


def _lost_padding(df: pd.DataFrame) -> list:
    """Zero-padded codes read as integers lose their padding before we ever see them.

    This is a caveat, not a finding: the evidence is gone by the time the column is numeric.
    Saying "the padding may be lost" is honest; saying "it was lost" would be a guess.
    """
    found = []
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]) or not ID_NAME.search(str(col)):
            continue
        widths = df[col].dropna().astype(str).str.len()
        if len(widths) and widths.nunique() > 1:
            found.append(finding(
                f"{col} is numeric with {widths.min()}–{widths.max()} digit values — any "
                f"zero-padding is already gone",
                f"`{col}` is stored as a number and its values are "
                f"{widths.min()}–{widths.max()} digits long. If the source codes were "
                f"zero-padded (`00742`), that padding was lost when the column became "
                f"numeric. Re-upload with the column as text if the padding matters."))
    return found
