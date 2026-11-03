"""Segment comparison: does this dimension actually explain this measure?

The preset exists because of one failure mode. "North is up 300%!" is the most quotable
sentence a dashboard can produce and it is usually two rows of data. So every row of the
output carries **effect size and sample size together**, and the verdict column states which
differences survive both.

Cohen's d over a t-test p-value as the headline number: with 240 rows almost anything reaches
p<0.05, and a d of 0.05 is a real difference nobody should act on. Both are reported; d is the
one that answers "is this big".
"""

import pandas as pd
from scipy import stats as sp

from analysis.report import Report, dimensions, finding, measures

MIN_SEGMENT = 8
SMALL, MEDIUM, LARGE = 0.2, 0.5, 0.8     # Cohen's conventional bands


def report(df: pd.DataFrame, table: str, dimension: str | None = None,
           metric: str | None = None) -> Report:
    out = Report("Segment comparison", table)
    dimension = dimension or _best_dimension(df)
    metric = metric or _best_metric(df, dimension)
    if not dimension or not metric:
        out.headline = "needs one low-cardinality column to group by and one numeric measure"
        return out

    out.preset = f"Segment comparison — {metric} by {dimension}"
    rows, overall = [], df[metric].dropna()

    for value, group in df.groupby(dimension, observed=True):
        values = group[metric].dropna()
        rest = df.loc[df[dimension] != value, metric].dropna()
        if len(values) < MIN_SEGMENT or len(rest) < MIN_SEGMENT:
            rows.append({"Segment": str(value), "n": len(values),
                         "Mean": values.mean() if len(values) else None,
                         "vs rest": None, "Cohen's d": None, "p": None,
                         "Verdict": "too few rows to compare"})
            continue
        d = _cohens_d(values, rest)
        p = float(sp.ttest_ind(values, rest, equal_var=False).pvalue)
        rows.append({"Segment": str(value), "n": len(values), "Mean": float(values.mean()),
                     "vs rest": float(values.mean() - rest.mean()), "Cohen's d": d,
                     "p": p, "Verdict": _verdict(d, p)})

    # to_numeric before abs: a segment too small to compare has None for d, and
    # Series.abs() raises on it before na_position ever applies.
    out.frames[f"{metric} by {dimension}"] = pd.DataFrame(rows).sort_values(
        "Cohen's d", key=lambda s: pd.to_numeric(s, errors="coerce").abs(),
        ascending=False, na_position="last")
    out.findings += _notable(out.frames[f"{metric} by {dimension}"], metric, dimension, overall)
    out.stats[(table, metric, f"mean_by_{dimension}")] = (
        ", ".join(f"{r['Segment']}={r['Mean']:,.10g}" for r in rows if r["Mean"] is not None),
        "the segment comparison preset")
    out.headline = (f"{metric} across {len(rows)} {dimension} segments; "
                    f"{sum(1 for r in rows if r['Verdict'].startswith(('large', 'medium')))} "
                    f"worth acting on")
    return out


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled standard deviation, so the number means the same thing across columns."""
    na, nb = len(a), len(b)
    pooled = (((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2)) ** 0.5
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def _verdict(d: float, p: float) -> str:
    size = abs(d)
    if p >= 0.05:
        return "not distinguishable from the rest"
    if size >= LARGE:
        return "large difference"
    if size >= MEDIUM:
        return "medium difference"
    if size >= SMALL:
        return "small difference"
    return "statistically significant but too small to matter"


def _notable(frame: pd.DataFrame, metric: str, dimension: str, overall: pd.Series) -> list:
    real = frame[frame["Verdict"].str.startswith(("large", "medium"))]
    if real.empty:
        return [finding(
            f"no {dimension} segment differs from the rest on {metric} by more than a small "
            f"effect",
            f"Every `{dimension}` group's `{metric}` sits within a small effect size of the "
            f"others. Differences you see between them are noise at this sample size — which "
            f"is itself worth knowing before anyone builds a plan on one.")]
    found = []
    # by name, not itertuples: "vs rest" and "Cohen's d" arrive as `_5` and `_6` positionally,
    # which keeps working until someone reorders a column and the numbers quietly swap.
    for _, row in real.iterrows():
        gap, d = row["vs rest"], row["Cohen's d"]
        found.append(finding(
            f"{dimension}={row['Segment']}: {metric} averages {row['Mean']:,.10g}, "
            f"{gap:+,.10g} vs the rest (d={d:.2f}, n={row['n']:,})",
            f"`{dimension} = {row['Segment']}` averages {row['Mean']:,.2f} on `{metric}`, "
            f"{gap:+,.2f} against everyone else — a {_verdict(d, row['p'])} "
            f"(Cohen's d={d:.2f}) over {int(row['n']):,} rows."))
    return found


def _best_dimension(df: pd.DataFrame) -> str | None:
    """The dimension with the most segments that are each big enough to compare."""
    candidates = [(c, (df[c].value_counts() >= MIN_SEGMENT).sum()) for c in dimensions(df)]
    usable = [(c, n) for c, n in candidates if n >= 2]
    return max(usable, key=lambda pair: pair[1])[0] if usable else None


def _best_metric(df: pd.DataFrame, dimension: str | None = None) -> str | None:
    """The measure this dimension actually separates — the one with a story.

    Relative spread was the first rule here and it picked whichever column happened to be
    noisiest: on a sales table it chose `units` (uniform 1–100, CV 0.6) over `revenue`, which
    is what anyone opening this preset came to see. Ranking by the largest segment gap answers
    the question the preset is for, and it is one groupby per measure.
    """
    candidates = measures(df)
    if not candidates:
        return None
    if dimension is None:
        return max(candidates, key=lambda c: df[c].std() / abs(df[c].mean() or 1))

    def separation(col: str) -> float:
        grouped = df.groupby(dimension, observed=True)[col].agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= MIN_SEGMENT]
        spread = df[col].std()
        if len(grouped) < 2 or not spread or pd.isna(spread):
            return 0.0
        return float((grouped["mean"].max() - grouped["mean"].min()) / spread)

    return max(candidates, key=separation)
