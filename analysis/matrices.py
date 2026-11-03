"""Correlation and contingency — every pair at once, with the sample size attached.

No covariance matrix. Covariance is correlation without the normalization, so its numbers are
unreadable without knowing each column's scale, and nobody has ever acted on one. Correlation
answers the same question legibly; add covariance when something actually needs the units.

The sample size travels with every coefficient. `r = 0.9` over four complete pairs is noise
wearing a convincing number, and the whole reason this is deterministic code is so the
qualifier cannot get dropped on the way to the screen.
"""

import pandas as pd
from scipy import stats as sp

from analysis.report import Report, dimensions, finding, measures

MIN_PAIRS = 8
STRONG = 0.50
CRAMER_STRONG = 0.30     # above this, two categoricals carry overlapping information


def report(df: pd.DataFrame, table: str) -> Report:
    out = Report("Correlation & contingency", table)
    numeric, categorical = measures(df), dimensions(df)

    if len(numeric) >= 2:
        out.frames["Correlation (Pearson r)"] = df[numeric].corr().round(4)
        out.frames["Complete pairs"] = _pair_counts(df, numeric)
        out.findings += _strongest(df, numeric, out)

    for i, a in enumerate(categorical):
        for b in categorical[i + 1:]:
            table_, v, p, n = _cramers_v(df, a, b)
            out.frames[f"{a} × {b}"] = table_
            out.stats[(table, f"{a}×{b}", "cramers_v")] = (
                round(v, 6), f"the matrices preset, over {n:,} rows")
            if v >= CRAMER_STRONG and p < 0.05:
                out.findings.append(finding(
                    f"{a} and {b} are associated: Cramér's V={v:.2f} (n={n:,}, p={p:.1e})",
                    f"`{a}` and `{b}` are not independent (V={v:.2f}). Grouping by both "
                    f"splits the data finely without adding much new information."))

    if not out.findings:
        out.headline = ("no relationship above the reporting threshold — the columns are "
                        "close to independent")
    else:
        out.headline = f"{len(out.findings)} relationship(s) above threshold"
    return out


def _pair_counts(df: pd.DataFrame, numeric: list) -> pd.DataFrame:
    """How many rows each coefficient was actually computed over. A correlation matrix that
    hides this lets one 95%-null column quietly contribute a coefficient from twelve rows."""
    return pd.DataFrame(
        [[int(df[[a, b]].dropna().shape[0]) for b in numeric] for a in numeric],
        index=numeric, columns=numeric)


def _strongest(df: pd.DataFrame, numeric: list, out: Report) -> list:
    found = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < MIN_PAIRS or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            r, p = sp.pearsonr(pair[a], pair[b])
            rho, _ = sp.spearmanr(pair[a], pair[b])
            if abs(r) < STRONG or p >= 0.05:
                continue
            # Spearman catches a monotonic-but-curved relationship that Pearson understates.
            note = (f" (Spearman ρ={rho:.2f} — the relationship is monotonic but not linear)"
                    if abs(rho) - abs(r) > 0.15 else "")
            found.append(finding(
                f"{a} ~ {b}: r={r:.2f} over {len(pair):,} pairs{note}",
                f"`{a}` and `{b}` correlate at r={r:.2f} across {len(pair):,} complete "
                f"pairs{note}."))
    return found


def _cramers_v(df: pd.DataFrame, a: str, b: str):
    """Chi-square normalized to 0–1, so two 4×5 tables are comparable."""
    crossed = pd.crosstab(df[a], df[b])
    n = int(crossed.values.sum())
    if crossed.shape[0] < 2 or crossed.shape[1] < 2 or not n:
        return crossed, 0.0, 1.0, n
    chi2, p = sp.chi2_contingency(crossed)[:2]
    smaller = min(crossed.shape) - 1
    return crossed, float((chi2 / n / smaller) ** 0.5) if smaller else 0.0, float(p), n
