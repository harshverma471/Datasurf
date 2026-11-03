"""What looks wrong here, without being asked.

Two scans, because they catch different things. A **z-score** finds a value extreme against
the whole column. A **per-segment deviation** finds a value that is unremarkable overall but
wrong *for its group* — a discount of 40% is fine until you notice every one of them is in
one region. The second is where the real findings live, and it is invisible to the first.

Every per-segment claim carries its n. A segment of three rows deviating by 300% is arithmetic,
not a finding (plans/datasurf-plan.md §10).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp

from analysis.report import Report, dimensions, finding, measures

Z = 3.0
# A normal sample puts ~0.27% of its values beyond |z|>3, so 240 clean rows yield about one
# and 100,000 yield 270. A fixed count threshold is therefore wrong at both ends: it cries
# wolf on a small table and drowns a large one. Compare against what the distribution would
# produce and report only a genuine excess.
EXPECTED_SHARE = float(2 * sp.norm.sf(Z))
MIN_EXCESS = 3.0         # times the expected count
MIN_COUNT = 3            # floor, so a tiny table cannot trip on one value
MIN_SEGMENT = 8          # below this a segment mean is not worth comparing
MIN_DEVIATION = 0.20     # relative gap from the overall mean before it is worth saying


def report(df: pd.DataFrame, table: str) -> Report:
    out = Report("Anomaly hunt", table)
    out.findings += _extreme_values(df, out)
    out.findings += _odd_segments(df, out)
    out.headline = (f"{len(out.findings)} anomal{'y' if len(out.findings) == 1 else 'ies'}"
                    if out.findings else "nothing stands out")
    return out


def _extreme_values(df: pd.DataFrame, out: Report) -> list:
    found = []
    for col in measures(df):
        values = df[col].dropna()
        sigma = values.std()
        if len(values) < MIN_SEGMENT or not sigma or np.isnan(sigma):
            continue
        z = (values - values.mean()).abs() / sigma
        beyond = values[z > Z]
        out.stats[(out.table, col, "extreme_count")] = (
            len(beyond), f"the anomaly preset, |z| > {Z}")

        expected = len(values) * EXPECTED_SHARE
        if len(beyond) < max(MIN_COUNT, MIN_EXCESS * expected):
            continue
        worst = beyond.iloc[(beyond - values.mean()).abs().argmax()]
        found.append(finding(
            f"{col}: {len(beyond):,} values beyond |z|>{Z} where ~{expected:.1f} would be "
            f"normal, worst {worst:,.10g}",
            f"{len(beyond):,} `{col}` values sit more than {Z} standard deviations from the "
            f"mean — against roughly {expected:.1f} that a normal column of {len(values):,} "
            f"rows would produce on its own. The furthest is {worst:,.2f} against a mean of "
            f"{values.mean():,.2f}. Either there are real extremes here, or the column is not "
            f"normally distributed — the auto-EDA brief's skew figure tells you which."))
    return found


def _odd_segments(df: pd.DataFrame, out: Report) -> list:
    """A segment whose mean is far from the overall mean, reported with its sample size."""
    found = []
    for metric in measures(df):
        overall = df[metric].mean()
        if not overall or pd.isna(overall):
            continue
        for dimension in dimensions(df):
            grouped = df.groupby(dimension, observed=True)[metric].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= MIN_SEGMENT]
            if len(grouped) < 2:
                continue
            gap = (grouped["mean"] - overall) / abs(overall)
            worst = gap.abs().idxmax()
            if abs(gap[worst]) < MIN_DEVIATION:
                continue
            direction = "above" if gap[worst] > 0 else "below"
            found.append(finding(
                f"{metric} in {dimension}={worst} is {abs(gap[worst]):.0%} {direction} "
                f"the overall mean (n={int(grouped['count'][worst]):,})",
                f"`{metric}` averages {grouped['mean'][worst]:,.2f} where "
                f"`{dimension}` is `{worst}`, against {overall:,.2f} overall — "
                f"{abs(gap[worst]):.0%} {direction}, over "
                f"{int(grouped['count'][worst]):,} rows."))
    return found
