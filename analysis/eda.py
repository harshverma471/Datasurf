"""Auto-EDA: what this table *is*, before anyone asks it anything.

The Schema Card says what each column holds. This says what each column is *for* — a key, a
measure, a dimension — which is the distinction every later question depends on. Nothing here
is a model's opinion; the role guess is a rule over dtype, cardinality and name.
"""

import pandas as pd
from scipy import stats as sp

from analysis.report import Report, dimensions, finding, measures
from ingest.profile import ID_NAME, ID_UNIQUE

STRONG = 0.50            # |r| worth reporting as a relationship
SKEWED = 1.0             # |skew| beyond which the mean stops being the typical value
MIN_PAIRS = 8


def report(df: pd.DataFrame, table: str) -> Report:
    out = Report("Auto-EDA", table)
    roles = _roles(df)

    out.frames["Column roles"] = pd.DataFrame(
        [{"Column": c, "Role": r, "Distinct": int(df[c].nunique(dropna=True)),
          "Type": str(df[c].dtype)} for c, r in roles.items()])

    counts = pd.Series(list(roles.values())).value_counts()
    out.headline = " · ".join(f"{n} {role}{'s' if n != 1 else ''}"
                              for role, n in counts.items())
    out.findings += _role_note(roles)
    out.findings += _correlations(df, out)
    out.findings += _shape(df, out)
    out.findings += _dominant_categories(df)
    return out


def _roles(df: pd.DataFrame) -> dict[str, str]:
    """key · measure · dimension · date · text, by the same rule the Schema Card uses."""
    rows = len(df)
    roles = {}
    for col in df.columns:
        series = df[col]
        unique = series.nunique(dropna=True) / rows if rows else 0
        if pd.api.types.is_datetime64_any_dtype(series):
            roles[col] = "date"
        # The integer gate is load-bearing, and the Schema Card learned it first: cardinality
        # alone calls a 240-distinct float `revenue` a key, because every revenue figure
        # happens to be unique. A name like `rep_id` identifies at any dtype; high
        # cardinality only does for integers.
        elif ID_NAME.search(str(col)) or (pd.api.types.is_integer_dtype(series)
                                          and unique >= ID_UNIQUE):
            roles[col] = "key"
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            roles[col] = "measure"
        elif series.nunique(dropna=True) <= 20:
            roles[col] = "dimension"
        else:
            roles[col] = "text"
    return roles


def _role_note(roles: dict) -> list:
    by_role: dict[str, list] = {}
    for col, role in roles.items():
        by_role.setdefault(role, []).append(str(col))
    text = " | ".join(f"{role}: {', '.join(cols)}" for role, cols in by_role.items())
    return [finding(f"roles — {text}",
                    "Measures are what you aggregate, dimensions what you group by, keys what "
                    "you join on. Every question is some combination of the three.")]


def _correlations(df: pd.DataFrame, out: Report) -> list:
    """Pearson with a p-value, because |r|=0.6 on n=5 is noise and should not be reported."""
    cols = measures(df)
    found = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = df[[a, b]].dropna()
            if len(pair) < MIN_PAIRS or pair[a].nunique() < 2 or pair[b].nunique() < 2:
                continue
            r, p = sp.pearsonr(pair[a], pair[b])
            out.stats[(out.table, f"{a}~{b}", "pearson_r")] = (
                round(float(r), 6), f"the auto-EDA preset, over {len(pair):,} complete pairs")
            if abs(r) < STRONG or p >= 0.05:
                continue
            direction = "rises with" if r > 0 else "falls as"
            found.append(finding(
                f"{a} {direction} {b}: r={r:.2f} (n={len(pair):,}, p={p:.1e})",
                f"`{a}` and `{b}` move together, r={r:.2f} over {len(pair):,} rows — "
                f"{r ** 2:.0%} of the variation in one is shared with the other. "
                f"Correlation, not cause."))
    return found


def _shape(df: pd.DataFrame, out: Report) -> list:
    found = []
    for col in measures(df):
        values = df[col].dropna()
        if len(values) < MIN_PAIRS or values.nunique() < 3:
            continue
        skew = float(sp.skew(values))
        out.stats[(out.table, col, "skew")] = (
            round(skew, 6), "the auto-EDA preset, Fisher–Pearson skewness")
        if abs(skew) < SKEWED:
            continue
        tail = "right" if skew > 0 else "left"
        found.append(finding(
            f"{col} is {tail}-skewed (skew={skew:.2f}); median {values.median():,.10g} vs "
            f"mean {values.mean():,.10g}",
            f"`{col}` has a long {tail} tail, so its mean ({values.mean():,.2f}) sits away "
            f"from its median ({values.median():,.2f}). Report the median unless you mean "
            f"the total."))
    return found


def _dominant_categories(df: pd.DataFrame) -> list:
    found = []
    for col in dimensions(df):
        share = df[col].value_counts(normalize=True)
        if len(share) and share.iloc[0] >= 0.80:
            found.append(finding(
                f"{col} is {share.iloc[0]:.0%} '{share.index[0]}' — barely a dimension",
                f"`{col}` is `{share.index[0]}` for {share.iloc[0]:.0%} of rows. Grouping by "
                f"it compares one large group against a handful of rows."))
    return found
