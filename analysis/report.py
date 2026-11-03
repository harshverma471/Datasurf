"""What every analysis returns. One type, because five modules and the UI all speak it.

`Finding` is reused from `ingest/profile.py` rather than redefined: it already carries the
two renderings this project needs — `text` for a prompt where tokens are budgeted, `human`
for a screen where the reader needs to know why it matters.
"""

from dataclasses import dataclass, field

import pandas as pd

from ingest.profile import ID_NAME, ID_UNIQUE, Finding


@dataclass
class Report:
    """One preset's output. No LLM touched any of it."""
    preset: str
    table: str
    headline: str = ""
    findings: list[Finding] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    # (table, column, stat) -> (value, provenance). Merged into Session.stats so column_stat
    # can serve these at tier 1 for the rest of the session (plans/datasurf-plan.md §6).
    stats: dict = field(default_factory=dict)

    def to_text(self, limit: int = 12) -> str:
        """The compact rendering. A preset's findings feed a prompt; its frames do not."""
        lines = [f"{self.preset} · {self.table}" + (f" — {self.headline}" if self.headline else "")]
        lines += [f"  {f.text}" for f in self.findings[:limit]]
        if len(self.findings) > limit:
            lines.append(f"  … {len(self.findings) - limit} more")
        return "\n".join(lines)


def finding(text: str, human: str) -> Finding:
    return Finding(text, human)


def measures(df: pd.DataFrame) -> list[str]:
    """Numeric columns that are quantities, not identifiers — the ones worth averaging."""
    out = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        rows = len(df)
        unique = df[col].nunique(dropna=True) / rows if rows else 0
        if ID_NAME.search(str(col)) or (pd.api.types.is_integer_dtype(df[col])
                                        and unique >= ID_UNIQUE):
            continue
        out.append(col)
    return out


def dimensions(df: pd.DataFrame, max_cardinality: int = 20) -> list[str]:
    """Low-cardinality non-numeric columns: the things worth grouping by."""
    return [c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and 1 < df[c].nunique(dropna=True) <= max_cardinality]
