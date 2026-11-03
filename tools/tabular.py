"""run_sql — the one tool that executes logic a model wrote.

This is the trust boundary of the whole project, and the reason DuckDB is here at all
(plans/datasurf-plan.md §4b): a model emitting pandas gives you `exec()`, while a model
emitting SQL gives you a *string* — parseable, loggable and refusable before it runs.

Three layers, cheapest first:

  parse        DuckDB tells us the statement type. `extract_statements` is the real parser,
               so a keyword blocklist would be both longer and more wrong — `SELECT set_id`
               trips a regex for `\\bset\\b`, and nothing trips the parser.
  no escape    `enable_external_access=false` on the session connection (see store.Session)
               kills COPY TO, ATTACH, INSTALL and any http read, whatever the SQL says.
  cap          every result is truncated, so a cartesian join is a wasted second rather
               than a dead tab.
"""

from dataclasses import dataclass

import duckdb
import pandas as pd

import config


@dataclass
class Result:
    rows: pd.DataFrame
    truncated: bool
    sql: str

    def to_markdown(self, limit: int = config.ANSWER_ROWS) -> str:
        """What the model is shown. The rows themselves stop here.

        `floatfmt` is load-bearing. tabulate defaults to `g` — six significant digits — which
        renders a revenue total of 3,383,130.45 as `3.38313e+06`, and a model cannot report a
        figure it was never shown. Same rule as the Schema Card: a rounded number in the
        context becomes a wrong number in the answer.
        """
        text = self.rows.head(limit).to_markdown(index=False, floatfmt=".10g")
        if len(self.rows) > limit:
            text += f"\n({len(self.rows) - limit} more rows not shown)"
        if self.truncated:
            text += f"\n(result was cut off at {config.MAX_RESULT_ROWS} rows)"
        return text


def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> Result:
    """Execute one read-only statement. Raises ValueError on anything else."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise ValueError("empty query")

    try:
        kinds = [s.type for s in con.extract_statements(sql)]
    except duckdb.Error as exc:
        raise ValueError(f"not valid SQL: {exc}") from exc

    if len(kinds) != 1:
        raise ValueError(f"one statement per query, got {len(kinds)}")
    if kinds[0] != duckdb.StatementType.SELECT:
        raise ValueError(f"read-only: SELECT or WITH, not {str(kinds[0]).split('.')[-1]}")

    # .limit() on the relation caps rows before they materialize; +1 reveals the truncation
    rows = con.sql(sql).limit(config.MAX_RESULT_ROWS + 1).df()
    truncated = len(rows) > config.MAX_RESULT_ROWS
    return Result(rows.head(config.MAX_RESULT_ROWS), truncated, sql)


# --- tier 1 ----------------------------------------------------------------------

AGGREGATES = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "min": lambda s: s.min(),
    "max": lambda s: s.max(),
    "std": lambda s: s.std(),
    "sum": lambda s: s.sum(),
    "count": lambda s: s.count(),
    "nulls": lambda s: s.isna().sum(),
    "distinct": lambda s: s.nunique(dropna=True),
    "mode": lambda s: s.mode().iloc[0] if len(s.mode()) else None,
}

NUMERIC_ONLY = ("mean", "median", "std", "sum")


def column_stat(tables: dict, table: str, column: str, stat: str,
                cached: dict | None = None) -> tuple[str, str]:
    """One table, one column, one stat → one value and where it came from.

    Tier 1's narrow signature *is* the optimization (plans/datasurf-plan.md §6): ~20 tokens
    in, ~30 out, and the value cannot be wrong because our code computed it. The thing this
    must never become is a `get_eda_results()` that dumps a blob into context — that recreates
    the exact problem the Schema Card solves.

    `cached` is `Session.stats`, where a preset that has already run leaves its findings. It is
    checked first, so tier 1 gets richer as the session goes on rather than staying frozen at
    whatever pandas can aggregate.

    Raises ValueError with a usable message rather than guessing — the planner is told to fall
    back to `run_sql`, and a silent wrong number is the one outcome worth avoiding.
    """
    if table not in tables:
        raise ValueError(f"no table named {table!r} — loaded: {', '.join(tables) or 'none'}")
    df = tables[table].df
    if column not in df.columns:
        raise ValueError(f"{table} has no column {column!r} — "
                         f"it has: {', '.join(map(str, df.columns))}")

    if cached and (key := (table, column, stat)) in cached:
        value, source = cached[key]
        return _fmt(value), source

    if stat not in AGGREGATES:
        raise ValueError(f"{stat!r} is not precomputed — use run_sql. "
                         f"Precomputed: {', '.join(AGGREGATES)}")

    series = df[column]
    if stat in NUMERIC_ONLY and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{column} is {series.dtype}, so {stat} is undefined — use run_sql")

    return _fmt(AGGREGATES[stat](series)), f"computed over {len(df):,} rows of {table}.{column}"


def _fmt(value) -> str:
    """Full precision, same rule as the Schema Card: a rounded number here is a wrong number
    in the answer."""
    if value is None or (isinstance(value, float) and value != value):
        return "undefined (no values)"
    if isinstance(value, float):
        return f"{value:,.10g}" if abs(value) >= 1 else f"{value:.10g}"
    if isinstance(value, (int, pd.Timestamp)):
        return f"{value:,}" if isinstance(value, int) else f"{value:%Y-%m-%d}"
    return str(value)
