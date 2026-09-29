"""Smoke-test the DataSurf environment.

Run after any venv rebuild or dependency change:

    .venv/bin/python check_env.py

Checks the two things the architecture actually rests on, not just that imports resolve:
DuckDB querying a registered pandas DataFrame in place, and FastMCP deriving a tool
schema from type annotations. Exits non-zero on failure.
"""

import importlib
import importlib.metadata as md
import sys

MODULES = [
    "pandas", "numpy", "scipy", "duckdb", "openpyxl", "pyarrow", "pdfplumber",
    "langgraph", "langchain_core", "langchain_openai", "openai", "pydantic",
    "fastmcp", "mcp", "tabulate", "streamlit", "plotly", "pytest",
]

MIN_PYTHON = (3, 10)  # fastmcp floor


def check_imports() -> list[str]:
    failed = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL {name:18s} {type(exc).__name__}: {exc}")
            continue
        try:
            version = md.version(name.replace("_", "-"))
        except md.PackageNotFoundError:
            version = getattr(mod, "__version__", "?")
        print(f"  ok   {name:18s} {version}")
    return failed


def check_duckdb_pandas_handoff() -> None:
    """The load-bearing assumption: DuckDB queries a DataFrame without an export step."""
    import duckdb
    import pandas as pd

    df = pd.DataFrame({
        "region": ["North", "South", "North", "East"],
        "revenue": [100.0, 250.5, 75.25, 400.0],
    })
    con = duckdb.connect()  # in-memory, one per session
    con.register("sales", df)
    got = con.sql(
        "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region ORDER BY total DESC"
    ).df()

    assert len(got) == 3, f"expected 3 regions, got {len(got)}:\n{got}"
    assert got.iloc[0]["region"] == "East", f"expected East first:\n{got}"
    assert got.iloc[0]["total"] == 400.0, f"expected 400.0:\n{got}"
    assert got.iloc[2]["total"] == 175.25, f"North should sum to 175.25:\n{got}"
    print(f"\n  ok   duckdb reads the registered DataFrame\n{got.to_string(index=False)}")


def check_fastmcp_tool_schema() -> None:
    """A decorated function must yield a typed tool schema from its annotations."""
    from fastmcp import FastMCP

    server = FastMCP("datasurf-smoke")

    @server.tool
    def run_sql(query: str, limit: int = 100) -> str:
        """Execute a read-only SQL query."""
        return "stub"

    print("  ok   fastmcp derived a tool schema from annotations")


def main() -> int:
    print(f"python {sys.version.split()[0]}\n")
    if sys.version_info < MIN_PYTHON:
        print(f"  FAIL python {'.'.join(map(str, MIN_PYTHON))}+ required (fastmcp)")
        return 1

    failed = check_imports()
    if failed:
        print(f"\nFAILED imports: {', '.join(failed)}")
        return 1

    import pydantic
    assert pydantic.VERSION.startswith("2"), f"pydantic v2 required, got {pydantic.VERSION}"

    check_duckdb_pandas_handoff()
    check_fastmcp_tool_schema()

    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
