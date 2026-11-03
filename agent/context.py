"""Prompt assembly. Everything that decides how many tokens a question costs.

Two jobs, both from plans/datasurf-plan.md §9.

**Order for the cache.** An OpenAI-compatible endpoint caches on the prompt *prefix*, so the
only thing that makes caching work is putting the stable text first and the volatile text
last: rules, then Schema Cards, then — in the user turn, never the system prompt — the
question. Get the order wrong and every question is a cache miss at ten times the price.
There is no API call for this. It is just discipline, which is why it lives in one file.

**The handle pattern.** A tool result goes to `Session.artifacts` and the prompt gets a
reference, a shape and three rows. A 200-row result and a 3-row result cost the same context.
"""

from agent import lanes
from agent.state import STATS

HEAD_ROWS = 3
MAX_PREVIEW_CHARS = 1600      # ≈400 tokens, the cap §9 puts on a tool result

PLAN_RULES = f"""You route one question about tables already registered in DuckDB to the \
cheapest tier that can answer it correctly.

tier 0 — the Schema Cards below already contain the answer. Row counts, column names, types,
         ranges, means, null shares, the notes block. Answer from the card and say which
         figure you read. Never estimate from a range.
tier 1 — one precomputed statistic of one column answers it. Give table, column and one of
         {STATS}. Nothing else: no filter, no grouping, no second column.
tier 2 — anything else. Write one DuckDB SELECT. Any filter, group-by, join, ratio, ranking,
         date arithmetic or multi-column comparison is tier 2.

Choosing a tier that is too cheap produces a wrong answer; too expensive only wastes a
fraction of a cent. When the two are close, go up.

Writing SQL for tier 2:
- SELECT or WITH only, one statement, no trailing semicolon, no markdown fence.
- Only the tables and columns in the cards. `[was "Revenue ($)"]` is a column's original
  spreadsheet name — the user may ask using it, but query the canonical name.
- The notes block holds facts computed from the real rows. Trust it over a guess, especially
  the candidate join keys and the clustered-missingness warnings.
- Name every computed column. Aggregate unless the question asks for individual records.
- Dates are real DATE/TIMESTAMP columns; Excel's serial integers are already converted."""

ANSWER_RULES = """You state the answer to a question from a query result that has already run.

- One or two sentences. No preamble, no restating the question, no describing the SQL.
- Use the numbers in the result exactly as given. Never round, never estimate, and never
  introduce a figure that is not in the result.
- If the result is empty, or was cut off, or disagrees with the question's premise, say so."""

REPAIR_RULES = """Your previous DuckDB query failed. Return one corrected SELECT and nothing
else — no explanation, no fence.

Read the error literally. A missing column means you guessed a name: use one from the cards.
A type error means you aggregated something that is not numeric. If the question genuinely
cannot be answered from these columns, return the single word IMPOSSIBLE."""


def cards(tables: dict) -> str:
    """Every registered table's card, not just one.

    Cross-table questions and the join candidates in the notes block are only answerable if
    the model can see both sides, and all cards together are still a rounding error next to
    one page of rows.
    """
    return "\n\n".join(t.card.to_text() for t in tables.values())


def plan_prompt(tables: dict, lane: str) -> str:
    """The stable prefix: rules, then tools for this lane, then cards. No question."""
    tools = ", ".join(lanes.TOOLS[lane])
    return f"{PLAN_RULES}\n\nTools available in the {lane} lane: {tools}.\n\n{cards(tables)}"


def preview(handle: str, result) -> str:
    """handle + shape + as many rows as the cap allows. The rows stay in the artifact store.

    §9 writes this as `handle + shape + head(3) + summary, hard-capped at 400 tokens`, and the
    cap is the real rule — `head(3)` is what a 200-row result collapses to, not a limit to
    impose on a 4-row aggregate. Fixing the row count at three would hide North from
    "revenue by region" and the answer would be confidently incomplete, which is worse than
    expensive. So: fit what fits.
    """
    rows, cols = result.rows.shape
    header = (f"{handle}: {rows} rows × {cols} cols "
              f"[{', '.join(map(str, result.rows.columns))}]")

    body = result.rows.to_markdown(index=False, floatfmt=".10g")
    per_row = max(len(body) // max(rows, 1), 1)
    shown = min(rows, max(HEAD_ROWS, (MAX_PREVIEW_CHARS - len(header)) // per_row))
    if shown < rows:
        body = result.rows.head(shown).to_markdown(index=False, floatfmt=".10g")
        body += f"\n({rows - shown} more rows, in {handle} but not shown)"
    if result.truncated:
        body += "\n(the query itself was cut off at the row cap)"
    return f"{header}\n{body}"[:MAX_PREVIEW_CHARS]
