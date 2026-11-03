"""What the graph carries, and what the model is allowed to return.

Two rules shape this file.

**State is serializable.** Seam 1 from plans/datasurf-plan.md §4a is that swapping `MemorySaver`
for `SqliteSaver` is a one-line change — true only if every field the agent needs to resume
lives here. So the DuckDB connection, the DataFrames and the Trace do *not* live here: they
are passed per run in `config["configurable"]`, and State holds a **handle** to a result
rather than the result. That is the handle pattern's real job (§9): a result costs 4 lines of
context no matter how many rows it has.

**The model's reply is a type, not a string.** `Decision` is the plan node's structured
output. One call picks the tier and, for tier 0, answers outright — which is what makes a
tier-0 question cost one call instead of two.
"""

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["P", "0", "1", "2"]
Lane = Literal["tabular", "document", "hybrid"]

# Everything column_stat can serve without writing a query. Named here because the plan node
# is shown this list and may only choose from it — a stat we cannot compute is a wasted call.
STATS = ["mean", "median", "min", "max", "std", "sum", "count", "nulls", "distinct", "mode"]


class Decision(BaseModel):
    """The plan node's one structured reply. Exactly one tier's fields are filled.

    Field order is deliberate: `reason` first makes the model commit to why before it commits
    to a tier, which costs ~15 tokens and stops `tier` being a coin flip on an ambiguous
    question.
    """
    reason: str = Field(description="One short clause: why this tier, not a cheaper one.")
    tier: Literal[0, 1, 2] = Field(description=(
        "0 if the Schema Cards already contain the answer. "
        "1 if one precomputed stat of one column answers it. "
        "2 if it needs a query: any filter, group-by, join, ratio or ranking."))
    answer: str = Field(default="", description="Tier 0 only: the answer, citing the card.")
    table: str = Field(default="", description="Tier 1 only: the table name.")
    column: str = Field(default="", description="Tier 1 only: the column name.")
    stat: str = Field(default="", description=f"Tier 1 only: one of {STATS}.")
    sql: str = Field(default="", description="Tier 2 only: one DuckDB SELECT. No fence.")


class State(BaseModel):
    """One question's journey. Everything here survives a checkpoint."""
    question: str
    lane: Lane = "tabular"
    tier: Tier | None = None
    reason: str = ""

    table: str = ""           # tier 1: the one column_stat lookup plan asked for
    column: str = ""
    stat: str = ""

    sql: str = ""
    handle: str = ""          # key into Session.artifacts — never the rows themselves
    preview: str = ""         # the ≤400-token rendering of that result, for the prompt

    answer: str = ""
    error: str = ""
    retries: int = 0
    verified: str = ""        # what verify actually checked, so the UI can show the check
