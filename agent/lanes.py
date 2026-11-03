"""The first conditional edge, and the one that cost a model call in the first draft.

`route` used to be a node: one model call per question classifying it as tabular, document or
hybrid. But `ingest/router.py` sorted every upload by extension at ingest time, so the session
inventory already held the answer — the node was paying for a fact we had
(plans/datasurf-plan.md §7). Reading a dict is free; asking a model is not.

The per-lane tables below are data, not dispatch. Shared nodes read them; they do not fork the
graph. M3 fills in the two rows that currently say "not yet" and adds nothing else here.
"""

from agent.state import Lane

TOOLS: dict[Lane, tuple[str, ...]] = {
    "tabular": ("run_sql", "column_stat"),
    "document": ("search_docs", "get_page", "quote"),          # M3.6
    "hybrid": ("run_sql", "column_stat", "search_docs", "get_page", "quote"),
}

VERIFY_RULES: dict[Lane, tuple[str, ...]] = {
    "tabular": ("row count re-runs", "figures reconcile against a second aggregate"),
    "document": ("cited page exists", "the quote appears on it"),               # M3.6
    "hybrid": ("both of the above", "do the two sources agree"),                # M3.7
}

NOT_YET = ("Questions about documents arrive with M3. This session can answer questions about "
           "the spreadsheets listed in the sidebar.")


def lane_for(tables: dict, documents: dict) -> Lane:
    """Free whenever the session holds one kind of thing, which is the common case."""
    if tables and documents:
        return "hybrid"
    return "document" if documents else "tabular"


def edge(state) -> str:
    """The graph's first conditional edge, and the cheapest one in the system.

    `pre_route` already set `state.lane` from the inventory and tried tier P, so this reads an
    attribute. All three lanes share `plan` — the lane parameterizes the shared nodes through
    TOOLS and VERIFY_RULES above rather than forking the graph, which is what keeps one repair
    loop, one budget guard and one state schema (plans/datasurf-plan.md §7 Q-B).
    """
    return "synthesize" if state.answer else "plan"
