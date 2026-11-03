"""The wiring. Four conditional edges and exactly one cycle.

    START → pre_route ─lane─→ plan → act ─tier─→ verify ─pass?─→ synthesize → END
                  └─tier P──────────────────────────────────────────↑   │
                                                     repair ←───────────┘
                                                        └→ act, unless it gave up

Reading it against plans/datasurf-plan.md §7:

- **`route` is not here.** The lane edge replaced it, and reads the session inventory for
  nothing. One model call saved on every question (§7).
- **`verify` is conditional on tier.** Tiers P, 0 and 1 bypass it — there is nothing to check
  when your own code produced the number (§6).
- **`repair` is the only cycle in the system**, capped at two attempts.

`MemorySaver` is Seam 1 (§4a): because State holds no connection, no DataFrame and no Trace,
swapping it for `SqliteSaver` is the one-line change it is advertised to be.
"""

from dataclasses import dataclass, field

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import lanes, nodes
from agent.state import State
from harness.trace import Trace


def after_act(state: State) -> str:
    """Tiers P/0/1 are exact by construction. Only a generated query earns a check."""
    return "verify" if state.tier == "2" else "synthesize"


def after_verify(state: State) -> str:
    return "repair" if state.error and state.retries < nodes.MAX_RETRIES else "synthesize"


def after_repair(state: State) -> str:
    """Only back to `act` if repair actually produced a new query.

    Without this edge, a repair that answers IMPOSSIBLE still returns to `act`, which re-runs
    the same broken SQL, fails the same way, and overwrites repair's honest "that cannot be
    answered from these columns" with the raw binder error. A wasted query and a worse message.
    """
    return "synthesize" if state.retries >= nodes.MAX_RETRIES else "act"


def build():
    graph = StateGraph(State)
    for name in ("pre_route", "plan", "act", "verify", "repair", "synthesize"):
        graph.add_node(name, getattr(nodes, name))

    graph.add_edge(START, "pre_route")
    graph.add_conditional_edges("pre_route", lanes.edge, ["plan", "synthesize"])
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", after_act, ["verify", "synthesize"])
    graph.add_conditional_edges("verify", after_verify, ["repair", "synthesize"])
    graph.add_conditional_edges("repair", after_repair, ["act", "synthesize"])
    graph.add_edge("synthesize", END)
    return graph.compile(checkpointer=MemorySaver())


GRAPH = build()


@dataclass
class Answer:
    """What the UI renders. A flat view of the final State plus the rows behind its handle.

    State carries a handle so a checkpoint stays small; the screen needs the actual frame, so
    it is fetched once, here, at the edge.
    """
    question: str
    trace: Trace
    tier: str = ""
    reason: str = ""
    sql: str = ""
    text: str = ""
    verified: str = ""
    rows: pd.DataFrame | None = None
    truncated: bool = False
    error: str | None = None
    preset: str = ""                        # set when a preset produced this, not a question
    report: object = None                   # analysis.Report, for a preset turn
    frames: dict = field(default_factory=dict)


def ask(sess, question: str) -> Answer:
    """One question through the graph. Every failure lands on the Answer, never at the UI."""
    trace = Trace.new(question)
    config = {"configurable": {"session": sess, "trace": trace, "thread_id": trace.run_id}}

    try:
        final = State(**GRAPH.invoke(State(question=question), config))
    except Exception as exc:
        return Answer(question, trace, error=_message(exc))

    result = sess.artifacts.get(final.handle)
    return Answer(
        question, trace, tier=final.tier or "", reason=final.reason, sql=final.sql,
        text=final.answer, verified=final.verified,
        rows=result.rows if result is not None else None,
        truncated=result.truncated if result is not None else False,
        error=final.error or None)


def _message(exc: Exception) -> str:
    from agent.llm import NoKey

    if isinstance(exc, NoKey):
        return str(exc)
    return f"the run failed — {type(exc).__name__}: {exc}"
