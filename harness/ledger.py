"""The sidebar panel, and the only place cost is formatted.

Separate from trace.py because trace.py must stay importable without Streamlit — M4.8 runs
the golden set headless across three models and should not drag a web server in to do it.
"""

import pandas as pd
import streamlit as st

from harness.trace import Usage

NODE_LABELS = {"write_sql": "Wrote SQL", "run_sql": "Ran the query", "say_answer": "Phrased the answer"}


def usd(amount: float | None) -> str:
    if amount is None:
        return "—"
    return f"${amount:.6f}".rstrip("0").rstrip(".") if amount < 0.01 else f"${amount:.4f}"


def panel(traces: list) -> None:
    """Every question this session, what it cost, and where the tokens went."""
    with st.sidebar:
        st.subheader("Token ledger")
        if not traces:
            st.caption("No questions asked yet. Every model call is counted here — tokens in, "
                       "out and cached, plus OpenRouter's own charge per call.")
            return

        total = sum((t.usage for t in traces), Usage())
        with st.container(horizontal=True):
            st.metric("Questions", len(traces), border=True)
            st.metric("Model calls", sum(t.calls for t in traces), border=True)
        with st.container(horizontal=True):
            st.metric("Tokens in", f"{total.tok_in:,}", border=True,
                      help=f"{total.cached:,} of these were cache reads, billed at a tenth")
            st.metric("Tokens out", f"{total.tok_out:,}", border=True)
        st.metric("Session cost", usd(total.usd), border=True, width="stretch",
                  help="Reported by OpenRouter per call, not estimated from a price table")

        latest = traces[-1]
        with st.expander(f"Last question — {usd(latest.usage.usd)} · {latest.ms:,} ms",
                         icon=":material/receipt_long:"):
            st.dataframe(events_frame(latest), hide_index=True, width="stretch",
                         column_config={"ms": st.column_config.NumberColumn(format="%d ms")})
            st.caption(f"`{latest.path.name}` — one JSON line per node, replayable offline.")


def events_frame(trace) -> pd.DataFrame:
    return pd.DataFrame([{
        "Step": NODE_LABELS.get(e.node, e.node),
        "ms": e.ms,
        "In": e.usage.tok_in if e.usage else 0,
        "Out": e.usage.tok_out if e.usage else 0,
        "Cached": e.usage.cached if e.usage else 0,
        "Cost": usd(e.usage.usd) if e.usage else "—",
    } for e in trace.events])
