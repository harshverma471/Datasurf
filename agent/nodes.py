"""The nodes. Each one is either a single model call with a single job, or no model at all.

Three of the six spend nothing: `pre_route` reads the session inventory and the Schema Cards,
`act` dispatches and runs SQL, `verify` re-queries DuckDB. That ratio is the design
(plans/datasurf-plan.md §7), and the trace proves it per question rather than claiming it.

Dependencies that cannot be checkpointed — the DuckDB connection, the DataFrames, the Trace —
arrive in `config["configurable"]`, never in State. That is what keeps Seam 1 a one-line
change to `SqliteSaver` (§4a).

No `clarify` node. The diagram has one, M2.2's deliverable list does not, and an interrupt
that survives a Streamlit rerun is real work for a case tier 0 already covers: the planner can
answer "two columns could mean that — which did you mean?" as a tier-0 answer for one call.
"""

import re

from agent import context, lanes, preroute
from agent.state import Decision, State
from tools import tabular

MAX_RETRIES = 2
IMPOSSIBLE = "IMPOSSIBLE"

# An unfiltered aggregate: its group totals are checkable against the source table's total.
# A WHERE legitimately changes the total, so a filtered query is unknowable and verify says
# so. A JOIN is checkable one-sidedly — see _reconcile.
# sum( [qualifier.]column ) [AS alias]
_SUM_OF = re.compile(r"\bsum\s*\(\s*(?:([a-z_]\w*)\.)?([a-z_]\w*)\s*\)"
                     r"(?:\s+as\s+([a-z_]\w*))?", re.I)
# FROM table [AS alias] — the alias group must not swallow the next keyword
_KEYWORD = r"(?!left\b|right\b|inner\b|outer\b|full\b|cross\b|join\b|where\b|group\b|" \
           r"order\b|having\b|limit\b|using\b|on\b|qualify\b)"
_FROM = re.compile(rf"\bfrom\s+([a-z_]\w*)(?:\s+(?:as\s+)?{_KEYWORD}([a-z_]\w*))?", re.I)
_FILTERED = re.compile(r"\b(where|having|qualify)\b", re.I)
_JOINED = re.compile(r"\bjoin\b", re.I)


def deps(config):
    configurable = config["configurable"]
    return configurable["session"], configurable["trace"]


# --- free ------------------------------------------------------------------------

def pre_route(state: State, config) -> dict:
    """Tier P and the lane, both without a model (§6, §7)."""
    sess, trace = deps(config)
    with trace.span("pre_route", question=state.question) as event:
        lane = lanes.lane_for(sess.tables, sess.documents)
        update = {"lane": lane}

        if lane != "tabular":
            update["answer"] = lanes.NOT_YET
        elif hit := preroute.answer(state.question, sess.tables):
            answer, source = hit
            update |= {"tier": "P", "answer": answer, "verified": f"read from {source}"}

        event.out = {k: v for k, v in update.items() if k != "answer"} | {
            "tier_p_hit": update.get("tier") == "P"}
        return update


def act(state: State, config) -> dict:
    """Run what plan decided. No model call: tier 1 is a lookup, tier 2 is DuckDB."""
    sess, trace = deps(config)
    with trace.span("act", tier=state.tier, sql=state.sql or None) as event:
        if state.tier == "0":
            event.out = {"note": "nothing to do — plan answered from the card"}
            return {}

        if state.tier == "1":
            try:
                value, source = tabular.column_stat(
                    sess.tables, state.table, state.column, state.stat, sess.stats)
            except ValueError as exc:
                # "not precomputed, use run_sql" is an explicit outcome, not a failure to
                # hide (§13 risks). Promote to tier 2 and let repair write the query.
                event.out = {"fell_through": str(exc)}
                return {"tier": "2", "error": str(exc)}
            event.out = {"value": value, "source": source}
            return {"answer": f"`{state.column}` {state.stat} is {value}.",
                    "verified": source, "error": ""}

        try:
            result = tabular.run_sql(sess.con, state.sql)
        except Exception as exc:
            event.out = {"failed": f"{type(exc).__name__}: {exc}"}
            return {"error": f"{type(exc).__name__}: {exc}", "handle": "", "preview": ""}

        handle = sess.put_artifact(result)
        preview = context.preview(handle, result)
        event.out = {"handle": handle, "rows": len(result.rows),
                     "cols": list(result.rows.columns), "truncated": result.truncated,
                     "preview_tokens": len(preview) // 4}
        return {"handle": handle, "preview": preview, "error": ""}


def verify(state: State, config) -> dict:
    """Tier 2 only. The node that makes a generated number defensible, for zero tokens.

    The reconciliation rule is the one worth having: a model that joins two tables and fans
    the rows out produces a plausible, wrong, larger total — and no amount of reading the SQL
    catches that reliably, while one second aggregate does.
    """
    sess, trace = deps(config)
    with trace.span("verify", sql=state.sql, rules=lanes.VERIFY_RULES[state.lane]) as event:
        if state.error:
            event.out = {"pass": False, "why": state.error}
            return {}

        result = sess.artifacts[state.handle]
        checks = []

        counted = sess.con.sql(f"SELECT count(*) AS n FROM ({state.sql})").fetchone()[0]
        if not result.truncated and counted != len(result.rows):
            event.out = {"pass": False, "why": f"re-ran and got {counted} rows, stored "
                                               f"{len(result.rows)}"}
            return {"error": f"the query is not deterministic: a re-run returned {counted} "
                             f"rows against {len(result.rows)} the first time"}
        checks.append(f"row count re-ran at {counted:,}")

        if result.rows.empty:
            event.out = {"pass": False, "why": "no rows"}
            return {"error": "the query returned no rows — a filter value may not match any "
                             "value actually in the column"}

        if reconciled := _reconcile(sess.con, state.sql, result):
            ok, note = reconciled
            checks.append(note)
            if not ok:
                event.out = {"pass": False, "why": note}
                return {"error": f"the figures do not reconcile: {note}. A join may be "
                                 f"duplicating rows."}

        if result.truncated:
            checks.append(f"⚠ cut off at the row cap, so any total over it is partial")

        event.out = {"pass": True, "checks": checks}
        return {"verified": " · ".join(checks), "error": ""}


def _reconcile(con, sql: str, result) -> tuple[bool, str] | None:
    """Do the result's group totals add up to the source table's total? None when unknowable.

    Two regimes, because they have different correct answers:

    **No join.** The totals must match exactly. What this catches is a query that silently
    looked at a subset — an inner `LIMIT`, a `sum(DISTINCT …)`, a subquery that dropped rows.

    **A join.** Inner joins legitimately drop rows, so a smaller total proves nothing. But no
    join can legitimately *increase* the sum of a left-table column, so a total that exceeds
    the source's is proof of row duplication — the fan-out bug, which is the one silent SQL
    error worth paying a second query to catch, and the one no amount of reading the SQL finds
    reliably.

    Which table owns the summed column is resolved, not guessed: a qualified `sum(s.revenue)`
    is only checked when `s` is the first table after FROM or its alias, because summing the
    *right* table's column against the *left* table's total would invent the expectation. And
    whether the column lives there at all is not parsed, it is asked — if it does not, DuckDB
    raises and this returns None. Self-validating beats a SQL parser.
    """
    if _FILTERED.search(sql):
        return None                        # a filter changes the total, legitimately
    summed, source = _SUM_OF.search(sql), _FROM.search(sql)
    if not (summed and source):
        return None
    qualifier, column, alias = summed.groups()
    table, table_alias = source.group(1), source.group(2)
    # `sum(r.revenue)` where `r` is the joined table: the left table's total is not the
    # expectation, and there is no cheap honest one.
    if qualifier and qualifier.lower() not in {table.lower(),
                                               (table_alias or "").lower()}:
        return None
    # Prefer the alias. `SELECT rep_id, sum(revenue) AS total` has two numeric columns and the
    # group key is one of them, so "the only numeric column" picks nothing — which is why the
    # planner is told to name every computed column.
    if alias and alias in result.rows.columns:
        totals = [alias]
    else:
        totals = [c for c in result.rows.columns
                  if result.rows[c].dtype.kind in "fiu" and c != column]
    if len(totals) != 1:
        return None
    try:
        expected = con.sql(f"SELECT sum({column}) AS t FROM {table}").fetchone()[0]
    except Exception:
        return None
    if expected is None:
        return None
    got, want = float(result.rows[totals[0]].sum()), float(expected)
    tolerance = max(abs(want) * 1e-9, 1e-6)

    if _JOINED.search(sql):
        if got <= want + tolerance:
            return True, f"join did not inflate {table}.{column} ({got:,.2f} ≤ {want:,.2f})"
        return False, (f"the join duplicated rows: totals come to {got:,.2f} against "
                       f"{want:,.2f} in {table}.{column} alone")

    if abs(got - want) <= tolerance:
        return True, f"group totals sum to {got:,.2f}, matching {table}.{column}"
    return False, (f"group totals sum to {got:,.2f} but {table}.{column} totals {want:,.2f}")


# --- one model call each ---------------------------------------------------------

def plan(state: State, config) -> dict:
    """The only node that chooses. One structured reply picks the tier and, at tier 0,
    answers outright — which is what makes a tier-0 question cost one call, not two."""
    from agent import llm                       # deferred: keeps `import nodes` cheap in evals

    sess, trace = deps(config)
    with trace.span("plan", question=state.question, lane=state.lane) as event:
        decision, usage = llm.decide(
            context.plan_prompt(sess.tables, state.lane), state.question, Decision)
        event.usage = usage
        event.out = decision.model_dump(exclude_defaults=True)
        return {"tier": str(decision.tier), "reason": decision.reason,
                "sql": llm.unfence(decision.sql), "answer": decision.answer,
                "table": decision.table, "column": decision.column, "stat": decision.stat,
                "verified": "read from the Schema Card" if decision.tier == 0 else ""}


def repair(state: State, config) -> dict:
    """One corrected query from the error text plus the cards, capped at two attempts."""
    from agent import llm

    sess, trace = deps(config)
    with trace.span("repair", attempt=state.retries + 1, error=state.error) as event:
        reply = llm.chat(
            f"{context.REPAIR_RULES}\n\n{context.cards(sess.tables)}",
            f"Question: {state.question}\n\nThe query that failed:\n{state.sql}\n\n"
            f"The error:\n{state.error}")
        event.usage = reply.usage
        sql = llm.unfence(reply.text)
        event.out = {"sql": sql, "giving_up": sql.upper().startswith(IMPOSSIBLE)}
        if sql.upper().startswith(IMPOSSIBLE):
            return {"retries": MAX_RETRIES,
                    "error": "that question cannot be answered from the columns in these "
                             "tables"}
        return {"sql": sql, "retries": state.retries + 1}


def synthesize(state: State, config) -> dict:
    """Free for tiers P, 0 and 1 — their answer is already exact, and rephrasing an exact
    number is a chance to get it wrong. One call for tier 2."""
    from agent import llm

    sess, trace = deps(config)
    if state.answer and not state.error:
        return {}

    with trace.span("synthesize", tier=state.tier, had_error=bool(state.error)) as event:
        if state.error:
            event.out = {"note": "answered from the failure, no model call"}
            return {"answer": f"I could not answer that — {state.error}."}

        reply = llm.chat(
            context.ANSWER_RULES,
            f"Question: {state.question}\n\nQuery:\n{state.sql}\n\nResult:\n{state.preview}")
        event.usage = reply.usage
        event.out = {"text": reply.text}
        return {"answer": reply.text}
