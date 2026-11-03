"""What breaks if the first agent call or the ledger regresses.

    .venv/bin/pytest evals/test_agent.py -q

No network. The model is stubbed, because what is worth protecting here is not the model's
SQL — it is the three things around it:

  the guard    model-authored SQL is the one trust boundary in this project
  the ledger   tokens and cost must be recorded from the first question, or there is no
               baseline to measure M4.7's optimizations against (plans/modules.md §6)
  the failure  a bad query must land on the Answer, not as a traceback in the UI
  the tiers    what each tier costs, asserted in model calls (evals/test_tiers.py covers
               the routing itself)
"""

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agent import graph, llm
from agent.state import Decision
from harness.trace import Trace, Usage
from ingest import store
from tools import tabular


@pytest.fixture
def sess():
    """A session with one real, queryable table and a card the planner can read."""
    session = store.Session.new()
    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "revenue": [100.0, 50.0, 25.0, 25.0],
    })
    card = type("Card", (), {"to_text": lambda self: "table sales   4 rows × 2 cols"})()
    session.add_table("sales", frame, card, [])
    return session


@pytest.fixture
def con(sess):
    return sess.con


@pytest.fixture
def stub(monkeypatch):
    """Script the model. `replies` takes Decision objects for plan, strings for chat."""
    calls, replies = [], []

    def fake_chat(system, user, **kwargs):
        calls.append(("chat", system, user))
        return llm.Reply(replies.pop(0), Usage(100, 10, 20, 0.000015))

    def fake_decide(system, user, schema):
        calls.append(("decide", system, user))
        return replies.pop(0), Usage(100, 10, 20, 0.000015)

    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(llm, "decide", fake_decide)
    return type("Stub", (), {"calls": calls, "replies": replies})()


def sql_plan(sql: str) -> Decision:
    return Decision(reason="needs a query", tier=2, sql=sql)


# --- the trust boundary ----------------------------------------------------------

def test_select_runs(con):
    result = tabular.run_sql(con, "SELECT region, sum(revenue) AS total FROM sales GROUP BY 1")
    assert set(result.rows["total"]) == {150.0, 50.0}


def test_with_runs(con):
    result = tabular.run_sql(con, "WITH x AS (SELECT * FROM sales) SELECT count(*) AS n FROM x")
    assert result.rows["n"][0] == 4


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "CREATE TABLE t AS SELECT 1",
    "COPY sales TO '/tmp/leak.csv'",
    "ATTACH 'other.db'",
    "INSTALL httpfs",
    "SET threads=1",
    "SELECT 1; DROP TABLE sales",
    "",
])
def test_refuses_anything_but_one_select(con, sql):
    with pytest.raises(ValueError):
        tabular.run_sql(con, sql)


def test_a_select_cannot_reach_the_filesystem(con):
    """The statement-type check passes this one — enable_external_access is what stops it."""
    with pytest.raises(duckdb.Error):
        tabular.run_sql(con, "SELECT * FROM read_csv('/etc/passwd')")


def test_the_model_sees_full_precision(con):
    """tabulate's default `g` format renders 3,383,134.57 as 3.38313e+06. A model cannot
    report a figure it was never shown."""
    con.register("money", pd.DataFrame({"total": [3383134.57], "pct": [0.0014493]}))
    text = tabular.run_sql(con, "SELECT * FROM money").to_markdown()
    assert "3383134.57" in text and "0.0014493" in text and "e+" not in text


def test_results_are_capped(con):
    con.register("big", pd.DataFrame({"n": range(config.MAX_RESULT_ROWS + 50)}))
    result = tabular.run_sql(con, "SELECT * FROM big")
    assert len(result.rows) == config.MAX_RESULT_ROWS and result.truncated
    assert "cut off" in result.to_markdown()


# --- the ledger ------------------------------------------------------------------

def test_usage_sums_and_keeps_unknown_cost_unknown():
    assert (Usage(10, 1, 2, 0.5) + Usage(20, 2, 0, 0.25)).usd == 0.75
    assert (Usage(10, 1) + Usage(20, 2)).usd is None          # a free endpoint reported nothing
    assert (Usage(10, 1, 0, 0.5) + Usage(20, 2)).usd == 0.5   # one of two did
    assert Usage(1000, 0, 400).billed_in == 600


OPENROUTER_RESPONSE = {
    "id": "gen-1", "object": "chat.completion", "created": 1, "model": "openai/gpt-6-luna",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "SELECT 1"}}],
}


def test_openrouters_cost_survives_the_sdk():
    """`cost` is not in the OpenAI schema — it rides along as an extra field, and the whole
    ledger is built on it reaching us. A free endpoint sends none, which must read as unknown
    rather than as free."""
    from openai.types.chat import ChatCompletion

    paid = dict(OPENROUTER_RESPONSE, usage={
        "prompt_tokens": 4210, "completion_tokens": 37, "total_tokens": 4247,
        "prompt_tokens_details": {"cached_tokens": 3900}, "cost": 0.00005035})
    assert llm.usage(ChatCompletion.model_validate(paid)) == Usage(4210, 37, 3900, 0.00005035)

    free = dict(OPENROUTER_RESPONSE, usage={
        "prompt_tokens": 900, "completion_tokens": 20, "total_tokens": 920})
    assert llm.usage(ChatCompletion.model_validate(free)).usd is None


def test_every_node_lands_in_the_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    trace = Trace.new("how many rows?")
    with trace.span("write_sql", question="how many rows?") as event:
        event.usage, event.out = Usage(700, 12, 0, 0.00007), {"sql": "SELECT count(*) FROM sales"}
    with trace.span("run_sql", sql="SELECT count(*) FROM sales") as event:
        event.out = {"rows": 1}

    lines = [json.loads(l) for l in trace.path.read_text().splitlines()]
    assert [l["node"] for l in lines] == ["write_sql", "run_sql"]
    assert [l["seq"] for l in lines] == [1, 2]
    assert lines[1]["usage"] is None                 # a deterministic node spends nothing
    assert trace.calls == 1 and trace.usage.tok_in == 700


def test_a_failing_node_is_still_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    trace = Trace.new("q")
    with pytest.raises(ZeroDivisionError):
        with trace.span("run_sql", sql="SELECT 1/0"):
            1 / 0
    assert "ZeroDivisionError" in trace.events[0].error
    assert json.loads(trace.path.read_text())["error"].startswith("ZeroDivisionError")


def test_a_trace_records_shapes_not_rows(tmp_path, monkeypatch, sess, stub):
    """Where the privacy line falls, asserted rather than promised.

    The result set never reaches disk — only its shape and column names. The answer sentence
    does, deliberately: M4.5's judge has to score something, and the ledger has to show what
    was said. So a figure can appear in a trace, but only one the model already published to
    the user, never a cell it merely read.
    """
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    stub.replies += [sql_plan("SELECT region, sum(revenue) AS total FROM sales GROUP BY 1"),
                     "East made 150."]
    answer = graph.ask(sess, "revenue by region")

    events = {json.loads(line)["node"]: json.loads(line)
              for line in answer.trace.path.read_text().splitlines()}
    assert events["act"]["out"]["cols"] == ["region", "total"]
    assert "East" not in json.dumps(events["act"])
    assert events["synthesize"]["out"]["text"] == "East made 150."


# --- the graph -------------------------------------------------------------------

def test_question_to_answer(sess, stub):
    stub.replies += [sql_plan("SELECT region, sum(revenue) AS total FROM sales "
                              "GROUP BY 1 ORDER BY 1"),
                     "East made $150 and West made $50."]
    answer = graph.ask(sess, "revenue by region")

    assert answer.error is None
    assert answer.text == "East made $150 and West made $50."
    assert list(answer.rows["total"]) == [150.0, 50.0]
    assert [e.node for e in answer.trace.events] == [
        "pre_route", "plan", "act", "verify", "synthesize"]
    assert answer.trace.calls == 2            # pre_route, act and verify spend nothing
    assert answer.tier == "2"


def test_the_schema_card_is_what_the_model_plans_against(sess, stub):
    stub.replies += [sql_plan("SELECT 1 AS n"), "One."]
    graph.ask(sess, "how much revenue in the east")
    kind, system, user = stub.calls[0]
    assert kind == "decide"
    assert "table sales   4 rows × 2 cols" in system
    assert user == "how much revenue in the east"       # rows are not in the prompt


def test_a_fenced_reply_still_runs(sess, stub):
    stub.replies += [sql_plan("```sql\nSELECT count(*) AS n FROM sales\n```"), "Four."]
    answer = graph.ask(sess, "how many rows have revenue over 10")
    assert answer.error is None and answer.rows["n"][0] == 4


def test_the_result_reaches_the_prompt_as_a_handle_not_rows(sess, stub):
    stub.replies += [sql_plan("SELECT region, sum(revenue) AS total FROM sales GROUP BY 1"),
                     "East leads."]
    graph.ask(sess, "revenue by region")
    _, _, user = stub.calls[-1]
    assert "r1: 2 rows × 2 cols" in user                 # the handle and the shape
    assert sess.artifacts["r1"].rows.shape == (2, 2)     # the rows stayed in the store


def test_a_broken_query_is_repaired(sess, stub):
    stub.replies += [sql_plan("SELECT nope FROM sales"),
                     "SELECT count(*) AS n FROM sales", "Four rows."]
    answer = graph.ask(sess, "what is nope")
    assert answer.error is None and answer.text == "Four rows."
    assert [e.node for e in answer.trace.events] == [
        "pre_route", "plan", "act", "verify", "repair", "act", "verify", "synthesize"]


def test_repair_gives_up_rather_than_looping(sess, stub):
    stub.replies += [sql_plan("SELECT diagnosis FROM sales"), "IMPOSSIBLE"]
    answer = graph.ask(sess, "what is the patient diagnosis")
    assert "cannot be answered from the columns" in answer.error
    assert [e.node for e in answer.trace.events].count("act") == 1    # no second attempt
    assert answer.trace.calls == 2


def test_repair_stops_after_two_attempts(sess, stub):
    stub.replies += [sql_plan("SELECT a FROM sales"), "SELECT b FROM sales",
                     "SELECT c FROM sales"]
    answer = graph.ask(sess, "something unanswerable")
    assert answer.error is not None
    assert [e.node for e in answer.trace.events].count("repair") == 2


def test_no_key_is_an_answer_not_a_traceback(sess, monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    llm.client.cache_clear()
    answer = graph.ask(sess, "revenue by region")
    assert "OPENROUTER_API_KEY" in answer.error
    llm.client.cache_clear()
