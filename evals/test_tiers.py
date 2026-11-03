"""What each tier costs, in model calls. The README's tier table is built from this.

    .venv/bin/pytest evals/test_tiers.py -q

plans/datasurf-plan.md §6 commits to a cost per tier: P spends nothing, 0 and 1 spend one
call, 2 spends two or more. Those are the numbers the whole optimization story rests on, so
they are asserted rather than described — and the assertion is on `trace.calls`, the same
figure the sidebar shows.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import graph, llm
from agent.state import Decision
from harness.trace import Usage
from ingest import profile, store


@pytest.fixture
def sess():
    session = store.Session.new()
    frame = pd.DataFrame({
        "region": ["East", "East", "West", "West"],
        "revenue": [100.0, 50.0, 25.0, 25.0],
        "units": [4, 2, 1, 1],
    })
    session.add_table("sales", frame, profile.build_schema_card(frame, "sales", []), [])
    return session


@pytest.fixture
def stub(monkeypatch):
    replies = []
    monkeypatch.setattr(llm, "chat", lambda s, u, **k: llm.Reply(replies.pop(0),
                                                                 Usage(100, 10, 0, 1e-5)))
    monkeypatch.setattr(llm, "decide",
                        lambda s, u, schema: (replies.pop(0), Usage(100, 10, 0, 1e-5)))
    return replies


def test_tier_p_spends_nothing(sess, stub):
    answer = graph.ask(sess, "how many rows")
    assert answer.tier == "P"
    assert answer.trace.calls == 0
    assert answer.trace.usage.tok_in == 0
    assert answer.text == "`sales` has 4 rows."
    assert stub == []                                  # the stub was never consulted


def test_tier_0_spends_one_call_and_skips_verify(sess, stub):
    stub.append(Decision(reason="the card gives the range", tier=0,
                         answer="Revenue runs 25.00 to 100.00, per the card."))
    answer = graph.ask(sess, "what is the range of revenue")
    assert answer.tier == "0" and answer.trace.calls == 1
    assert "verify" not in [e.node for e in answer.trace.events]
    assert answer.verified == "read from the Schema Card"


def test_tier_1_spends_one_call_and_cannot_be_wrong(sess, stub):
    stub.append(Decision(reason="one column, one stat", tier=1,
                         table="sales", column="revenue", stat="mean"))
    answer = graph.ask(sess, "what is the average revenue")
    assert answer.tier == "1" and answer.trace.calls == 1
    assert answer.text == "`revenue` mean is 50."         # our code computed it
    assert "verify" not in [e.node for e in answer.trace.events]


def test_tier_2_spends_two_calls_and_is_verified(sess, stub):
    stub += [Decision(reason="group-by", tier=2,
                      sql="SELECT region, sum(revenue) AS total FROM sales GROUP BY 1"),
             "East 150, West 50."]
    answer = graph.ask(sess, "revenue by region")
    assert answer.tier == "2" and answer.trace.calls == 2
    assert "verify" in [e.node for e in answer.trace.events]
    assert "row count re-ran at 2" in answer.verified


def test_a_stat_we_cannot_precompute_falls_through_to_tier_2(sess, stub):
    """`column_stat` returning "not precomputed, use run_sql" is an explicit outcome. The
    alternative — guessing — is the one failure mode worth engineering against (§13)."""
    stub += [Decision(reason="distribution shape", tier=1,
                      table="sales", column="revenue", stat="kurtosis"),
             "SELECT kurtosis(revenue) AS k FROM sales", "Kurtosis is -1.2."]
    answer = graph.ask(sess, "what is the kurtosis of revenue")
    assert answer.tier == "2"
    assert "kurtosis' is not precomputed" in answer.trace.events[2].out["fell_through"]
    assert answer.text == "Kurtosis is -1.2."


def test_a_preset_finding_becomes_a_tier_1_answer(sess, stub):
    """"Tier 1 gets richer as the session goes on" (§6), asserted rather than asserted of."""
    sess.stats[("sales", "revenue", "outliers")] = (2, "the data health preset, Tukey rule")
    stub.append(Decision(reason="the health preset computed this", tier=1,
                         table="sales", column="revenue", stat="outliers"))
    answer = graph.ask(sess, "how many revenue outliers are there")
    assert answer.trace.calls == 1
    assert answer.text == "`revenue` outliers is 2."
    assert answer.verified == "the data health preset, Tukey rule"


def test_verify_catches_a_join_that_duplicated_rows(sess, stub):
    """The one silent SQL bug worth a second query: a fan-out inflates a total plausibly."""
    sess.con.register("tags", pd.DataFrame({"region": ["East", "East", "West"],
                                            "tag": ["a", "b", "c"]}))
    stub += [Decision(reason="needs the tag table", tier=2,
                      sql="SELECT tag, sum(revenue) AS total FROM sales "
                          "JOIN tags ON sales.region = tags.region GROUP BY 1"),
             "SELECT region, sum(revenue) AS total FROM sales GROUP BY 1",
             "East 150, West 50."]
    answer = graph.ask(sess, "revenue by tag")
    verify = [e for e in answer.trace.events if e.node == "verify"][0]
    assert verify.out["pass"] is False
    assert "duplicated rows" in verify.out["why"]
    assert [e.node for e in answer.trace.events].count("repair") == 1


def test_verify_catches_an_empty_result(sess, stub):
    stub += [Decision(reason="filter", tier=2,
                      sql="SELECT * FROM sales WHERE region = 'east'"),
             "SELECT * FROM sales WHERE region = 'East'", "Two rows in East."]
    answer = graph.ask(sess, "show me the east rows")
    verify = [e for e in answer.trace.events if e.node == "verify"][0]
    assert verify.out["why"] == "no rows"
    assert answer.text == "Two rows in East."
