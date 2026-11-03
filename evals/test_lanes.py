"""Lane selection must stay token-free.

    .venv/bin/pytest evals/test_lanes.py -q

The lane edge exists because `route` used to be a model call that re-derived a fact
`ingest/router.py` already knew (plans/datasurf-plan.md §7). The test that protects that is
not "does it pick the right lane" — it is "does it pick one without calling anything".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import lanes
from agent.state import State


def test_one_kind_of_file_means_one_lane():
    assert lanes.lane_for({"sales": 1}, {}) == "tabular"
    assert lanes.lane_for({}, {"report.pdf": 1}) == "document"
    assert lanes.lane_for({"sales": 1}, {"report.pdf": 1}) == "hybrid"


def test_an_empty_session_is_tabular():
    """The upload prompt renders before any lane matters; nothing may raise here."""
    assert lanes.lane_for({}, {}) == "tabular"


def test_the_edge_reads_an_attribute():
    assert lanes.edge(State(question="q")) == "plan"
    assert lanes.edge(State(question="q", answer="240 rows")) == "synthesize"


def test_every_lane_declares_its_tools_and_rules():
    """M3 fills these in. If a lane is missing a row, a node parameterized by it fails at
    runtime instead of here."""
    for lane in ("tabular", "document", "hybrid"):
        assert lanes.TOOLS[lane] and lanes.VERIFY_RULES[lane]


def test_choosing_a_lane_imports_no_model_client():
    """lanes.py is imported by the graph, but deciding a lane must not touch a client."""
    import subprocess

    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from agent import lanes\n"
         "lanes.lane_for({'a': 1}, {'b': 2})\n"
         "print([m for m in sys.modules if m.split('.')[0] == 'openai'])"],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]"
