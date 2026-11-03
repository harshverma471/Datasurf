"""The zero-token path, protected. Its failure means money is being spent that shouldn't be.

    .venv/bin/pytest evals/test_preroute.py -q

Half these tests assert a *miss*. Tier P is the only tier with no model and no verify behind
it, so a pattern that matches too much answers confidently and wrongly for free. A miss costs
a tenth of a cent; a false hit costs the user's trust (agent/preroute.py).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import preroute


def _tables(**shapes):
    """{'sales': (240, ['region', 'revenue'])} -> something that quacks like Session.tables."""
    out = {}
    for name, (rows, columns) in shapes.items():
        card = type("Card", (), {
            "rows": rows,
            "columns": [type("Col", (), {"name": c})() for c in columns],
        })()
        out[name] = type("Table", (), {"card": card})()
    return out


ONE = _tables(sales=(240, ["region", "revenue", "units"]))
TWO = _tables(sales=(240, ["region", "revenue"]), reps=(18, ["rep_id", "territory"]))


@pytest.mark.parametrize("question", [
    "how many rows", "How many rows?", "how many rows are there?", "how many records",
    "how many rows do we have", "how many entries are there", "what's the number of rows",
    "how many rows in total", "HOW MANY ROWS IS THIS?",
])
def test_row_count_never_reaches_a_model(question):
    assert preroute.answer(question, ONE) == ("`sales` has 240 rows.",
                                              "the Schema Card for sales")


@pytest.mark.parametrize("question, expected", [
    ("how many columns", "`sales` has 3 columns."),
    ("how many fields are there?", "`sales` has 3 columns."),
    ("what is the shape of the data", "`sales` is 240 rows × 3 columns."),
    ("what columns are there?", "`sales` has 3 columns: `region`, `revenue`, `units`."),
    ("list the columns", "`sales` has 3 columns: `region`, `revenue`, `units`."),
    ("column names", "`sales` has 3 columns: `region`, `revenue`, `units`."),
])
def test_the_other_free_questions(question, expected):
    assert preroute.answer(question, ONE)[0] == expected


def test_table_inventory_needs_no_table():
    got, source = preroute.answer("what tables are loaded?", TWO)
    assert got == "2 tables loaded: `sales`, `reps`."
    assert source == "the session inventory"


# --- the half that matters more --------------------------------------------------

@pytest.mark.parametrize("question", [
    "how many rows have a null discount",
    "how many rows in the east region",
    "how many rows per region",
    "how many records sold more than 100 units",
    "how many rows were deduped",
    "how many distinct regions are there",
    "what columns have nulls",
    "which columns are numeric",
    "what is the shape of revenue by region",
    "how many rows and what is the average revenue",
    "list the columns with missing values",
    "how many unique rows",
])
def test_a_constraint_means_it_is_not_a_free_question(question):
    assert preroute.answer(question, ONE) is None


def test_two_tables_and_no_name_is_a_miss():
    """Picking the first table silently would be a wrong answer at zero cost."""
    assert preroute.answer("how many rows", TWO) is None
    assert preroute.answer("how many rows in sales", TWO)[0] == "`sales` has 240 rows."
    assert preroute.answer("how many rows in reps?", TWO)[0] == "`reps` has 18 rows."


def test_naming_both_tables_is_a_miss():
    assert preroute.answer("how many rows in sales vs reps", TWO) is None


def test_nothing_loaded_is_a_miss():
    assert preroute.answer("how many rows", {}) is None
