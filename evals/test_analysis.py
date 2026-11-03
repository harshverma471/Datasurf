"""The deterministic half of M2. No model, no network, no fixture file.

    .venv/bin/pytest evals/test_analysis.py -q

Every table here is built with a planted signal, because the committed fixture is synthetic
uniform data: it has no correlation, no outliers and no segment effect, so running the
presets against it proves they execute, not that they *find* anything. A planted effect with
a known size is the only way to tell "no findings" apart from "found nothing".

The tests that assert a preset stays quiet matter as much as the ones that assert it speaks.
A health report that flags something in clean data is worse than one that flags nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import anomalies, eda, health, matrices, segments

N = 240


def rng():
    """A fresh generator per call, not one shared across the module.

    A module-level `default_rng(7)` is seeded but not reproducible *per test*: every draw
    advances the shared state, so each test's data depends on which tests ran before it.
    Two of these tests passed alone and failed in the suite, which is the worst way to find
    out.
    """
    return np.random.default_rng(7)


@pytest.fixture
def clean() -> pd.DataFrame:
    """No planted signal: every preset should stay quiet about this one."""
    source = rng()
    return pd.DataFrame({
        "order_id": range(1000, 1000 + N),
        "region": source.choice(["North", "South", "East", "West"], N),
        "revenue": source.normal(500, 50, N).round(2),
        "units": source.integers(1, 100, N),
    })


def texts(report) -> str:
    return " ".join(f.text for f in report.findings)


# --- health ----------------------------------------------------------------------

def test_health_is_quiet_on_clean_data(clean):
    assert health.report(clean, "t").findings == []
    assert health.report(clean, "t").headline == "nothing to flag"


def test_health_finds_a_planted_outlier(clean):
    clean.loc[0, "revenue"] = 500_000
    report = health.report(clean, "t")
    assert "revenue has 1 value(s) beyond 3.0×IQR, worst 500,000" in texts(report)
    # the 1.5×IQR count still reaches tier 1 — it is the finding that needs the stricter fence
    assert report.stats[("t", "revenue", "outliers")][0] >= 1


def test_health_ignores_the_outliers_clean_normal_data_always_has(clean):
    """~0.7% of any normal sample sits outside Tukey's 1.5×IQR fence. Flagging those teaches
    the reader to ignore the report."""
    report = health.report(clean, "t")
    assert report.stats[("t", "revenue", "outliers")][0] > 0      # they exist
    assert "revenue" not in texts(report)                          # and are not a finding


def test_health_finds_missingness_and_constants(clean):
    clean.loc[:60, "revenue"] = None
    clean["currency"] = "USD"
    found = texts(health.report(clean, "t"))
    assert "revenue is 25.4% null" in found
    assert "constant columns: currency" in found


def test_health_spots_a_non_unique_identifier(clean):
    clean.loc[5, "order_id"] = clean.loc[4, "order_id"]
    assert "order_id repeats" in texts(health.report(clean, "t"))


def test_health_spots_dates_left_as_text(clean):
    clean["signed_on"] = ["2024-01-0" + str(i % 9 + 1) for i in range(N)]
    assert "signed_on is entirely dates, stored as text" in texts(health.report(clean, "t"))


def test_health_warns_that_zero_padding_is_already_gone(clean):
    clean["store_code"] = [7, 42, 913] * (N // 3)
    found = texts(health.report(clean, "t"))
    assert "store_code is numeric with 1–3 digit values" in found
    assert "zero-padding is already gone" in found


# --- auto-EDA --------------------------------------------------------------------

def test_roles_separate_keys_from_measures(clean):
    roles = dict(zip(eda.report(clean, "t").frames["Column roles"]["Column"],
                     eda.report(clean, "t").frames["Column roles"]["Role"]))
    assert roles == {"order_id": "key", "region": "dimension",
                     "revenue": "measure", "units": "measure"}


def test_a_unique_float_is_a_measure_not_a_key():
    """Cardinality alone called `revenue` a key: 240 distinct values in 240 rows. The Schema
    Card learned this first; the role guess has to use the same integer gate."""
    df = pd.DataFrame({"revenue": np.linspace(1.5, 240.5, N)})
    roles = eda.report(df, "t").frames["Column roles"]
    assert roles.loc[0, "Role"] == "measure"


def test_eda_reports_a_planted_correlation(clean):
    clean["cost"] = clean["revenue"] * 0.6 + rng().normal(0, 5, N)
    found = texts(eda.report(clean, "t"))
    assert "revenue rises with cost" in found or "cost rises with revenue" in found
    assert "r=0.9" in found or "r=1.00" in found


def test_eda_stays_silent_on_a_weak_correlation(clean):
    clean["noise"] = rng().normal(0, 1, N)
    correlations = [f.text for f in eda.report(clean, "t").findings if " r=" in f.text]
    assert correlations == []


def test_eda_reports_skew(clean):
    clean["revenue"] = rng().exponential(400, N)
    assert "revenue is right-skewed" in texts(eda.report(clean, "t"))


# --- matrices --------------------------------------------------------------------

def test_the_correlation_matrix_carries_its_sample_sizes(clean):
    clean.loc[:200, "units"] = None          # one sparse column
    report = matrices.report(clean, "t")
    pairs = report.frames["Complete pairs"]
    assert pairs.loc["revenue", "revenue"] == N
    assert pairs.loc["revenue", "units"] == 39     # not 240: the coefficient saw 39 rows


def test_contingency_finds_a_planted_association(clean):
    clean["tier"] = np.where(clean["region"].isin(["North", "South"]), "A", "B")
    report = matrices.report(clean, "t")
    assert "region and tier are associated" in texts(report)
    assert "region × tier" in report.frames


# --- anomalies -------------------------------------------------------------------

def test_anomalies_is_quiet_on_clean_data(clean):
    assert anomalies.report(clean, "t").headline == "nothing stands out"


def test_anomalies_finds_a_segment_that_is_only_odd_for_its_group(clean):
    """The finding a z-score scan cannot see: in range overall, wrong for its segment."""
    clean.loc[clean["region"] == "East", "revenue"] = 650
    found = texts(anomalies.report(clean, "t"))
    assert "revenue in region=East" in found and "above the overall mean" in found


def test_anomalies_reports_the_sample_size(clean):
    clean.loc[clean["region"] == "East", "revenue"] = 650
    assert "n=" in texts(anomalies.report(clean, "t"))


# --- segments --------------------------------------------------------------------

def test_a_big_lift_on_two_rows_is_not_a_finding():
    """The failure mode this preset exists to prevent."""
    df = pd.DataFrame({
        "region": ["North"] * 2 + ["South"] * 200,
        "revenue": [10_000.0, 11_000.0] + list(rng().normal(500, 50, 200)),
    })
    report = segments.report(df, "t", dimension="region", metric="revenue")
    north = report.frames["revenue by region"].set_index("Segment").loc["North"]
    assert north["n"] == 2 and north["Verdict"] == "too few rows to compare"
    assert "no region segment differs" in texts(report)


def test_a_real_difference_is_reported_with_both_numbers(clean):
    clean.loc[clean["region"] == "East", "revenue"] += 300
    report = segments.report(clean, "t", dimension="region", metric="revenue")
    found = texts(report)
    assert "region=East" in found and "d=" in found and "n=" in found
    east = report.frames["revenue by region"].set_index("Segment").loc["East"]
    assert east["Verdict"] == "large difference" and east["Cohen's d"] > 0.8


def test_significant_but_tiny_is_called_out_as_tiny():
    """With enough rows almost anything reaches p<0.05. The verdict has to say so."""
    df = pd.DataFrame({
        "region": ["North"] * 5000 + ["South"] * 5000,
        "revenue": list(rng().normal(500, 50, 5000)) + list(rng().normal(503, 50, 5000)),
    })
    row = (segments.report(df, "t", dimension="region", metric="revenue")
           .frames["revenue by region"].set_index("Segment").loc["North"])
    assert row["p"] < 0.05
    assert row["Verdict"] == "statistically significant but too small to matter"


def test_it_picks_the_measure_the_dimension_separates(clean):
    """Relative spread picked `units` here — noisiest, and not what anyone opened this for."""
    clean.loc[clean["region"] == "East", "revenue"] += 300
    assert segments.report(clean, "t").preset == "Segment comparison — revenue by region"


def test_nothing_to_compare_says_so():
    report = segments.report(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), "t")
    assert "needs one low-cardinality column" in report.headline
