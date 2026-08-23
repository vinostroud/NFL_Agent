"""
Regression tests for MetricsExecutor.

Each test in the "regression" section pins a bug that was found in review and
would silently produce wrong answers if reintroduced.
"""

from __future__ import annotations

import polars as pl
import pytest

from metrics_executor import PLAY_KIND, MetricsExecutor


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def make_pbp(**overrides) -> pl.DataFrame:
    """
    A small nflfastR-shaped frame. Crucially it includes columns literally
    named `pass` and `rush`, which is what broke play-type normalization.
    """
    base = {
        "season": [2023] * 8,
        "season_type": ["REG"] * 8,
        "posteam": ["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", "BBB", "BBB"],
        "defteam": ["BBB"] * 4 + ["AAA"] * 4,
        "pass": [1, 0, 1, 0, 1, 0, 0, 0],
        "rush": [0, 1, 0, 0, 0, 1, 0, 0],
        "play_type": [
            "pass",
            "run",
            "pass",
            "punt",
            "pass",
            "run",
            "field_goal",
            "kickoff",
        ],
        "qb_kneel": [0] * 8,
        "qb_spike": [0] * 8,
        "epa": [0.5, -0.2, 1.0, -0.3, 0.1, 0.4, -1.5, 0.9],
    }
    base.update(overrides)
    return pl.DataFrame(base)


# ----------------------------------------------------------------------
# Regression: play-type normalization
# ----------------------------------------------------------------------


def test_play_kind_is_a_string_label_not_the_pass_column():
    """
    `pl.when(...).then("pass")` resolves "pass" to the COLUMN named `pass`,
    yielding 1/0 instead of the label. That made every pass/rush query return
    zero rows.
    """
    ex = MetricsExecutor(make_pbp())
    kinds = ex.df[PLAY_KIND].to_list()
    assert kinds == ["pass", "rush", "pass", None, "pass", "rush", None, None]


def test_original_play_type_column_is_preserved():
    ex = MetricsExecutor(make_pbp())
    assert ex.df["play_type"].to_list()[0] == "pass"
    assert ex.df["play_type"].to_list()[3] == "punt"


def test_pass_and_rush_filters_return_rows():
    ex = MetricsExecutor(make_pbp())
    for kind in ("pass", "rush"):
        out, _ = ex.epa_per_play(side="offense", kind=kind)
        assert out.height > 0, f"kind={kind} returned no rows"


def test_pass_filter_selects_only_pass_plays():
    ex = MetricsExecutor(make_pbp())
    out, _ = ex.epa_per_play(side="offense", kind="pass")
    aaa = out.filter(pl.col("team") == "AAA")
    # AAA pass plays: 0.5 and 1.0 -> mean 0.75, 2 plays
    assert aaa["plays"][0] == 2
    assert aaa["epa_per_play"][0] == pytest.approx(0.75)


def test_unclassifiable_kind_request_raises_instead_of_silently_returning_all():
    df = pl.DataFrame(
        {"posteam": ["AAA"], "defteam": ["BBB"], "epa": [0.5], "season": [2023]}
    )
    ex = MetricsExecutor(df)
    with pytest.raises(ValueError, match="Cannot filter to kind"):
        ex.epa_per_play(side="offense", kind="pass")


# ----------------------------------------------------------------------
# Regression: special teams contaminating "EPA per play"
# ----------------------------------------------------------------------


def test_special_teams_excluded_from_epa_per_play():
    ex = MetricsExecutor(make_pbp())
    out, _ = ex.epa_per_play(side="offense", kind="all")

    aaa = out.filter(pl.col("team") == "AAA")
    # AAA: pass 0.5, rush -0.2, pass 1.0  (punt -0.3 excluded) -> 3 plays
    assert aaa["plays"][0] == 3
    assert aaa["epa_per_play"][0] == pytest.approx((0.5 - 0.2 + 1.0) / 3)

    bbb = out.filter(pl.col("team") == "BBB")
    # BBB: pass 0.1, rush 0.4 (field goal + kickoff excluded) -> 2 plays
    assert bbb["plays"][0] == 2
    assert bbb["epa_per_play"][0] == pytest.approx(0.25)


def test_kneels_and_spikes_excluded():
    df = make_pbp(qb_kneel=[0, 0, 0, 0, 0, 1, 0, 0])
    out, _ = MetricsExecutor(df).epa_per_play(side="offense")
    bbb = out.filter(pl.col("team") == "BBB")
    assert bbb["plays"][0] == 1  # the kneel (a "rush") is dropped


def test_plays_count_matches_rows_used_for_the_mean():
    """`plays` must not count rows whose EPA is null and therefore excluded."""
    df = pl.DataFrame(
        {
            "season": [2023] * 5,
            "posteam": ["AAA"] * 5,
            "defteam": ["BBB"] * 5,
            "pass": [1, 1, 1, 1, 1],
            "rush": [0, 0, 0, 0, 0],
            "epa": [1.0, None, None, None, 2.0],
        }
    )
    out, _ = MetricsExecutor(df).epa_per_play(side="offense")
    assert out["plays"][0] == 2
    assert out["epa_per_play"][0] == pytest.approx(1.5)


def test_null_posteam_rows_are_dropped():
    df = make_pbp(posteam=["AAA", "AAA", "AAA", "AAA", "BBB", "BBB", None, None])
    out, _ = MetricsExecutor(df).epa_per_play(side="offense")
    assert out["team"].null_count() == 0


# ----------------------------------------------------------------------
# Regression: defensive ranking direction
# ----------------------------------------------------------------------


def _defense_frame() -> pl.DataFrame:
    # AAA defends the first four plays and allows mostly negative EPA (good).
    # BBB defends the last four and allows mostly positive EPA (bad).
    return pl.DataFrame(
        {
            "season": [2023] * 8,
            "posteam": ["BBB"] * 4 + ["AAA"] * 4,
            "defteam": ["AAA"] * 4 + ["BBB"] * 4,
            "pass": [1] * 8,
            "rush": [0] * 8,
            "epa": [-0.5, -0.4, -0.3, 0.1, 0.6, 0.7, 0.8, -0.1],
        }
    )


def test_defensive_success_rate_ranks_stingiest_defense_first():
    """
    Defense allowing the MOST successful plays was previously ranked #1.
    """
    out, summary = MetricsExecutor(_defense_frame()).success_rate(side="defense")
    assert out["team"][0] == "AAA"  # allowed 1/4 successes
    assert out["team"][1] == "BBB"  # allowed 3/4 successes
    assert "lowest (best)" in summary


def test_offensive_success_rate_ranks_most_successful_offense_first():
    out, _ = MetricsExecutor(_defense_frame()).success_rate(side="offense")
    assert out["team"][0] == "AAA"  # AAA's offense succeeded 3/4


def test_defensive_epa_ranks_lowest_allowed_first():
    out, _ = MetricsExecutor(_defense_frame()).epa_per_play(side="defense")
    assert out["team"][0] == "AAA"
    assert out["epa_per_play"][0] < out["epa_per_play"][1]


def test_offensive_epa_ranks_highest_first():
    out, _ = MetricsExecutor(_defense_frame()).epa_per_play(side="offense")
    assert out["epa_per_play"][0] > out["epa_per_play"][1]


# ----------------------------------------------------------------------
# Regression: QB dropbacks must include sacks
# ----------------------------------------------------------------------


def _qb_frame() -> pl.DataFrame:
    # Two QBs, four dropbacks each. Both throw identically well, but SLACK
    # takes two brutal sacks. Excluding sacks makes them look equal.
    return pl.DataFrame(
        {
            "season": [2023] * 8,
            "posteam": ["AAA"] * 4 + ["BBB"] * 4,
            "passer": ["CLEAN"] * 4 + ["SLACK"] * 4,
            "qb_dropback": [1] * 8,
            "pass_attempt": [1, 1, 1, 1, 1, 1, 0, 0],
            "sack": [0, 0, 0, 0, 0, 0, 1, 1],
            "epa": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, -3.0, -3.0],
            "cpoe": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, None, None],
        }
    )


def test_qb_dropbacks_include_sacks():
    out, _ = MetricsExecutor(_qb_frame()).qb_epa_cpoe(min_dropbacks=1)
    slack = out.filter(pl.col("qb") == "SLACK")
    assert slack["dropbacks"][0] == 4  # 2 attempts + 2 sacks


def test_sacks_penalize_qb_epa():
    out, _ = MetricsExecutor(_qb_frame()).qb_epa_cpoe(min_dropbacks=1)
    clean = out.filter(pl.col("qb") == "CLEAN")["epa_per_dropback"][0]
    slack = out.filter(pl.col("qb") == "SLACK")["epa_per_dropback"][0]
    assert clean > slack
    assert slack < 0
    assert out["qb"][0] == "CLEAN"  # rank 1


def test_traded_qb_is_one_row_with_both_teams():
    df = pl.DataFrame(
        {
            "season": [2023] * 4,
            "posteam": ["NYJ", "NYJ", "PHI", "PHI"],
            "passer": ["MOVER"] * 4,
            "qb_dropback": [1] * 4,
            "epa": [0.1, 0.2, 0.3, 0.4],
            "cpoe": [1.0, 1.0, 1.0, 1.0],
        }
    )
    out, _ = MetricsExecutor(df).qb_epa_cpoe(min_dropbacks=1)
    assert out.height == 1
    assert out["dropbacks"][0] == 4
    assert out["team"][0] == "NYJ/PHI"


def test_min_dropbacks_filter_applies():
    out, _ = MetricsExecutor(_qb_frame()).qb_epa_cpoe(min_dropbacks=5)
    assert out.is_empty()


def test_cpoe_ranking_ignores_null_only_rows():
    out, _ = MetricsExecutor(_qb_frame()).qb_epa_cpoe(metric="cpoe", min_dropbacks=1)
    assert out.height == 2
    assert out["cpoe"].null_count() == 0


def test_missing_passer_column_raises():
    df = pl.DataFrame({"season": [2023], "posteam": ["AAA"], "epa": [0.1]})
    with pytest.raises(ValueError, match="passer"):
        MetricsExecutor(df).qb_epa_cpoe()


# ----------------------------------------------------------------------
# Aggregation policy: mean, not median
# ----------------------------------------------------------------------


def test_mean_rewards_explosive_offense_where_median_would_not():
    """
    Pins the documented decision to aggregate with mean.

    EXPLO wins on mean (it generates real expected points via explosive plays)
    but LOSES on median, because most of its individual plays are small
    negatives. Ranking by median would invert the correct answer.
    """
    explo = [-0.4, -0.3, -0.3, -0.2, -0.2, -0.1, 0.1, 6.0, 5.0, 4.0]
    dinks = [-0.1, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    df = pl.DataFrame(
        {
            "season": [2023] * 20,
            "posteam": ["EXPLO"] * 10 + ["DINKS"] * 10,
            "defteam": ["ZZZ"] * 20,
            "pass": [1] * 20,
            "rush": [0] * 20,
            "epa": explo + dinks,
        }
    )
    out, _ = MetricsExecutor(df).epa_per_play(side="offense")
    assert out["team"][0] == "EXPLO"

    medians = (
        df.group_by("posteam").agg(pl.col("epa").median().alias("m")).sort("m")
    )
    assert medians["posteam"][-1] == "DINKS"  # median would pick the wrong team


def test_success_rate_is_a_proportion_between_zero_and_one():
    out, _ = MetricsExecutor(make_pbp()).success_rate(side="offense")
    assert out["success_rate"].min() >= 0.0
    assert out["success_rate"].max() <= 1.0


# ----------------------------------------------------------------------
# Ranking mechanics
# ----------------------------------------------------------------------


def test_ties_share_the_same_rank():
    df = pl.DataFrame(
        {
            "season": [2023] * 4,
            "posteam": ["AAA", "AAA", "BBB", "BBB"],
            "defteam": ["ZZZ"] * 4,
            "pass": [1] * 4,
            "rush": [0] * 4,
            "epa": [0.5, 0.5, 0.5, 0.5],
        }
    )
    out, _ = MetricsExecutor(df).epa_per_play(side="offense")
    assert out["rank"].to_list() == [1, 1]


def test_rank_is_sorted_and_starts_at_one():
    out, _ = MetricsExecutor(make_pbp()).epa_per_play(side="offense")
    ranks = out["rank"].to_list()
    assert ranks[0] == 1
    assert ranks == sorted(ranks)


def test_input_dataframe_is_not_mutated():
    df = make_pbp()
    before = df.columns.copy()
    MetricsExecutor(df)
    assert df.columns == before
    assert PLAY_KIND not in df.columns
