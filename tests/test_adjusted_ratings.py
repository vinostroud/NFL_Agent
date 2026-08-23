"""
Tests for opponent-adjusted ratings.

The central test is `test_adjustment_corrects_a_schedule_distorted_ranking`,
which builds a league where raw EPA per play ranks the teams WRONG and checks
that the adjusted model ranks them right.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from adjusted_ratings import AdjustedRatings, schedule_report, select_half_life


# ----------------------------------------------------------------------
# Synthetic league builder
# ----------------------------------------------------------------------


def build_league(
    off_true: dict,
    def_true: dict,
    schedule: list,
    plays_per_game: int = 60,
    home_field: float = 0.02,
    noise: float = 0.0,
    seed: int = 0,
) -> pl.DataFrame:
    """
    Generate play-by-play where each play's EPA is exactly
        intercept + home + off_true[offense] + def_true[defense] + noise.

    `schedule` is a list of (home_team, away_team) tuples; both teams get an
    offensive series in each game.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gi, (home, away) in enumerate(schedule):
        game_id = f"G{gi:03d}"
        for offense, defense, is_home in (
            (home, away, 1.0),
            (away, home, 0.0),
        ):
            base = off_true[offense] + def_true[defense] + home_field * is_home
            eps = rng.normal(0.0, noise, plays_per_game) if noise else np.zeros(plays_per_game)
            for k in range(plays_per_game):
                rows.append(
                    {
                        "game_id": game_id,
                        "week": gi // 8 + 1,
                        "season": 2023,
                        "season_type": "REG",
                        "posteam": offense,
                        "defteam": defense,
                        "home_team": home,
                        "away_team": away,
                        "pass": 1,
                        "rush": 0,
                        "qb_kneel": 0,
                        "qb_spike": 0,
                        "epa": float(base + eps[k]),
                    }
                )
    return pl.DataFrame(rows)


def round_robin(teams: list) -> list:
    return [(h, a) for h in teams for a in teams if h != a]


# Offence/defence effects are only separable if the schedule graph is well
# connected. A league where each team faces just two opponents is genuinely
# under-determined -- the model cannot know whether an offence is good or its
# opponents' defences were bad. So the skew is layered on top of a full
# round-robin backbone, which is also how real NFL schedules behave.
SKEW_OFF = {
    "HARD": 0.10, "EASY": 0.05,
    "WALL1": 0.0, "WALL2": 0.0, "SIEVE1": 0.0, "SIEVE2": 0.0,
}
SKEW_DEF = {
    "HARD": 0.0, "EASY": 0.0,
    "WALL1": -0.20, "WALL2": -0.20, "SIEVE1": 0.20, "SIEVE2": 0.20,
}


def skewed_league(extra_games: int = 3) -> pl.DataFrame:
    """
    HARD is the better offence but is fed elite defences; EASY is worse but
    feasts on poor ones.
    """
    schedule = round_robin(list(SKEW_OFF))
    for _ in range(extra_games):
        schedule += [
            ("HARD", "WALL1"),
            ("WALL2", "HARD"),
            ("EASY", "SIEVE1"),
            ("SIEVE2", "EASY"),
        ]
    return build_league(SKEW_OFF, SKEW_DEF, schedule, plays_per_game=60)


# ----------------------------------------------------------------------
# Recovery of known effects
# ----------------------------------------------------------------------


def test_recovers_known_offensive_ordering():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.20, "B": 0.05, "C": -0.05, "D": -0.20}
    dfn = {t: 0.0 for t in teams}
    df = build_league(off, dfn, round_robin(teams), plays_per_game=80)

    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    out, _ = model.ratings()
    assert out.sort("off_rank")["team"].to_list() == ["A", "B", "C", "D"]


def test_recovers_known_defensive_ordering():
    teams = ["A", "B", "C", "D"]
    off = {t: 0.0 for t in teams}
    # More negative = better defence.
    dfn = {"A": -0.15, "B": -0.05, "C": 0.05, "D": 0.15}
    df = build_league(off, dfn, round_robin(teams), plays_per_game=80)

    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    out, _ = model.ratings()
    assert out.sort("def_rank")["team"].to_list() == ["A", "B", "C", "D"]


def test_recovers_home_field_advantage():
    teams = ["A", "B", "C", "D"]
    off = {t: 0.0 for t in teams}
    dfn = {t: 0.0 for t in teams}
    df = build_league(off, dfn, round_robin(teams), home_field=0.05, plays_per_game=80)

    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    assert model.home_field == pytest.approx(0.05, abs=1e-3)


def test_effects_are_centred_on_league_average():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.2, "B": 0.1, "C": -0.1, "D": -0.2}
    dfn = {"A": -0.1, "B": 0.0, "C": 0.05, "D": 0.05}
    df = build_league(off, dfn, round_robin(teams))

    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    assert model.off_effects.mean() == pytest.approx(0.0, abs=1e-9)
    assert model.def_effects.mean() == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------
# The point of the exercise: unbalanced schedules
# ----------------------------------------------------------------------


def test_adjustment_corrects_a_schedule_distorted_ranking():
    """
    HARD is a better offence than EASY, but plays only elite defences while
    EASY plays only terrible ones. Raw EPA per play ranks EASY above HARD;
    the adjusted model must put HARD on top.
    """
    df = skewed_league()

    # Raw ranking is fooled.
    raw = (
        df.group_by("posteam")
        .agg(pl.col("epa").mean().alias("raw"))
        .sort("raw", descending=True)
    )
    raw_order = raw["posteam"].to_list()
    assert raw_order.index("EASY") < raw_order.index("HARD"), (
        "fixture is not exercising the bias it claims to"
    )

    # Adjusted ranking is not.
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    out, _ = model.ratings()
    adj_order = out.sort("off_rank")["team"].to_list()
    assert adj_order.index("HARD") < adj_order.index("EASY")


def test_schedule_report_flags_the_easy_slate():
    df = skewed_league()
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    report, _ = schedule_report(df, model)

    easy = report.filter(pl.col("team") == "EASY")
    hard = report.filter(pl.col("team") == "HARD")

    # EASY faced defences that give up more EPA than average.
    assert easy["opp_def_faced"][0] > 0
    assert hard["opp_def_faced"][0] < 0
    # And EASY loses ground once that is accounted for.
    assert easy["rank_change"][0] < 0
    assert hard["rank_change"][0] > 0


# ----------------------------------------------------------------------
# Shrinkage behaviour
# ----------------------------------------------------------------------


def test_larger_alpha_shrinks_effects_toward_zero():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.3, "B": 0.1, "C": -0.1, "D": -0.3}
    dfn = {t: 0.0 for t in teams}
    df = build_league(off, dfn, round_robin(teams))

    small = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    large = AdjustedRatings.fit(df, alpha=100_000.0, min_plays=10)

    assert np.abs(large.off_effects).max() < np.abs(small.off_effects).max()


def test_shrinkage_preserves_ordering():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.3, "B": 0.1, "C": -0.1, "D": -0.3}
    dfn = {t: 0.0 for t in teams}
    df = build_league(off, dfn, round_robin(teams))

    out, _ = AdjustedRatings.fit(df, alpha=5000.0, min_plays=10).ratings()
    assert out.sort("off_rank")["team"].to_list() == ["A", "B", "C", "D"]


# ----------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------


def test_cv_selects_an_alpha_from_the_grid():
    teams = ["A", "B", "C", "D", "E", "F"]
    off = {t: v for t, v in zip(teams, [0.2, 0.1, 0.05, -0.05, -0.1, -0.2])}
    dfn = {t: 0.0 for t in teams}
    df = build_league(off, dfn, round_robin(teams), noise=1.0, seed=7)

    grid = (1.0, 100.0, 10000.0)
    model = AdjustedRatings.fit(df, alpha_grid=grid, n_folds=3, min_plays=10)
    assert model.alpha in grid
    assert len(model.cv_results) == len(grid)
    assert all(r.mse > 0 for r in model.cv_results)


def test_explicit_alpha_skips_cross_validation():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    model = AdjustedRatings.fit(df, alpha=42.0, min_plays=10)
    assert model.alpha == 42.0
    assert model.cv_results == []


# ----------------------------------------------------------------------
# Matchup projection
# ----------------------------------------------------------------------


def test_matchup_projection_uses_the_specific_opponent():
    off = {"GOOD": 0.15, "BAD": -0.15, "WALL": 0.0, "SIEVE": 0.0}
    dfn = {"GOOD": 0.0, "BAD": 0.0, "WALL": -0.20, "SIEVE": 0.20}
    df = build_league(off, dfn, round_robin(list(off)), plays_per_game=80)
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)

    vs_wall, _ = model.predict_matchup("GOOD", "WALL", neutral_site=True)
    vs_sieve, _ = model.predict_matchup("GOOD", "SIEVE", neutral_site=True)

    good_vs_wall = vs_wall.filter(pl.col("offense") == "GOOD")["proj_epa_per_play"][0]
    good_vs_sieve = vs_sieve.filter(pl.col("offense") == "GOOD")["proj_epa_per_play"][0]
    assert good_vs_sieve > good_vs_wall


def test_home_field_helps_the_home_team():
    teams = ["A", "B", "C", "D"]
    df = build_league(
        {t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams), home_field=0.05
    )
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)

    home, _ = model.predict_matchup("A", "B")
    neutral, _ = model.predict_matchup("A", "B", neutral_site=True)

    a_home = home.filter(pl.col("offense") == "A")["proj_epa_per_play"][0]
    a_neutral = neutral.filter(pl.col("offense") == "A")["proj_epa_per_play"][0]
    assert a_home > a_neutral


def test_neutral_site_matchup_is_symmetric_for_equal_teams():
    teams = ["A", "B", "C", "D"]
    df = build_league(
        {t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams), home_field=0.05
    )
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    out, _ = model.predict_matchup("A", "B", neutral_site=True)
    a, b = out["proj_epa_per_play"].to_list()
    assert a == pytest.approx(b, abs=1e-6)


def test_unknown_team_raises():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    with pytest.raises(ValueError, match="Unknown team"):
        model.predict_matchup("A", "NOPE")


# ----------------------------------------------------------------------
# Guards
# ----------------------------------------------------------------------


def test_thin_sample_teams_raise_rather_than_returning_noise():
    teams = ["A", "B", "C", "D"]
    df = build_league(
        {t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams), plays_per_game=5
    )
    with pytest.raises(ValueError, match="fewer than"):
        AdjustedRatings.fit(df, alpha=1.0, min_plays=1000)


def test_missing_epa_column_raises():
    df = pl.DataFrame({"posteam": ["A"], "defteam": ["B"]})
    with pytest.raises(ValueError, match="epa"):
        AdjustedRatings.fit(df)


def test_kind_filter_is_respected():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    model = AdjustedRatings.fit(df, kind="pass", alpha=1.0, min_plays=10)
    assert model.kind == "pass"
    assert model.n_plays == df.height  # fixture is all pass plays

    with pytest.raises(ValueError, match="No plays available"):
        AdjustedRatings.fit(df, kind="rush", alpha=1.0, min_plays=10)


def test_ratings_frame_shape_and_ranks():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.2, "B": 0.1, "C": -0.1, "D": -0.2}
    df = build_league(off, {t: 0.0 for t in teams}, round_robin(teams))
    out, summary = AdjustedRatings.fit(df, alpha=1.0, min_plays=10).ratings()

    assert out.height == 4
    assert out["rank"].to_list() == [1, 2, 3, 4]
    assert set(out.columns) == {
        "rank", "team", "net_adj", "off_adj", "off_rank",
        "def_adj", "def_rank", "off_plays", "def_plays",
    }
    assert "home-field" in summary.lower() or "home field" in summary.lower()


def test_source_dataframe_is_not_mutated_by_fit():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    before = df.columns.copy()
    AdjustedRatings.fit(df, alpha=1.0, min_plays=10)
    assert df.columns == before


# ----------------------------------------------------------------------
# Recency weighting
# ----------------------------------------------------------------------


def two_era_league(early_off: dict, late_off: dict, weeks_each: int = 6) -> pl.DataFrame:
    """
    A league that CHANGES midway: `early_off` holds for the first block of
    weeks, `late_off` for the second. A recency-weighted fit should land nearer
    the late values; an unweighted fit averages the two eras.
    """
    teams = list(early_off)
    frames = []
    for block, off in enumerate((early_off, late_off)):
        sched = round_robin(teams)
        part = build_league(off, {t: 0.0 for t in teams}, sched, plays_per_game=40)
        # Re-stamp weeks so block 0 is older than block 1.
        part = part.with_columns(
            (pl.lit(block * weeks_each) + pl.col("week")).alias("week")
        )
        frames.append(part)
    out = pl.concat(frames, how="vertical")
    # game_id must stay unique across blocks.
    return out.with_columns(
        (pl.col("game_id") + "_" + pl.col("week").cast(pl.Utf8)).alias("game_id")
    )


def test_recency_weighting_tracks_a_changed_team():
    teams = ["A", "B", "C", "D"]
    early = {"A": -0.20, "B": 0.0, "C": 0.0, "D": 0.20}
    late = {"A": 0.20, "B": 0.0, "C": 0.0, "D": -0.20}  # A and D swap
    df = two_era_league(early, late)

    flat = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=None)
    recent = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=2.0)

    ia = flat.teams.index("A")
    # Unweighted averages the two eras, so A lands near zero.
    assert abs(flat.off_effects[ia]) < 0.05
    # Recency-weighted should follow A's improvement.
    assert recent.off_effects[recent.teams.index("A")] > flat.off_effects[ia]


def test_shorter_half_life_weights_recent_plays_more():
    teams = ["A", "B", "C", "D"]
    early = {"A": -0.20, "B": 0.0, "C": 0.0, "D": 0.20}
    late = {"A": 0.20, "B": 0.0, "C": 0.0, "D": -0.20}
    df = two_era_league(early, late)

    short = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=1.0)
    long_ = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=20.0)

    a_short = short.off_effects[short.teams.index("A")]
    a_long = long_.off_effects[long_.teams.index("A")]
    assert a_short > a_long


def test_no_decay_is_equivalent_to_uniform_weights():
    teams = ["A", "B", "C", "D"]
    off = {"A": 0.2, "B": 0.1, "C": -0.1, "D": -0.2}
    df = build_league(off, {t: 0.0 for t in teams}, round_robin(teams))

    a = AdjustedRatings.fit(df, alpha=10.0, min_plays=10, half_life=None)
    b = AdjustedRatings.fit(df, alpha=10.0, min_plays=10, half_life=float("inf"))
    assert np.allclose(a.off_effects, b.off_effects)


def test_effective_sample_size_shrinks_with_decay():
    teams = ["A", "B", "C", "D"]
    early = {t: 0.0 for t in teams}
    df = two_era_league(early, early)

    flat = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=None)
    decayed = AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=1.0)

    assert flat.effective_n == pytest.approx(flat.n_plays)
    assert decayed.effective_n < flat.effective_n


def test_auto_half_life_is_off_within_a_single_season():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)  # default "auto"
    assert model.half_life is None
    assert model.effective_n == pytest.approx(model.n_plays)


def test_auto_half_life_engages_across_seasons():
    teams = ["A", "B", "C", "D"]
    off = {t: 0.0 for t in teams}
    prev = build_league(off, off, round_robin(teams)).with_columns(
        pl.lit(2022).alias("season")
    )
    cur = build_league(off, off, round_robin(teams)).with_columns(
        pl.lit(2023).alias("season")
    )
    cur = cur.with_columns((pl.col("game_id") + "_b").alias("game_id"))
    df = pl.concat([prev, cur], how="vertical")

    model = AdjustedRatings.fit(df, alpha=1.0, min_plays=10)  # default "auto"
    assert model.half_life == pytest.approx(12.0)
    # Last season is down-weighted, so effective sample is well under raw count.
    assert model.effective_n < model.n_plays


def test_half_life_requires_a_week_column():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    with pytest.raises(ValueError, match="week"):
        AdjustedRatings.fit(df.drop("week"), alpha=1.0, min_plays=10, half_life=4.0)


def test_negative_half_life_raises():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    with pytest.raises(ValueError, match="positive"):
        AdjustedRatings.fit(df, alpha=1.0, min_plays=10, half_life=-3.0)


def test_forward_chaining_selection_returns_a_grid_value():
    teams = ["A", "B", "C", "D"]
    early = {"A": -0.20, "B": 0.0, "C": 0.0, "D": 0.20}
    late = {"A": 0.20, "B": 0.0, "C": 0.0, "D": -0.20}
    df = two_era_league(early, late, weeks_each=6)

    grid = (1.0, 8.0, float("inf"))
    best, results = select_half_life(
        df, grid=grid, min_train_weeks=3, alpha=1.0, min_plays=5
    )
    assert best in grid
    assert len(results) == len(grid)
    assert all(r.n_eval > 0 for r in results)


def test_forward_chaining_prefers_decay_when_the_league_changes():
    """
    In a league that genuinely shifts partway through, forward-chaining
    validation should not pick 'no decay'.
    """
    early = {"A": -0.25, "B": 0.0, "C": 0.0, "D": 0.25}
    late = {"A": 0.25, "B": 0.0, "C": 0.0, "D": -0.25}
    df = two_era_league(early, late, weeks_each=6)

    best, _ = select_half_life(
        df, grid=(1.0, float("inf")), min_train_weeks=3, alpha=1.0, min_plays=5
    )
    assert np.isfinite(best)


def test_forward_chaining_needs_enough_weeks():
    teams = ["A", "B", "C", "D"]
    df = build_league({t: 0.0 for t in teams}, {t: 0.0 for t in teams}, round_robin(teams))
    with pytest.raises(ValueError, match="validate forward"):
        select_half_life(df, min_train_weeks=99, alpha=1.0, min_plays=5)
