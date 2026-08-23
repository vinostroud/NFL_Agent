from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Literal, Tuple

import nflreadpy as nfl
import polars as pl
from mcp.server.fastmcp import FastMCP

from adjusted_ratings import AdjustedRatings, schedule_report
from metrics_executor import MetricsExecutor

mcp = FastMCP("nfl-metrics-mcp")

# nflfastR play-by-play begins in 1999.
MIN_SEASON = 1999
MAX_ROWS = 200
SeasonType = Literal["REG", "POST"]


def _validate_seasons(seasons: List[int]) -> Tuple[int, ...]:
    if not seasons:
        raise ValueError("At least one season must be provided.")

    cleaned = []
    for s in seasons:
        try:
            s_int = int(s)
        except (TypeError, ValueError):
            raise ValueError(f"Season {s!r} is not an integer.")
        if s_int < MIN_SEASON:
            raise ValueError(
                f"Season {s_int} is out of range; play-by-play data starts at "
                f"{MIN_SEASON}."
            )
        cleaned.append(s_int)

    return tuple(sorted(set(cleaned)))


@lru_cache(maxsize=8)
def _load_pbp_cached(seasons: Tuple[int, ...], season_type: str) -> pl.DataFrame:
    """
    Load and filter play-by-play data.

    Cached because every tool call would otherwise re-download a full season of
    play-by-play from nflreadpy. The cache key is the exact (seasons,
    season_type) pair, so results stay correct.
    """
    df = nfl.load_pbp(seasons=list(seasons))

    if "season_type" in df.columns:
        df = df.with_columns(
            pl.col("season_type").cast(pl.Utf8).str.to_uppercase()
        ).filter(pl.col("season_type") == season_type)

    if df.is_empty():
        raise ValueError(
            f"No {season_type} play-by-play data found for seasons {list(seasons)}."
        )

    return df


def _load_pbp(seasons: List[int], season_type: str) -> pl.DataFrame:
    return _load_pbp_cached(_validate_seasons(seasons), season_type)


def _df_to_rows(df: pl.DataFrame, max_rows: int = MAX_ROWS) -> List[Dict[str, Any]]:
    if df.is_empty():
        return []
    return df.head(max_rows).to_dicts()


def _respond(out_df: pl.DataFrame, summary: str) -> Dict[str, Any]:
    rows = _df_to_rows(out_df)
    truncated = out_df.height > len(rows)
    return {
        "summary": summary,
        "rows": rows,
        "row_count": out_df.height,
        "truncated": truncated,
    }


# ---------- tools ----------


@mcp.tool()
async def get_team_epa_per_play(
    seasons: List[int],
    season_type: SeasonType = "REG",
    side: Literal["offense", "defense"] = "offense",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Team-level EPA per play (mean), limited to pass and rush plays.

    side: 'offense' (EPA generated) or 'defense' (EPA allowed)
    kind: 'all', 'pass', or 'rush'
    Rank 1 is always best for the requested side.
    """
    df = _load_pbp(seasons, season_type)
    out_df, summary = MetricsExecutor(df).epa_per_play(side=side, kind=kind)
    return _respond(out_df, summary)


@mcp.tool()
async def get_success_rate(
    seasons: List[int],
    season_type: SeasonType = "REG",
    side: Literal["offense", "defense"] = "offense",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Team-level success rate: the fraction of plays with positive EPA.

    side: 'offense' (own success rate) or 'defense' (success rate allowed)
    kind: 'all', 'pass', or 'rush'
    Rank 1 is always best: highest for offense, lowest allowed for defense.
    """
    df = _load_pbp(seasons, season_type)
    out_df, summary = MetricsExecutor(df).success_rate(side=side, kind=kind)
    return _respond(out_df, summary)


@mcp.tool()
async def get_epa_per_dropback(
    seasons: List[int],
    season_type: SeasonType = "REG",
    side: Literal["offense", "defense"] = "offense",
) -> Dict[str, Any]:
    """
    Team-level EPA per dropback (passing efficiency).

    Dropbacks include pass attempts, sacks and scrambles.
    Rank 1 is always best for the requested side.
    """
    df = _load_pbp(seasons, season_type)
    out_df, summary = MetricsExecutor(df).epa_per_dropback(side=side)
    return _respond(out_df, summary)


@mcp.tool()
async def get_qb_stats(
    seasons: List[int],
    season_type: SeasonType = "REG",
    metric: Literal["epa_per_dropback", "cpoe"] = "epa_per_dropback",
    min_dropbacks: int = 100,
) -> Dict[str, Any]:
    """
    QB-level stats ranked by EPA per dropback or CPOE.

    Dropbacks include sacks and scrambles, so sack-prone quarterbacks are
    correctly penalised. QBs traded mid-season appear as a single row.

    min_dropbacks: minimum dropbacks required to appear (default 100).
    """
    df = _load_pbp(seasons, season_type)
    out_df, summary = MetricsExecutor(df).qb_epa_cpoe(
        metric=metric, min_dropbacks=min_dropbacks
    )
    return _respond(out_df, summary)


@lru_cache(maxsize=8)
def _fit_ratings(seasons: Tuple[int, ...], season_type: str, kind: str) -> AdjustedRatings:
    """
    Fit and cache the ridge model. Fitting scans every play, so repeated
    questions about the same season should not refit.
    """
    df = _load_pbp_cached(seasons, season_type)
    return AdjustedRatings.fit(df, kind=kind)  # type: ignore[arg-type]


@mcp.tool()
async def get_adjusted_ratings(
    seasons: List[int],
    season_type: SeasonType = "REG",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Opponent-adjusted team ratings, controlling for strength of schedule.

    Fits a ridge regression of EPA on offense, defense and home field across
    every play, so each team is rated against a league average opponent rather
    than against whoever it happened to play. Use this instead of
    get_team_epa_per_play whenever the question is about how good a team really
    is, or when comparing teams with different schedules — especially early in
    a season.

    Returns off_adj (higher better), def_adj (lower better) and net_adj in EPA
    per play, plus each team's rank.

    When more than one season is requested, older plays are down-weighted with
    a 12-week half-life; a single season is weighted evenly. The summary states
    which applied.
    """
    seasons_t = _validate_seasons(seasons)
    model = _fit_ratings(seasons_t, season_type, kind)
    out_df, summary = model.ratings()
    return _respond(out_df, summary)


@mcp.tool()
async def get_strength_of_schedule(
    seasons: List[int],
    season_type: SeasonType = "REG",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Compare raw and opponent-adjusted offensive EPA per play.

    Shows the average adjusted quality of the defenses each offense faced
    (`opp_def_faced`; positive means an easy slate) and how many rank places a
    team gains or loses once schedule is accounted for. Use this for questions
    about who was helped or hurt by their schedule, or who is overrated.
    """
    seasons_t = _validate_seasons(seasons)
    model = _fit_ratings(seasons_t, season_type, kind)
    df = _load_pbp_cached(seasons_t, season_type)
    out_df, summary = schedule_report(df, model)
    return _respond(out_df, summary)


@mcp.tool()
async def get_matchup_projection(
    home_team: str,
    away_team: str,
    seasons: List[int],
    season_type: SeasonType = "REG",
    kind: Literal["all", "pass", "rush"] = "all",
    neutral_site: bool = False,
) -> Dict[str, Any]:
    """
    Project EPA per play for both offenses in a specific matchup.

    Each offense is evaluated against the actual defense it faces, using
    opponent-adjusted ratings plus home-field advantage. Team names are
    standard nflfastR abbreviations (SF, KC, DAL...).

    This is a per-play efficiency projection, not a point spread or a win
    probability, and it does not account for injuries, weather or rest.
    """
    seasons_t = _validate_seasons(seasons)
    model = _fit_ratings(seasons_t, season_type, kind)
    out_df, summary = model.predict_matchup(
        home_team=home_team.upper().strip(),
        away_team=away_team.upper().strip(),
        neutral_site=neutral_site,
    )
    return _respond(out_df, summary)


# ---------- entrypoint ----------

if __name__ == "__main__":
    mcp.run()
