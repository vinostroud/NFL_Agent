from __future__ import annotations

from typing import Any, Dict, List, Literal
from mcp.server.fastmcp import FastMCP
from metrics_executor import MetricsExecutor

import polars as pl
import nflreadpy as nfl


mcp = FastMCP("nfl-metrics-mcp")


def _load_pbp(seasons: List[int], season_type: str) -> pl.DataFrame:
    df = nfl.load_pbp(seasons=seasons)
    if "season_type" in df.columns:
        df = df.with_columns(pl.col("season_type").cast(pl.Utf8).str.to_uppercase())
        df = df.filter(pl.col("season_type") == season_type)
    return df


def _df_to_rows(df: pl.DataFrame, max_rows: int = 200) -> List[Dict[str, Any]]:
    if df.is_empty():
        return []
    return df.head(max_rows).to_dicts()


# ---------- tools ----------

@mcp.tool()
async def get_team_epa_per_play(
    seasons: List[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Compute team-level EPA per play.
    side: 'offense' or 'defense'
    kind: 'all', 'pass', or 'rush'
    """
    df = _load_pbp(seasons, season_type)
    executor = MetricsExecutor(df)
    out_df, summary = executor.epa_per_play(side=side, kind=kind)
    return {"summary": summary, "rows": _df_to_rows(out_df)}


@mcp.tool()
async def get_success_rate(
    seasons: List[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
    kind: Literal["all", "pass", "rush"] = "all",
) -> Dict[str, Any]:
    """
    Compute team-level success rate (fraction of plays with positive EPA).
    side: 'offense' or 'defense'
    kind: 'all', 'pass', or 'rush'
    """
    df = _load_pbp(seasons, season_type)
    executor = MetricsExecutor(df)
    out_df, summary = executor.success_rate(side=side, kind=kind)
    return {"summary": summary, "rows": _df_to_rows(out_df)}


@mcp.tool()
async def get_epa_per_dropback(
    seasons: List[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
) -> Dict[str, Any]:
    """
    Compute team-level EPA per dropback (passing efficiency).
    side: 'offense' or 'defense'
    """
    df = _load_pbp(seasons, season_type)
    executor = MetricsExecutor(df)
    out_df, summary = executor.epa_per_dropback(side=side)
    return {"summary": summary, "rows": _df_to_rows(out_df)}


@mcp.tool()
async def get_qb_stats(
    seasons: List[int],
    season_type: Literal["REG", "POST"] = "REG",
    metric: Literal["epa_per_pass", "cpoe"] = "epa_per_pass",
) -> Dict[str, Any]:
    """
    Compute QB-level stats ranked by epa_per_pass or cpoe.
    Only QBs with at least 100 dropbacks are included.
    """
    df = _load_pbp(seasons, season_type)
    executor = MetricsExecutor(df)
    out_df, summary = executor.qb_epa_cpoe(metric=metric)
    return {"summary": summary, "rows": _df_to_rows(out_df)}


# ---------- entrypoint ----------

if __name__ == "__main__":
    mcp.run()
