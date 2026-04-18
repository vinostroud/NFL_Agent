from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from metrics_executor import MetricsExecutor  

import polars as pl
import nflreadpy as nfl



# ---------- state & context ----------

@dataclass
class SessionState:
    """Holds play-by-play dataframes keyed by session_id."""
    frames: Dict[str, pl.DataFrame] = field(default_factory=dict)


@dataclass
class AppContext:
    state: SessionState


@asynccontextmanager
async def lifespan(server: FastMCP):
    """MCP lifespan hook: create and clean shared state."""
    state = SessionState()
    try:
        yield AppContext(state=state)
    finally:
        state.frames.clear()

mcp = FastMCP("nfl-metrics-mcp", lifespan=lifespan)


def _ensure_session(
    ctx: Context[ServerSession, AppContext],
    session_id: str,
) -> pl.DataFrame:
    frames = ctx.request_context.lifespan_context.state.frames
    if session_id not in frames:
        raise ValueError(
            f"Unknown session_id: {session_id}. "
            f"Call load_pbp first to create a session."
        )
    return frames[session_id]


def _df_to_rows(df: pl.DataFrame, max_rows: int = 200) -> List[Dict[str, Any]]:
    if df.is_empty():
        return []
    return df.head(max_rows).to_dicts()


# ---------- tools ----------

@mcp.tool()
async def list_examples() -> List[str]:
    """
    Return example queries that MetricsExecutor understands.
    """
    return [
        "ranked epa per play",
        "ranked defensive epa per play allowed",
        "ranked rushing epa per play",
        "offensive success rate",
        "defensive pass success rate",
        "epa per dropback",
        "defensive epa per dropback",
        "qb ranked by epa per pass",
        "qb ranked by cpoe",
    ]


@mcp.tool()
async def load_pbp(
    ctx: Context[ServerSession, AppContext],
    seasons: List[int],
    season_type: Literal["REG", "POST"] = "REG",
) -> Dict[str, Any]:
    """
    Load nflfastR play-by-play data for given seasons and season_type.
    """
    # nflreadpy returns a Polars DataFrame
    df = nfl.load_pbp(seasons=seasons)

    # Filter season type if present
    if "season_type" in df.columns:
        df = df.with_columns(pl.col("season_type").cast(pl.Utf8).str.to_uppercase())
        df = df.filter(pl.col("season_type") == season_type)

    # Normalize team columns
    for col_name in ("posteam", "defteam"):
        if col_name in df.columns:
            df = df.with_columns(
                pl.col(col_name).cast(pl.Utf8).str.to_lowercase().alias(col_name)
            )

    session_id = str(uuid.uuid4())
    ctx.request_context.lifespan_context.state.frames[session_id] = df

    await ctx.info(
        f"Loaded {df.height:,} rows for seasons={seasons}, "
        f"season_type={season_type}, session_id={session_id}"
    )

    return {"session_id": session_id, "rows": int(df.height)}


@mcp.tool()
async def compute_metric(
    ctx: Context[ServerSession, AppContext],
    session_id: str,
    query: str,
) -> Dict[str, Any]:
    """
    Run a metrics query using MetricsExecutor.

    Args:
        session_id: id returned by load_pbp
        query: natural-language-ish query understood by MetricsExecutor

    Returns:
        {
          "summary": str,
          "rows": [ {col: value, ...}, ... ]
        }
    """
    df = _ensure_session(ctx, session_id)
    executor = MetricsExecutor(df)
    out_df, summary = executor.execute(query)

    rows = _df_to_rows(out_df, max_rows=200)
    return {"summary": summary, "rows": rows}


@mcp.tool()
async def save_result(
    ctx: Context[ServerSession, AppContext],
    session_id: str,
    query: str,
    path: str = "metrics.csv",
) -> Dict[str, Any]:
    """
    Run a metrics query and save FULL result to CSV.
    """
    df = _ensure_session(ctx, session_id)
    executor = MetricsExecutor(df)
    out_df, summary = executor.execute(query)

    out_df.write_csv(path)
    return {
        "path": os.path.abspath(path),
        "rows_saved": int(out_df.height),
        "summary": summary,
    }


# ---------- entrypoint ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "transport",
        choices=["stdio", "http"],
        help="How to expose this MCP server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "stdio":
        
        mcp.run()
    else:
        
        try:
            from mcp.server.fastmcp import FastMCPHTTPServer
            import uvicorn
        except Exception as e:
            raise RuntimeError(
                "HTTP mode requires 'mcp[http]' and 'uvicorn' installed."
            ) from e

        server = FastMCPHTTPServer(mcp, host=args.host, port=args.port)
        uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()