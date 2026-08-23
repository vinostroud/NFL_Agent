import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import streamlit as st
from agents.mcp import MCPServerStdio

# --- Prompt + local model config ---
PROMPTS_DIR = Path(__file__).parent / "prompts"
PLANNER_PROMPT = (PROMPTS_DIR / "planner_system.txt").read_text()
OLLAMA_BIN = shutil.which("ollama") or "ollama"
DEFAULT_MODEL = "llama3.1:8b"

# Season assumed when the question does not name one.
DEFAULT_SEASON = 2024

# Metric column produced by each tool, used for charting.
_TOOL_METRIC_COL = {
    "get_team_epa_per_play": "epa_per_play",
    "get_epa_per_dropback": "epa_per_dropback",
    "get_success_rate": "success_rate",
    "get_adjusted_ratings": "net_adj",
    "get_strength_of_schedule": "off_adj",
    "get_matchup_projection": "proj_epa_per_play",
}

_VALID_TOOLS = {
    "get_team_epa_per_play",
    "get_success_rate",
    "get_epa_per_dropback",
    "get_qb_stats",
    "get_adjusted_ratings",
    "get_strength_of_schedule",
    "get_matchup_projection",
}

# Tools whose "side" concept does not apply (they rate both sides at once).
_SIDELESS_TOOLS = {
    "get_adjusted_ratings",
    "get_strength_of_schedule",
    "get_matchup_projection",
}

# Metrics where a LOWER value is better (defensive stats: EPA and success rate
# allowed). Rank 1 already accounts for this; the chart needs to as well.
_DEFENSIVE_TOOLS = {
    "get_team_epa_per_play",
    "get_epa_per_dropback",
    "get_success_rate",
}


class PlannerError(RuntimeError):
    """Raised when the local model fails to produce a usable plan."""


# ---------- Local model call ----------


def ollama_chat(prompt: str, model: str = DEFAULT_MODEL, json_mode: bool = False) -> str:
    cmd = [OLLAMA_BIN, "run", model]
    if json_mode:
        cmd.extend(["--format", "json"])

    try:
        proc = subprocess.run(
            cmd,
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    except FileNotFoundError:
        raise PlannerError(
            "Could not find the `ollama` executable. Install Ollama and ensure "
            "it is on your PATH."
        )
    except subprocess.TimeoutExpired:
        raise PlannerError("The local model timed out after 120s.")

    if proc.returncode != 0:
        raise PlannerError(
            "Ollama call failed. Check that Ollama is running and the model is "
            f"pulled (`ollama pull {model}`).\n\n"
            f"{proc.stderr.decode('utf-8', errors='replace')}"
        )

    return proc.stdout.decode("utf-8", errors="replace")


# ---------- Planner: question -> {seasons, season_type, tool, ...params} ----------


def _normalize_plan(plan: dict) -> dict:
    """
    Coerce a raw model plan into something the MCP tools will accept.

    The planner is a small local model, so every field is validated and given a
    safe fallback rather than trusted.
    """
    # Seasons
    raw_seasons = plan.get("seasons", plan.get("season"))
    seasons: list[int] = []
    if raw_seasons is not None:
        if not isinstance(raw_seasons, list):
            raw_seasons = [raw_seasons]
        for value in raw_seasons:
            try:
                seasons.append(int(value))
            except (TypeError, ValueError):
                continue
    plan["seasons"] = seasons or [DEFAULT_SEASON]
    plan.pop("season", None)

    # Season type
    stype = str(plan.get("season_type", "REG")).upper()
    plan["season_type"] = stype if stype in ("REG", "POST") else "REG"

    # Tool
    if plan.get("tool") not in _VALID_TOOLS:
        plan["tool"] = "get_team_epa_per_play"

    # Tool-specific parameters. Drop anything not relevant to the chosen tool so
    # the MCP call does not receive unexpected keyword arguments.
    tool = plan["tool"]
    if tool in ("get_team_epa_per_play", "get_success_rate"):
        if plan.get("side") not in ("offense", "defense"):
            plan["side"] = "offense"
        if plan.get("kind") not in ("all", "pass", "rush"):
            plan["kind"] = "all"
        plan.pop("metric", None)
        plan.pop("min_dropbacks", None)
    elif tool == "get_epa_per_dropback":
        if plan.get("side") not in ("offense", "defense"):
            plan["side"] = "offense"
        plan.pop("kind", None)
        plan.pop("metric", None)
        plan.pop("min_dropbacks", None)
    elif tool == "get_qb_stats":
        if plan.get("metric") not in ("epa_per_dropback", "cpoe"):
            plan["metric"] = "epa_per_dropback"
        plan.pop("side", None)
        plan.pop("kind", None)
    elif tool in ("get_adjusted_ratings", "get_strength_of_schedule"):
        if plan.get("kind") not in ("all", "pass", "rush"):
            plan["kind"] = "all"
        for key in ("side", "metric", "min_dropbacks", "home_team", "away_team"):
            plan.pop(key, None)
    elif tool == "get_matchup_projection":
        if plan.get("kind") not in ("all", "pass", "rush"):
            plan["kind"] = "all"
        plan["neutral_site"] = bool(plan.get("neutral_site", False))
        for key in ("home_team", "away_team"):
            value = plan.get(key)
            plan[key] = str(value).upper().strip() if value else ""
        if not plan["home_team"] or not plan["away_team"]:
            # Without two teams there is nothing to project; fall back rather
            # than sending an invalid call to the server.
            plan["tool"] = "get_adjusted_ratings"
            for key in ("home_team", "away_team", "neutral_site"):
                plan.pop(key, None)
        for key in ("side", "metric", "min_dropbacks"):
            plan.pop(key, None)

    return plan


def plan_metric_from_question(question: str, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"{PLANNER_PROMPT}\n\nUser question:\n{question}\n\nJSON only:\n"
    raw = ollama_chat(prompt, model=model, json_mode=True)

    try:
        plan = json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise PlannerError(
            f"The planner did not return valid JSON ({e}).\n\nRaw output:\n{raw}"
        )

    if not isinstance(plan, dict):
        raise PlannerError(f"The planner returned {type(plan).__name__}, not an object.")

    return _normalize_plan(plan)


# ---------- MCP tool result unpacker ----------


def unpack_tool_result(tool_response):
    if getattr(tool_response, "isError", False):
        raise RuntimeError(f"Tool call failed: {tool_response.content}")

    sc = tool_response.structuredContent
    if sc is None:
        raise RuntimeError(f"No structuredContent in tool response: {tool_response}")

    if isinstance(sc, dict) and "result" in sc:
        return sc["result"]
    return sc


# ---------- Core pipeline: planner -> MCP -> metrics ----------


async def run_planned_metric(question: str):
    plan = plan_metric_from_question(question)
    server_path = Path(__file__).with_name("mcp_nfl_metrics_server.py").resolve()

    async with MCPServerStdio(
        name="NFL Metrics MCP",
        params={"command": sys.executable, "args": [str(server_path)]},
        cache_tools_list=True,
    ) as mcp_server:
        tool_params = {
            "seasons": plan["seasons"],
            "season_type": plan["season_type"],
        }
        for key in (
            "side",
            "kind",
            "metric",
            "min_dropbacks",
            "home_team",
            "away_team",
            "neutral_site",
        ):
            if key in plan:
                tool_params[key] = plan[key]

        result = await mcp_server.call_tool(plan["tool"], tool_params)
        payload = unpack_tool_result(result)

        return plan, payload["summary"], payload["rows"]


@st.cache_data(show_spinner=False)
def answer_question(question: str, model: str = DEFAULT_MODEL):
    return asyncio.run(run_planned_metric(question.strip()))


# ---------- Rendering helpers ----------


def _prepare_table(df: pl.DataFrame, plan: dict):
    """
    Return (display_df, metric_col, index_col) or raise ValueError.
    """
    tool = plan.get("tool")

    if tool == "get_matchup_projection":
        cols = [
            c
            for c in ("offense", "defense", "site", "proj_epa_per_play", "off_adj", "opp_def_adj")
            if c in df.columns
        ]
        return df.select(cols), "proj_epa_per_play", "offense"

    if tool == "get_adjusted_ratings":
        cols = [
            c
            for c in ("rank", "team", "net_adj", "off_adj", "off_rank", "def_adj", "def_rank")
            if c in df.columns
        ]
        return df.select(cols).sort("rank"), "net_adj", "team"

    if tool == "get_strength_of_schedule":
        cols = [
            c
            for c in ("adj_rank", "team", "off_adj", "raw_epa", "raw_rank", "rank_change", "opp_def_faced")
            if c in df.columns
        ]
        return df.select(cols).sort("adj_rank"), "off_adj", "team"

    if tool == "get_qb_stats":
        # Chart the metric the user actually asked to rank by. Previously this
        # preferred cpoe whenever the column had any values, so a question about
        # QB EPA produced an EPA-ranked table beside a CPOE chart.
        metric_col = plan.get("metric", "epa_per_dropback")
        if metric_col not in df.columns:
            metric_col = "epa_per_dropback"
        cols = [
            c
            for c in ("rank", "qb", "team", "season", "dropbacks", "epa_per_dropback", "cpoe")
            if c in df.columns
        ]
        return df.select(cols).sort("rank"), metric_col, "qb"

    metric_col = _TOOL_METRIC_COL.get(plan.get("tool"), "epa_per_play")
    expected = ["rank", "team", metric_col, "plays" if "plays" in df.columns else None]
    expected = [c for c in expected if c and c in df.columns]

    missing = [c for c in ("rank", "team", metric_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in result: {missing}")

    return df.select(expected).sort("rank"), metric_col, "team"


def _render_chart(df: pl.DataFrame, plan: dict, metric_col: str, index_col: str):
    tool = plan.get("tool")
    is_defensive = plan.get("side") == "defense" and tool in _DEFENSIVE_TOOLS

    st.subheader(f"Bar chart: {metric_col}")

    if tool in _SIDELESS_TOOLS:
        # These rate both sides at once, so "higher is better" already holds
        # for the charted column and no sign flip is appropriate.
        st.bar_chart(df, x=index_col, y=metric_col)
        if tool == "get_strength_of_schedule":
            st.caption(
                "Opponent-adjusted offensive EPA per play. See `rank_change` in "
                "the table for how far each team moved once schedule was "
                "accounted for."
            )
        elif tool == "get_matchup_projection":
            st.caption(
                "Projected EPA per play for each offense against the specific "
                "defense it faces. Not a point spread or win probability."
            )
        else:
            st.caption("Net rating = adjusted offense minus adjusted defense.")
        return

    if is_defensive and metric_col != "success_rate":
        # EPA allowed is best when most negative, so flip the sign to keep
        # "taller bar = better" true on the chart.
        chart_df = df.with_columns((-pl.col(metric_col)).alias(metric_col))
        st.bar_chart(chart_df, x=index_col, y=metric_col)
        st.caption("Values negated for display — taller bar = better defense.")
    elif is_defensive:
        # Success rate allowed is a positive proportion; negating it would just
        # produce downward bars. Show the true value and state the direction.
        st.bar_chart(df, x=index_col, y=metric_col)
        st.caption("Success rate allowed — shorter bar = better defense.")
    else:
        st.bar_chart(df, x=index_col, y=metric_col)


# ---------- Streamlit UI ----------


def main():
    st.set_page_config(page_title="NFL EPA Analytics Agent", layout="wide")
    st.title("NFL EPA Analytics Agent")
    st.write(
        "Ask a question about team-level or QB-level metrics. "
        "This app will respond with a table and a chart."
    )

    default_q = "For the 2023 regular season, which offenses had the best EPA per play?"
    with st.form("ask_form", clear_on_submit=False):
        question = st.text_input("Your question:", value=default_q)
        top_n = st.slider("How many rows to show?", min_value=5, max_value=32, value=10)
        submitted = st.form_submit_button("Run analysis")

    if not (submitted and question.strip()):
        return

    try:
        with st.spinner("Computing metrics..."):
            plan, summary, rows = answer_question(question)
    except PlannerError as e:
        st.error("Could not plan this question.")
        st.code(str(e))
        return
    except Exception as e:
        st.error(f"Error while running analysis: {e}")
        return

    with st.expander("Plan", expanded=False):
        st.json(plan)

    st.subheader("Summary")
    st.write(summary)

    if not rows:
        st.warning("No rows returned for that query.")
        return

    try:
        df, metric_col, index_col = _prepare_table(pl.DataFrame(rows), plan)
    except ValueError as e:
        st.error(str(e))
        st.dataframe(pl.DataFrame(rows), use_container_width=True)
        return

    df = df.head(top_n)

    st.subheader("Results table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    _render_chart(df, plan, metric_col, index_col)


if __name__ == "__main__":
    main()
