import sys
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
import streamlit as st
from agents.mcp import MCPServerStdio

# --- Prompt + local model config ---
PLANNER_PROMPT = Path("prompts/planner_system.txt").read_text()
OLLAMA_BIN = shutil.which("ollama") or "ollama"

_TOOL_METRIC_COL = {
    "get_team_epa_per_play": "epa_per_play",
    "get_epa_per_dropback": "epa_per_dropback",
    "get_success_rate": "success_rate",
}

_VALID_TOOLS = {"get_team_epa_per_play", "get_success_rate", "get_epa_per_dropback", "get_qb_stats"}


def ollama_chat(prompt: str, model: str = "llama3.1:8b", json_mode: bool = False) -> str:
    cmd = [OLLAMA_BIN, "run", model]
    if json_mode:
        cmd.extend(["--format", "json"])
    proc = subprocess.run(
        cmd,
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        st.error("Error calling Ollama. Check that Ollama is installed and running.")
        st.text(proc.stderr.decode("utf-8"))
        raise RuntimeError("Ollama call failed")
    return proc.stdout.decode("utf-8")


# ---------- Planner: question -> {seasons, season_type, tool, ...params} ----------

def plan_metric_from_question(question: str, model: str = "llama3.1:8b") -> dict:
    prompt = (
        PLANNER_PROMPT
        + "\n\nUser question:\n"
        + question
        + "\n\nJSON only:\n"
    )

    raw = ollama_chat(prompt, model=model, json_mode=True)

    try:
        plan = json.loads(raw.strip())
    except Exception as e:
        st.write("Planner raw output:")
        st.code(raw)
        raise RuntimeError(f"Failed to parse planner JSON: {e}")

    # Normalize seasons
    if "seasons" not in plan:
        if "season" in plan:
            val = plan["season"]
            plan["seasons"] = [int(x) for x in val] if isinstance(val, list) else [int(val)]
        else:
            plan["seasons"] = [2023]

    # Normalize season_type
    stype = str(plan.get("season_type", "REG")).upper()
    plan["season_type"] = stype if stype in ("REG", "POST") else "REG"

    # Normalize tool
    if plan.get("tool") not in _VALID_TOOLS:
        plan["tool"] = "get_team_epa_per_play"

    # Normalize tool-specific params
    tool = plan["tool"]
    if tool in ("get_team_epa_per_play", "get_success_rate"):
        if plan.get("side") not in ("offense", "defense"):
            plan["side"] = "offense"
        if plan.get("kind") not in ("all", "pass", "rush"):
            plan["kind"] = "all"
    elif tool == "get_epa_per_dropback":
        if plan.get("side") not in ("offense", "defense"):
            plan["side"] = "offense"
    elif tool == "get_qb_stats":
        if plan.get("metric") not in ("epa_per_pass", "cpoe"):
            plan["metric"] = "epa_per_pass"

    return plan


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
        params={
            "command": sys.executable,
            "args": [str(server_path)],
        },
        cache_tools_list=True,
    ) as mcp_server:
        tool_params = {"seasons": plan["seasons"], "season_type": plan["season_type"]}
        for key in ("side", "kind", "metric"):
            if key in plan:
                tool_params[key] = plan[key]

        metric_result = await mcp_server.call_tool(plan["tool"], tool_params)
        metric_payload = unpack_tool_result(metric_result)

        return plan, metric_payload["summary"], metric_payload["rows"]


@st.cache_data(show_spinner=False)
def answer_question(question: str):
    return asyncio.run(run_planned_metric(question.strip()))


# ---------- Streamlit UI ----------

def main():
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

    if submitted and question.strip():
        try:
            with st.spinner("Computing metrics..."):
                plan, summary, rows = answer_question(question)
        except Exception as e:
            st.error(f"Error while running analysis: {e}")
            return

        st.subheader("Plan")
        st.json(plan)

        st.subheader("Summary")
        st.write(summary)

        if not rows:
            st.warning("No rows returned.")
            return

        df = pl.DataFrame(rows)
        is_qb = plan.get("tool") == "get_qb_stats"

        if is_qb:
            metric_col = "cpoe" if "cpoe" in df.columns and df["cpoe"].is_not_null().any() else "epa_per_pass"
            cols = ["rank", "qb", "team", "dropbacks", "epa_per_pass", "cpoe"]
            cols = [c for c in cols if c in df.columns]
            df = df.select(cols).sort("rank")
        else:
            metric_col = _TOOL_METRIC_COL.get(plan.get("tool"), "epa_per_play")
            expected_cols = ["rank", "team", metric_col]
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns in result: {missing}")
                st.dataframe(df, use_container_width=True)
                return
            df = df.select(expected_cols).sort("rank")

        if top_n is not None:
            df = df.head(top_n)

        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)

        st.subheader(f"Bar chart: {metric_col}")
        index_col = "qb" if is_qb else "team"
        is_defensive_epa = (
            plan.get("side") == "defense"
            and plan.get("tool") in ("get_team_epa_per_play", "get_epa_per_dropback")
        )
        if is_defensive_epa:
            chart_df = df.with_columns((-pl.col(metric_col)).alias(metric_col))
            st.bar_chart(chart_df, x=index_col, y=metric_col)
            st.caption("Values negated for display — taller bar = better defense.")
        else:
            st.bar_chart(df, x=index_col, y=metric_col)


if __name__ == "__main__":
    main()
