import sys
import asyncio
import json
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from agents.mcp import MCPServerStdio

# --- Prompt + local model config ---
PLANNER_PROMPT = Path("prompts/planner_system.txt").read_text()
OLLAMA_BIN = "/usr/local/bin/ollama"


def ollama_chat(prompt: str, model: str = "llama3.1:8b") -> str:
    """
    Call a local Ollama model and return its text response as a string.
    """
    proc = subprocess.run(
        [OLLAMA_BIN, "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        st.error("Error calling Ollama. Check that Ollama is installed and running.")
        st.text(proc.stderr.decode("utf-8"))
        raise RuntimeError("Ollama call failed")
    return proc.stdout.decode("utf-8")


# ---------- Planner: question -> {seasons, season_type, query} ----------

def plan_metric_from_question(question: str, model: str = "llama3.1:8b") -> dict:
    """
    Use the local model to decide:
      - seasons (list[int])
      - season_type ("REG" or "POST")
      - query string for MetricsExecutor (e.g. "ranked epa per play").
    """
    prompt = (
        PLANNER_PROMPT
        + "\n\nUser question:\n"
        + question
        + "\n\nJSON only:\n"
    )

    raw = ollama_chat(prompt, model=model)

    # Try to find the first JSON object in the response
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_str = raw[start:end]
        plan = json.loads(json_str)
    except Exception as e:
        st.write("Planner raw output:")
        st.code(raw)
        raise RuntimeError(f"Failed to parse planner JSON: {e}")

    # --- Normalize / add defaults ---

    # seasons: if missing, try "season" or default to [2023]
    if "seasons" not in plan:
        if "season" in plan:
            val = plan["season"]
            if isinstance(val, list):
                plan["seasons"] = [int(x) for x in val]
            else:
                plan["seasons"] = [int(val)]
        else:
            plan["seasons"] = [2023]

    # season_type: default to REG if missing/invalid
    stype = str(plan.get("season_type", "REG")).upper()
    if stype not in ("REG", "POST"):
        stype = "REG"
    plan["season_type"] = stype

    # query: must exist
    if "query" not in plan or not str(plan["query"]).strip():
        plan["query"] = "ranked epa per play"

    # --- QB-specific override based on the *user's* question ---
    q_lower = question.lower()
    if "qb" in q_lower or "quarterback" in q_lower or "quarterbacks" in q_lower:
        if "cpoe" in q_lower or "completion percentage over expected" in q_lower:
            plan["query"] = "qb ranked by cpoe"
        elif "epa per pass" in q_lower or ("epa" in q_lower and "pass" in q_lower):
            plan["query"] = "qb ranked by epa per pass"

    return plan


# ---------- MCP tool result unpacker ----------

def unpack_tool_result(tool_response):
    """
    Normalize MCP tool responses so we always get the actual payload dict.
    """
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
    """
    Full pipeline:
      1. Ask local model to plan seasons/season_type/query.
      2. Use MCP to load_pbp and compute_metric.
      3. Return the plan, summary, and rows.
    """
    plan = plan_metric_from_question(question)
    seasons = plan["seasons"]
    season_type = plan["season_type"]
    query = plan["query"]

    server_path = Path(__file__).with_name("mcp_nfl_metrics_server.py").resolve()

    async with MCPServerStdio(
        name="NFL Metrics MCP",
        params={
            "command": sys.executable,
            "args": [str(server_path), "stdio"],
        },
        cache_tools_list=True,
    ) as mcp_server:
        # 1) Load play-by-play
        load_result = await mcp_server.call_tool(
            "load_pbp",
            {
                "seasons": seasons,
                "season_type": season_type,
            },
        )
        load_payload = unpack_tool_result(load_result)
        session_id = load_payload["session_id"]

        # 2) Compute requested metric
        metric_result = await mcp_server.call_tool(
            "compute_metric",
            {
                "session_id": session_id,
                "query": query,
            },
        )
        metric_payload = unpack_tool_result(metric_result)
        summary = metric_payload["summary"]
        rows = metric_payload["rows"]

        return plan, summary, rows


@st.cache_data(show_spinner=False)
def answer_question(question: str):
    """
    Cached synchronous wrapper for the async run_planned_metric, for Streamlit.
    Streamlit will cache the result based on the question string.
    """
    normalized_question = question.strip()
    return asyncio.run(run_planned_metric(normalized_question))


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
        submitted = st.form_submit_button("Run analysis")  # Enter submits too

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

        df = pd.DataFrame(rows)
        is_qb = "qb" in df.columns

        if is_qb:
            metric_col = "cpoe" if "cpoe" in df.columns and df["cpoe"].notna().any() else "epa_per_pass"
            cols = ["rank", "qb", "team", "dropbacks", "epa_per_pass", "cpoe"]
            cols = [c for c in cols if c in df.columns]
            df = df[cols].sort_values("rank")
        else:
            metric_col = "epa_per_play"
            expected_cols = ["rank", "team", metric_col]
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                st.error(f"Missing columns in result: {missing}")
                st.dataframe(df, use_container_width=True)
                return
            df = df[expected_cols].sort_values("rank")

        # Apply top_n limit
        if top_n is not None:
            df = df.head(top_n)

        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)

        st.subheader(f"Bar chart: {metric_col}")
        index_col = "qb" if is_qb else "team"
        chart_df = df.set_index(index_col)[metric_col]
        st.bar_chart(chart_df)


if __name__ == "__main__":
    main()