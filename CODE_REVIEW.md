# NFL Agent — Code Review

A review of the project at `~/NFL_Agent`, focused on code quality, dead code, and how to redesign the MCP tool surface so questions like *"EPA per team"*, *"run EPA per team"*, or *"third-down EPA for KC"* route correctly.

---

## 0. Urgent — rotate your OpenAI key

`.env` contains a live `OPENAI_API_KEY`. Because `.env` is git-ignored it may not be in your repo, but it is sitting plaintext in the workspace folder (and I was able to read it). **Rotate that key now** at platform.openai.com and never paste it anywhere. Then keep `.env` out of version control (which `.gitignore` already does — good).

Note: you don't actually use OpenAI anywhere in the code I see — everything runs through local Ollama — so you could probably delete the key entirely.

---

## 1. One important factual correction

You said the project "uses the `nfl_data_py` library for all data." That's true for the three notebooks, but **the production path (`mcp_nfl_metrics_server.py` → `app_st_nfl_agent.py`) actually uses `nflreadpy`**, not `nfl_data_py`. They are different libraries:

| library | return type | status |
|---|---|---|
| `nfl_data_py` | pandas | older, still popular |
| `nflreadpy` | polars | newer, actively maintained |

`requirements_mcp.txt` lists only `nflreadpy`, and `mcp_nfl_metrics_server.py` imports `nflreadpy as nfl` and calls `nfl.load_pbp(seasons=seasons)`. Meanwhile the notebooks call `nfl.import_pbp_data(...)` which is `nfl_data_py`.

**Pick one and remove the other.** I recommend standardizing on `nflreadpy` because (a) it returns Polars natively (no `pl.from_pandas` conversion), (b) it's the one your server already uses, and (c) the rest of the pipeline is Polars. Then update the planner prompt — it currently tells the LLM that the app uses `nfl_data_py`, which is wrong.

---

## 2. Dead / unused files and code

Safe to delete (or move to an `archive/` folder):

- `__pycache__/` — compiled Python; never commit this. Add `__pycache__/` to `.gitignore`.
- `.ipynb_checkpoints/` (both the root one and `prompts/.ipynb_checkpoints/` and `tests/.ipynb_checkpoints/`) — Jupyter autosave junk. Add `.ipynb_checkpoints/` to `.gitignore`.
- `.DS_Store` — macOS finder metadata. Add `.DS_Store` to `.gitignore`.
- `agent_app_w_mcp.ipynb` (66 KB) — this was your scratch notebook and is now superseded by `app_st_nfl_agent.py`. The notebook still contains a copy of `ollama_chat`, `plan_metric_from_question`, `unpack_tool_result`, `run_planned_metric`, and `summarize_with_table_and_chart`. **Keeping two copies guarantees they will drift** (and they already have — the notebook has an `explain_metric` function that the Streamlit app doesn't use). Delete the notebook or rename it `archive/agent_dev_scratch.ipynb`.
- `demo_for_qb_stats.ipynb` — your own cell 0 says *"Keeping it for posterity"*. It's already fully absorbed into `_qb_epa_cpoe()`. Move to `archive/`.
- `data_cleaning_nfl_agent.ipynb` — the cleaning logic here is also in `MetricsExecutor._normalize_play_type` / `_ensure_success_flag`, so keeping both invites drift. Move to `archive/` once you're confident the executor matches.

Dead code inside live files:

- `save_result` MCP tool (`mcp_nfl_metrics_server.py` lines 144–163) — never called from `app_st_nfl_agent.py`. Either wire up a "Download CSV" button in the Streamlit UI or remove it. (`st.download_button` does this more naturally without touching the filesystem, by the way.)
- `list_examples` MCP tool — never called. If you want it as a "hint" surface for the LLM, call it during planning. Otherwise delete.
- `FastMCPHTTPServer` branch in `main()` — the import path `from mcp.server.fastmcp import FastMCPHTTPServer` doesn't exist in the current `mcp` package; this branch will raise as soon as you try `http`. Either delete the HTTP branch or re-implement with `mcp.run(transport="streamable-http")`, which is the supported path.
- `explain_metric()` in the notebook — orphaned; decide whether you want an LLM narration step in the Streamlit app and move it in, or delete.

---

## 3. Bugs and correctness issues

### 3a. Team names are being lowercased
Both `mcp_nfl_metrics_server.load_pbp` and `MetricsExecutor._normalize_teams` force `posteam`/`defteam` to lowercase. nflfastR's convention is uppercase (`KC`, `GB`, `SF`). Your output tables therefore show `kc`, `gb`, which is ugly and makes joins against any other nflfastR-derived table fail. **Remove the `.str.to_lowercase()` calls** — or change them to `.str.to_uppercase()` if you want to be defensive against inconsistent input.

Also: you lowercase teams in two places. Do it once, in `MetricsExecutor.__init__`, and drop it from the server.

### 3b. `pl.count()` is deprecated
Polars ≥ 0.20 prefers `pl.len()` inside `.agg()` for "number of rows in the group". `pl.count()` still works for now but emits a deprecation warning and will be removed. Replace the three occurrences in `metrics_executor.py` (`_epa_per_play`, `_success_rate`, `_epa_per_dropback`).

### 3c. "run" isn't an alias for "rush" in the router
`_normalize_play_type` treats both `run` and `rush` as rushes — good. But the router in `execute()` only checks `if "rush" in q`, so a user question like *"run EPA per team"* or *"which offense had the best running EPA"* falls through to all-plays EPA. Add:
```python
if "rush" in q or "run" in q or "rushing" in q or "running" in q:
    ...
```
or better, see §4 below.

### 3d. Defensive bar chart is misleading
For defensive metrics, `rank=1` means *lowest EPA allowed* (best defense), so the bar chart shows defenses sorted by rank but the bars themselves are the raw EPA — i.e. the "best" team has the longest (most negative) bar, and the eye reads the chart upside-down. Either:
- Label the chart "EPA/play allowed — more negative is better", or
- For defensive metrics, chart `-epa_per_play` so bigger always means better.

### 3e. JSON extraction from Ollama is fragile
```python
start = raw.index("{")
end = raw.rindex("}") + 1
```
This breaks the moment the model emits `{` inside prose before the JSON (e.g. in a chain-of-thought preamble or a markdown code fence containing `}` inside a comment). Use Ollama's JSON mode instead:
```python
proc = subprocess.run(
    [OLLAMA_BIN, "run", "--format", "json", model],
    ...
)
```
Or — see §5 — switch from `subprocess` to the Ollama HTTP API, where `format="json"` is a first-class parameter.

### 3f. The planner's QB override runs *twice*, inconsistently
The notebook `agent_app_w_mcp.ipynb` and the Streamlit app `app_st_nfl_agent.py` both implement `plan_metric_from_question`, and they have *slightly different* override rules. The notebook also overrides for "offensive epa". This is the #1 reason to delete the notebook copy.

### 3g. `OLLAMA_BIN = "/usr/local/bin/ollama"` is hard-coded
Works on your Mac; breaks on anybody else's machine and breaks if you `brew upgrade` into `/opt/homebrew`. Use `shutil.which("ollama") or "ollama"`.

### 3h. Session abstraction is pointless in stdio mode
Every Streamlit query spawns a new MCP subprocess, calls `load_pbp` (re-downloads PBP), gets a `session_id`, then calls `compute_metric`, then the subprocess exits. The `session_id` never outlives a single request. Either:
- Make the MCP server long-lived (HTTP mode) so the session cache actually helps, or
- Collapse `load_pbp` + `compute_metric` into one tool per metric and drop the session plumbing (this is what I recommend — see §4).

Either way, add an on-disk cache for PBP by `(seasons, season_type)` so you don't re-download the same 50k-row season every question:
```python
@functools.lru_cache(maxsize=8)
def _load_pbp_cached(seasons_tuple, season_type):
    ...
```
Note: the `lru_cache` must key on a hashable, so pass `tuple(seasons)`.

---

## 4. Redesigning the tool surface (your main question)

Right now there's effectively **one** tool that takes a natural-language string and routes it with a pile of `if "epa" in q` checks. That's why "EPA per team", "run EPA per team", "rushing epa", "running epa" all have to be handled individually in the router. The LLM does one fuzzy thing (turn user question → `query`) and the executor does another fuzzy thing (turn `query` → metric). Two fuzzy layers is one too many.

Better: **make each metric its own MCP tool with structured parameters.** The LLM then only has to pick the right tool and fill in enum fields — something models (even small local ones) are good at — and MCP's JSON-Schema validation gives you free error handling.

Proposed tool catalog:

```python
@mcp.tool()
async def get_team_epa_per_play(
    seasons: list[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
    play_kind: Literal["all", "pass", "rush"] = "all",
    min_plays: int = 0,
) -> dict:
    """Rank teams by EPA per play. Rank 1 is best."""

@mcp.tool()
async def get_team_success_rate(
    seasons: list[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
    play_kind: Literal["all", "pass", "rush"] = "all",
) -> dict: ...

@mcp.tool()
async def get_team_epa_per_dropback(
    seasons: list[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
) -> dict: ...

@mcp.tool()
async def get_qb_leaderboard(
    seasons: list[int],
    season_type: Literal["REG", "POST"] = "REG",
    metric: Literal["epa_per_pass", "cpoe", "success_rate"] = "epa_per_pass",
    min_dropbacks: int = 100,
) -> dict: ...

@mcp.tool()
async def get_team_metric_by_situation(
    seasons: list[int],
    season_type: Literal["REG", "POST"] = "REG",
    side: Literal["offense", "defense"] = "offense",
    metric: Literal["epa_per_play", "success_rate"] = "epa_per_play",
    down: Literal["any", "1", "2", "3", "4"] = "any",
    red_zone: bool = False,
    two_minute: bool = False,
) -> dict: ...
```

This gives you concrete wins:

1. **The planner prompt becomes simpler.** You ask the model to emit a tool name + arguments, not free-text. The MCP client library can even do tool-use natively if you wire the model through an agent loop.
2. **"run epa per team" just works.** The model emits `get_team_epa_per_play(..., play_kind="rush")`. No regex in the router.
3. **"third down EPA for KC"** becomes one tool call with `down="3"` — previously you couldn't express this at all.
4. **You delete a lot of code.** The giant `execute(query)` router in `metrics_executor.py` goes away; each MCP tool is ~5 lines calling into a private method.
5. **Errors become structured.** If the model emits `side="defence"` (British spelling), MCP rejects it before your code runs, and the error surfaces back to the planner to retry.

Concretely, refactor `MetricsExecutor` so its public surface is the per-metric methods (drop the leading underscores), and delete `execute()`. Then each `@mcp.tool()` is:

```python
@mcp.tool()
async def get_team_epa_per_play(seasons, season_type="REG", side="offense", play_kind="all"):
    df = _load_pbp_cached(tuple(seasons), season_type)
    ex = MetricsExecutor(df)
    out_df, summary = ex.epa_per_play(side=side, kind=play_kind)
    return {"summary": summary, "rows": out_df.to_dicts()}
```

No session IDs. No natural-language routing. No `compute_metric`.

### Where does the LLM fit?
Two options:

- **Keep the current "planner emits JSON" approach but target the richer schema.** You rewrite `prompts/planner_system.txt` to describe each tool + its arguments; the LLM emits `{"tool": "get_team_epa_per_play", "args": {...}}`; Streamlit dispatches.
- **Use a real agent loop.** The `openai-agents` package you already depend on supports MCP tool use — you let the model call tools itself (possibly multiple times) instead of a rigid two-step plan → execute. The tradeoff is latency on an 8B local model, but for ≤2 tool calls it should still feel fine. This would also let follow-up questions ("now do the same for 2024") work without the user re-specifying everything.

---

## 5. Other recommended improvements

### 5a. Replace subprocess-based Ollama with HTTP
`subprocess.run(["ollama", "run", ...])` forks a process and reloads the model each call. Use the Ollama REST API (`http://localhost:11434/api/generate` or `/api/chat`) — the daemon keeps the model warm, it supports JSON mode natively, and you can stream tokens to Streamlit for a nicer UX:

```python
import httpx

def ollama_chat(prompt: str, model: str = "llama3.1:8b") -> str:
    r = httpx.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["response"]
```

### 5b. Cache PBP downloads
`nflreadpy` already caches by default on disk, but the in-process cost of re-reading + re-normalizing every question still adds up. Wrap with `functools.lru_cache` as shown above, or use Streamlit's `@st.cache_resource` (since a Polars DataFrame is large and mutable).

### 5c. Typed plan object
Use Pydantic for the planner output:
```python
class Plan(BaseModel):
    seasons: conlist(int, min_length=1)
    season_type: Literal["REG", "POST"] = "REG"
    tool: Literal["get_team_epa_per_play", "get_qb_leaderboard", ...]
    args: dict
```
`Plan.model_validate_json(raw)` replaces the fragile `index("{")` / `rindex("}")` parsing *and* gives you a single source of truth shared with the planner prompt.

### 5d. Tests
`tests/tests_executor.ipynb` is a notebook, which is hard to run in CI and has its own warning at the top saying it needs updating. Port the smoke tests to `pytest` files:

```
tests/
  test_metrics_executor.py
  test_planner.py
  conftest.py   # fixture that returns a small synthetic PBP frame
```

This also makes it possible to run the tests on a 20-row synthetic dataset so you're not hitting the network on every test.

### 5e. Logging instead of `st.error`/`print`
Right now error paths leak Ollama stderr into Streamlit. Use the `logging` module, route to both a log file and the Streamlit sidebar with a debug toggle.

### 5f. Small prompt quality wins
- The prompt claims *"Rank 1 is best. Rank 32 is the worst"* but the executor uses `method="dense"` ranking, which can produce fewer than 32 ranks when there are ties. Either switch to `method="ordinal"` or update the prompt to not promise 32.
- The `answering_style.md` file isn't actually being loaded anywhere in the production path — only `planner_system.txt` is. Either load both into the prompt or delete the orphan file.

### 5g. CLI smoke entrypoint
Add a tiny script so you can sanity-check without Streamlit:
```
python -m nfl_agent.cli "run EPA per team 2024"
```
Great for regression tests and for wiring into a scheduled task later.

---

## 6. Suggested file structure after cleanup

```
NFL_Agent/
├── .env.example             # template, no secret
├── .gitignore               # add __pycache__, .DS_Store, .ipynb_checkpoints
├── README.md                # NEW — setup, how to run, architecture
├── requirements.txt         # renamed from requirements_mcp.txt; add httpx, pydantic
├── prompts/
│   ├── planner_system.txt   # rewritten for tool-based planning
│   └── answering_style.md   # or delete
├── src/nfl_agent/
│   ├── __init__.py
│   ├── app.py               # from app_st_nfl_agent.py
│   ├── mcp_server.py        # from mcp_nfl_metrics_server.py
│   ├── metrics.py           # from metrics_executor.py, no execute() router
│   ├── planner.py           # ollama_chat + Plan pydantic model
│   └── pbp_cache.py         # lru_cache-wrapped loader
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   └── test_planner.py
└── archive/                 # old notebooks, kept for posterity
    ├── agent_app_w_mcp.ipynb
    ├── data_cleaning_nfl_agent.ipynb
    └── demo_for_qb_stats.ipynb
```

---

## 7. Prioritized punch list

If you only have an afternoon, do these in order:

1. **Rotate the OpenAI key** and delete it from `.env` (5 min)
2. **Stop lowercasing team names** — one-line fix in two places (10 min)
3. **Add `__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/` to `.gitignore`** (2 min)
4. **Fix the "run" alias bug** so "run EPA" routes to rushing (5 min)
5. **Move the three notebooks to `archive/`** and make `app_st_nfl_agent.py` the single source of truth (15 min)
6. **Decide on `nfl_data_py` vs `nflreadpy`** and purge the other (30 min)
7. **Refactor to per-metric MCP tools** (§4) — the biggest win, a few hours
8. **Switch Ollama to the HTTP API** with `format="json"` (30 min)
9. **Port notebook tests to `pytest`** (1 hour)
