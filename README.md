# NFL EPA Analytics Agent

A natural-language interface over nflfastR play-by-play data. Ask a question in
plain English, and a local LLM plans which metric to compute; the metric itself
is calculated deterministically in Polars and returned as a ranked table plus a
chart.

## Architecture

```
Streamlit UI  (app_st_nfl_agent.py)
      |
      |  question -> JSON plan   [local Ollama model + prompts/planner_system.txt]
      v
MCP server    (mcp_nfl_metrics_server.py)   <- loads + caches nflreadpy data
      |
      v
Metrics engine (metrics_executor.py)        <- all statistics live here
```

The split matters: **the language model only chooses parameters, it never
computes a number.** Every statistic is produced by `MetricsExecutor`, which is
pure, deterministic, and unit-tested. A hallucinated plan produces the wrong
*question*, never a wrong *answer* to the question it asked.

## Setup

```bash
uv sync                      # installs from pyproject.toml / uv.lock
ollama pull llama3.1:8b      # the planner model
uv run streamlit run app_st_nfl_agent.py
```

`pyproject.toml` + `uv.lock` are the source of truth for dependencies.
`requirements_mcp.txt` is a convenience export for pip-only environments.

## Running the tests

```bash
uv run pytest
```

The suite runs entirely on small synthetic frames — no network, no data
download. Most tests are regression tests pinned to specific bugs (see
`tests/test_metrics_executor.py`), so a failure names the behaviour that broke.

## Metrics

| Tool | Metric | Rank 1 means |
|---|---|---|
| `get_team_epa_per_play` | mean EPA per pass/rush play | offense: highest generated; defense: lowest allowed |
| `get_epa_per_dropback` | mean EPA per dropback | offense: highest; defense: lowest allowed |
| `get_success_rate` | fraction of plays with EPA > 0 | offense: highest; defense: **lowest** allowed |
| `get_qb_stats` | mean EPA per dropback, or CPOE | highest |
| `get_adjusted_ratings` | opponent-adjusted EPA per play | best net rating |
| `get_strength_of_schedule` | raw vs adjusted, and slate difficulty | best adjusted offense |
| `get_matchup_projection` | projected EPA/play for one game | n/a |

**Rank 1 is always best for the side requested.** This is worth stating
explicitly because it is easy to get backwards: for a defense, *lower* EPA
allowed and *lower* opponent success rate are better, so those rankings sort
ascending while offensive ones sort descending.

### What counts as a play

"EPA per play" means pass and rush plays only. Punts, field goals, kickoffs,
extra points, timeouts, and no-play rows are excluded, as are QB kneels and
spikes — they are clock management, not offensive attempts. Rows with a null
EPA are dropped, and the reported `plays` count reflects only the rows that
actually contributed to the mean.

### What counts as a dropback

Dropbacks are pass attempts **plus sacks and scrambles**. Sacks carry strongly
negative EPA and are substantially attributable to the quarterback, so
excluding them inflates every passer's efficiency — particularly sack-prone
ones. The engine prefers nflfastR's `passer` column over `passer_player_name`
precisely because the former is populated on sacks and scrambles.

QBs are grouped by season and player, not by team, so a mid-season trade
produces one row (`team` reads e.g. `NYJ/PHI`) rather than two samples that
each fall below the minimum-dropback threshold.

## Why these metrics use the mean, not the median

Every metric here is a **rate statistic**, and all of them are aggregated with
the mean. That is deliberate:

1. **EPA is additive.** `mean(EPA) × plays` equals total expected points added.
   The mean therefore has a direct, interpretable relationship to scoring. The
   median has no such relationship — you cannot add up medians to get anything
   meaningful.

2. **The skew is the signal, not noise.** Per-play EPA is heavily
   right-skewed: most plays are small negatives, and a handful of explosive
   plays carry large positive EPA. An offense that generates explosive plays is
   genuinely better, and the mean credits it. The median throws that away. In
   the test suite, `test_mean_rewards_explosive_offense_where_median_would_not`
   pins a case where an explosive offense has a strongly positive mean EPA but
   a *negative* median — ranking by median would put a conservative
   dink-and-dunk offense above it, which is simply wrong.

3. **Success rate is already a proportion.** `success_rate` is
   `mean(epa > 0)`. The median of a 0/1 indicator is just 0 or 1 and carries no
   information at all.

This also matches the nflfastR / public-analytics convention, so numbers here
are comparable to published figures.

**If you want distribution shape**, add it as a separate statistic — standard
deviation of EPA, explosive-play rate, or quantiles — rather than replacing the
measure of central tendency. Median only makes sense for describing a *typical
per-game* outcome robust to blowouts, which is a different question from
efficiency ranking.

## Opponent adjustment

Raw EPA per play conflates *how good a team is* with *who it happened to play*.
`adjusted_ratings.py` separates the two by fitting one ridge regression over
every play in the window:

```
epa = intercept + home_field·home + off_effect[posteam] + def_effect[defteam] + error
```

Because all 32 offenses and 32 defenses are solved simultaneously, each rating
is estimated holding the opponent constant. `off_adj` answers "how much EPA per
play would this offense generate against a league average defense on a neutral
field" — exactly the quantity you want when projecting a future matchup.

The ridge penalty does double duty. It makes the system identifiable (intercept
plus the offense indicators are otherwise perfectly collinear), and it shrinks
each team effect toward league average in proportion to how little evidence
supports it. That is regression to the mean applied automatically and
continuously: a team with 200 plays is pulled toward average much harder than
one with 1,000. The penalty is chosen by cross-validation **grouped on
`game_id`**, because plays within a game are highly correlated and splitting
them across folds would leak information and select too small a penalty.

### Does it actually help? Backtest results

Fit on weeks 1–N, then predict each team-game's EPA per play over the rest of
the season. Mean squared error across 2021–2024, lower is better:

| Train window | League average | Raw EPA | **Adjusted** | Gain vs raw |
|---|---|---|---|---|
| Weeks 1–4 | 0.03774 | 0.03974 | **0.03671** | **+7.6%** |
| Weeks 1–6 | 0.03823 | 0.03702 | **0.03574** | **+3.4%** |
| Weeks 1–9 | 0.03829 | 0.03539 | **0.03469** | **+2.0%** |
| Weeks 1–12 | 0.03968 | **0.03456** | 0.03464 | −0.3% |

Adjusted beat raw in **14 of 16** season/cutoff combinations. Two things are
worth reading off this table honestly:

- **Early in a season the adjustment matters a lot.** Through four weeks, raw
  EPA per play is *worse than simply guessing the league average* — schedules
  are so unbalanced that the raw numbers are actively misleading. The adjusted
  model is the only one that beats that baseline.
- **By late season it is a wash.** After twelve weeks everyone has played a
  broad slate, schedule effects have largely averaged out, and the adjustment
  buys nothing. Use it for early-season and mid-season questions; do not expect
  it to add much in week 15.

### Recency weighting: where it helps and where it doesn't

Plays can be weighted by age, `w = 0.5 ** (weeks_ago / half_life)`, so recent
football counts for more. Two backtests were run to decide when to use it, and
they gave opposite answers.

**Within a single season it does nothing.** Across 12 season/cutoff
combinations, every half-life tried (3, 5, 8, 12 weeks) came out slightly
*worse* than weighting all plays equally, winning at most 4 of 12, with mean
differences smaller than their own standard deviation. That is a null result
and it is treated as one — a knob that does not help should not be switched on.

**Across seasons it is essential.** Carrying last season forward, 15
season/cutoff combinations (2022–24, weeks 1-2 through 1-6):

| Strategy | Mean MSE | vs current-season-only | Beats no-decay |
|---|---|---|---|
| Current season only | 0.03572 | — | 11/15 |
| Both seasons, **no decay** | 0.03600 | +0.8% (worse) | — |
| Both, half-life 8w | 0.03469 | −2.9% | 14/15 |
| **Both, half-life 12w** | **0.03453** | **−3.3%** | **15/15** |

The headline is the second row: **adding last season without decay is worse
than ignoring last season entirely.** Stale data actively hurts unless it is
discounted. With a 12-week half-life it becomes a real gain instead.

So `half_life="auto"` (the default) applies a 12-week half-life when the window
spans more than one season, and no decay within a single season. Pass an
explicit number to override.

Selecting a half-life needs **forward-chaining** validation — fit on weeks
1..k, score on week k+1, walk forward — which `select_half_life()` implements.
Ordinary k-fold cannot do this job even when grouped by game: random folds ask
the model to predict plays from the same period it trained on, and
down-weighting data never helps there, so such a search always returns "no
decay" no matter what the truth is. `float("inf")` is in the search grid on
purpose so the procedure is able to conclude that decay does not help.

### Limitations

The model needs a **connected schedule graph**. If teams have faced only one or
two opponents, there is genuinely no way to distinguish a good offense from bad
opposing defenses, and the effects smear across both. A real NFL season is
comfortably connected after a few weeks; a single week is not, and ratings from
such a window should be treated as close to meaningless.

These are per-play efficiency ratings. They are **not point spreads, win
probabilities, or a betting model**, and they carry no adjustment for injuries,
weather, rest, or personnel changes.

### Other levers if you push further toward prediction

- **Stability.** Success rate and EPA per dropback are more stable
  week-over-week than raw EPA per play, so they generalise better as inputs.
- **Separate pass and rush models.** `kind="pass"` and `kind="rush"` fit
  independently; passing efficiency stabilises faster than rushing.

## Project layout

```
app_st_nfl_agent.py        Streamlit UI, planner, rendering
mcp_nfl_metrics_server.py  MCP tool surface; loads and caches nflreadpy data
metrics_executor.py        Descriptive statistics; pure and unit-tested
adjusted_ratings.py        Ridge model for opponent-adjusted ratings
prompts/                   Planner system prompt and answering-style rules
tests/                     pytest suite over synthetic frames
archive/                   Exploratory notebooks, kept for reference
```

## Notes and known limitations

- The planner runs on a small local model. Every field it emits is validated
  and given a safe fallback in `_normalize_plan`; it is never trusted directly.
- `prompts/answering_style.md` documents intended answer formatting but is not
  currently wired into the pipeline.
- Play-by-play data is cached per `(seasons, season_type)` for the life of the
  server process.
- Opponent adjustment covers home/away but not weather, injuries, rest or
  personnel changes.
- The ridge model is refit per `(seasons, season_type, kind)` and cached for
  the life of the server process.
