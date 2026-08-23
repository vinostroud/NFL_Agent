"""
Opponent-adjusted team ratings.

Raw EPA per play conflates two things: how good a team is, and who it happened
to play. A team that faced six elite defenses looks worse than it is; a team
that feasted on the league's worst looks better. Strength of schedule is
usually the single largest source of bias in team rankings, especially early in
a season when schedules are most unbalanced.

This module separates the two by fitting one ridge regression over every play:

    epa_i = intercept
          + home_field * home_i
          + off_effect[posteam_i]
          + def_effect[defteam_i]
          + error_i

Solving for all 32 offenses and 32 defenses simultaneously means each team's
rating is estimated *holding the opponent constant*. The offensive rating
answers "how much EPA per play would this offense generate against a league
average defense on a neutral field", which is exactly the quantity you want
when projecting a future matchup against a known opponent.

Why ridge rather than ordinary least squares
--------------------------------------------
The penalty term does double duty:

1. **Identifiability.** Intercept plus the offense indicators are perfectly
   collinear, so OLS has no unique solution. Any positive penalty on the team
   effects makes the system solvable.
2. **Shrinkage.** The penalty pulls team effects toward the league mean in
   proportion to how little evidence supports them. This is regression to the
   mean applied automatically and continuously -- a team with 200 plays is
   pulled toward average much harder than one with 1,000. That is precisely the
   small-sample correction that raw per-play averages lack.

The intercept and home-field term are left unpenalised; only team effects are
shrunk.

The penalty strength is chosen by cross-validation grouped on game_id, because
plays within a game are highly correlated -- splitting them across folds would
leak information and select an over-optimistic (too small) penalty.

Requirement: schedule connectivity
----------------------------------
Separating offensive from defensive quality requires a well-connected schedule
graph. If every team faced only one or two opponents, there is genuinely no way
to tell a good offence from a set of bad opposing defences, and the model will
smear the effect across both -- not a bug, but an underdetermined problem. A
real NFL season is comfortably connected after a handful of weeks; very short
windows (a single week) are not, and ratings from them should be treated as
close to meaningless regardless of what the penalty term does.

Scope
-----
These are per-play efficiency ratings. They are not point spreads, win
probabilities, or a betting model, and they carry no adjustment for injuries,
weather, rest, or personnel changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from metrics_executor import PLAY_KIND, MetricsExecutor

Kind = Literal["all", "pass", "rush"]

# Penalty grid searched by cross-validation. Spans "almost unpenalised" to
# "heavily shrunk toward league average".
DEFAULT_ALPHA_GRID: Tuple[float, ...] = (
    1.0,
    10.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
)

# Calendar weeks between the same week of consecutive seasons. Used so that a
# play from last season is aged by roughly a year rather than by one week.
SEASON_SPAN_WEEKS = 52.0

# Half-lives (in weeks) searched by forward-chaining selection. `inf` means no
# decay at all, and is included so the search can conclude that recency
# weighting does not help.
DEFAULT_HALF_LIFE_GRID: Tuple[float, ...] = (2.0, 4.0, 6.0, 9.0, 14.0, 24.0, float("inf"))

# Half-life applied by half_life="auto" when the window spans more than one
# season. Chosen by backtest over 2022-24 (see README): across 15 season/cutoff
# combinations it beat "no decay" 15 times out of 15 and cut error ~3% versus
# using the current season alone. Within a single season, decay showed no
# reliable effect, so "auto" leaves single-season fits unweighted.
CROSS_SEASON_HALF_LIFE = 12.0


@dataclass(frozen=True)
class CVResult:
    alpha: float
    mse: float


class AdjustedRatings:
    """
    Opponent-adjusted offensive and defensive ratings.

    Usage:
        model = AdjustedRatings.fit(pbp_df, kind="all")
        ratings_df, summary = model.ratings()
        matchup_df, summary = model.predict_matchup("SF", "DAL")

    Ratings are in EPA per play. Offensive ratings are better when higher;
    defensive ratings are better when lower (a negative defensive rating means
    the defense suppresses EPA relative to league average).
    """

    def __init__(
        self,
        teams: List[str],
        intercept: float,
        home_field: float,
        off_effects: np.ndarray,
        def_effects: np.ndarray,
        off_plays: np.ndarray,
        def_plays: np.ndarray,
        alpha: float,
        cv_results: Sequence[CVResult],
        n_plays: int,
        kind: Kind,
        half_life: Optional[float] = None,
        effective_n: Optional[float] = None,
    ) -> None:
        self.teams = teams
        self.intercept = intercept
        self.home_field = home_field
        self.off_effects = off_effects
        self.def_effects = def_effects
        self.off_plays = off_plays
        self.def_plays = def_plays
        self.alpha = alpha
        self.cv_results = list(cv_results)
        self.n_plays = n_plays
        self.kind = kind
        self.half_life = half_life
        self.effective_n = float(effective_n) if effective_n is not None else float(n_plays)

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _build_design(
        df: pl.DataFrame, teams: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (X, y, groups).

        X columns: [intercept, home, off_0..off_31, def_0..def_31]
        """
        n_teams = len(teams)
        index = {team: i for i, team in enumerate(teams)}

        off_idx = np.array([index[t] for t in df["posteam"].to_list()], dtype=np.int64)
        def_idx = np.array([index[t] for t in df["defteam"].to_list()], dtype=np.int64)
        y = df["epa"].to_numpy().astype(np.float64)

        n = df.height
        X = np.zeros((n, 2 + 2 * n_teams), dtype=np.float64)
        X[:, 0] = 1.0

        if "home_team" in df.columns:
            home = (
                df.select(
                    (pl.col("posteam") == pl.col("home_team")).cast(pl.Float64)
                )
                .to_series()
                .to_numpy()
            )
            X[:, 1] = home

        rows = np.arange(n)
        X[rows, 2 + off_idx] = 1.0
        X[rows, 2 + n_teams + def_idx] = 1.0

        if "game_id" in df.columns:
            groups = (
                df.select(
                    pl.col("game_id").cast(pl.Utf8).cast(pl.Categorical).to_physical()
                )
                .to_series()
                .to_numpy()
                .astype(np.int64)
            )
        else:
            # No game identifier: fall back to treating each play independently.
            groups = np.arange(n, dtype=np.int64)

        return X, y, groups

    @staticmethod
    def _time_weights(
        df: pl.DataFrame,
        half_life: Optional[float],
        season_span_weeks: float = SEASON_SPAN_WEEKS,
    ) -> np.ndarray:
        """
        Exponential decay weights: w = 0.5 ** (weeks_ago / half_life).

        Age is measured back from the most recent week present in the data, so
        the fitted ratings describe the team as it is *now* rather than
        averaged over the whole window. Season boundaries add
        `season_span_weeks` of age, so last season's week 10 sits roughly a
        calendar year behind this season's week 10 rather than one week behind.

        Returns all-ones when half_life is None or infinite.
        """
        n = df.height
        if half_life is None or not np.isfinite(half_life):
            return np.ones(n, dtype=np.float64)
        if half_life <= 0:
            raise ValueError("half_life must be positive (or None for no decay).")
        if "week" not in df.columns:
            raise ValueError(
                "Recency weighting needs a `week` column; pass half_life=None to "
                "weight every play equally."
            )

        week = df["week"].cast(pl.Float64).to_numpy()
        if "season" in df.columns:
            season = df["season"].cast(pl.Float64).to_numpy()
        else:
            season = np.zeros(n, dtype=np.float64)

        absolute_week = season * season_span_weeks + week
        weeks_ago = absolute_week.max() - absolute_week
        return np.power(0.5, weeks_ago / float(half_life))

    @staticmethod
    def _resolve_half_life(
        df: pl.DataFrame, half_life: Optional[float] | str
    ) -> Optional[float]:
        """
        Resolve half_life="auto".

        Recency weighting earns its keep across seasons, not within one. A
        backtest over 2022-24 found no reliable within-season effect, but found
        that carrying last season forward *without* decay is worse than ignoring
        it entirely, while carrying it forward with a ~12 week half-life beats
        both. So "auto" decays only when the window spans multiple seasons.
        """
        if half_life != "auto":
            return half_life  # type: ignore[return-value]

        if "season" not in df.columns:
            return None
        n_seasons = df["season"].n_unique()
        return CROSS_SEASON_HALF_LIFE if n_seasons > 1 else None

    @staticmethod
    def _effective_n(weights: np.ndarray) -> float:
        """
        Kish effective sample size. With equal weights this is just n; with
        aggressive decay it reports how much data is really informing the fit.
        """
        s = float(weights.sum())
        ss = float((weights**2).sum())
        return (s * s / ss) if ss > 0 else 0.0

    @staticmethod
    def _penalty_matrix(n_features: int) -> np.ndarray:
        """
        Diagonal penalty: intercept (col 0) and home field (col 1) unpenalised.
        """
        d = np.ones(n_features, dtype=np.float64)
        d[0] = 0.0
        d[1] = 0.0
        return d

    @staticmethod
    def _solve(XtX: np.ndarray, Xty: np.ndarray, alpha: float, d: np.ndarray) -> np.ndarray:
        A = XtX + alpha * np.diag(d)
        try:
            return np.linalg.solve(A, Xty)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, Xty, rcond=None)[0]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        df: pl.DataFrame,
        kind: Kind = "all",
        alpha: Optional[float] = None,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        n_folds: int = 5,
        min_plays: int = 50,
        half_life: Optional[float] | str = "auto",
    ) -> "AdjustedRatings":
        """
        Fit opponent-adjusted ratings.

        alpha: ridge penalty. If None, chosen by game-grouped cross-validation
               over `alpha_grid`.
        min_plays: teams with fewer plays on either side of the ball are
               rejected, since their effects would be almost pure shrinkage.
        half_life: recency half-life in weeks. A play `half_life` weeks old
               counts half as much as the most recent play. None weights every
               play equally. The default, "auto", applies
               CROSS_SEASON_HALF_LIFE when the window spans more than one
               season and no decay within a single season -- which is what the
               backtest supports.

               Note that `alpha` is still selected by game-grouped CV, which
               deliberately cannot be used to choose `half_life`: random folds
               predict *contemporaneous* plays, and down-weighting data never
               helps there, so such a search would always return "no decay".
               Use `select_half_life()`, which validates forward in time.
        """
        executor = MetricsExecutor(df)
        work = executor.df

        for col in ("posteam", "defteam", "epa"):
            if col not in work.columns:
                raise ValueError(f"Missing required column `{col}` for adjusted ratings.")

        mask = executor._offensive_play_expr() & executor._filter_kind(kind)
        work = work.filter(
            mask & pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
        )

        if work.is_empty():
            raise ValueError(f"No plays available for kind={kind!r}.")

        teams = sorted(
            set(work["posteam"].unique().to_list())
            | set(work["defteam"].unique().to_list())
        )
        if len(teams) < 2:
            raise ValueError("Need at least two teams to estimate adjusted ratings.")

        off_counts = (
            work.group_by("posteam").len().rename({"posteam": "team", "len": "n"})
        )
        def_counts = (
            work.group_by("defteam").len().rename({"defteam": "team", "len": "n"})
        )
        off_plays = np.array(
            [
                off_counts.filter(pl.col("team") == t)["n"].to_list()[0]
                if off_counts.filter(pl.col("team") == t).height
                else 0
                for t in teams
            ],
            dtype=np.int64,
        )
        def_plays = np.array(
            [
                def_counts.filter(pl.col("team") == t)["n"].to_list()[0]
                if def_counts.filter(pl.col("team") == t).height
                else 0
                for t in teams
            ],
            dtype=np.int64,
        )

        thin = [
            teams[i]
            for i in range(len(teams))
            if off_plays[i] < min_plays or def_plays[i] < min_plays
        ]
        if thin:
            raise ValueError(
                f"These teams have fewer than {min_plays} plays and cannot be "
                f"rated reliably: {thin}. Widen the season range or lower "
                "min_plays."
            )

        X, y, groups = cls._build_design(work, teams)
        d = cls._penalty_matrix(X.shape[1])

        # Weighted least squares by row scaling: minimising ||sqrt(W)(y - Xb)||^2
        # is exactly minimising the W-weighted squared error.
        resolved_half_life = cls._resolve_half_life(work, half_life)
        weights = cls._time_weights(work, resolved_half_life)
        sw = np.sqrt(weights)
        Xw = X * sw[:, None]
        yw = y * sw

        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw

        cv_results: List[CVResult] = []
        if alpha is None:
            alpha, cv_results = cls._choose_alpha(
                X, y, Xw, yw, groups, XtX, Xty, d, alpha_grid, n_folds
            )

        beta = cls._solve(XtX, Xty, alpha, d)

        n_teams = len(teams)
        intercept = float(beta[0])
        home_field = float(beta[1])
        off = beta[2 : 2 + n_teams].copy()
        dfn = beta[2 + n_teams :].copy()

        # Re-centre so effects are exactly deviations from league average. This
        # leaves every fitted value unchanged but makes ratings interpretable.
        off_mean, def_mean = float(off.mean()), float(dfn.mean())
        off -= off_mean
        dfn -= def_mean
        intercept += off_mean + def_mean

        return cls(
            teams=teams,
            intercept=intercept,
            home_field=home_field,
            off_effects=off,
            def_effects=dfn,
            off_plays=off_plays,
            def_plays=def_plays,
            alpha=float(alpha),
            cv_results=cv_results,
            n_plays=int(work.height),
            kind=kind,
            half_life=resolved_half_life,
            effective_n=cls._effective_n(weights),
        )

    @classmethod
    def _choose_alpha(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        Xw: np.ndarray,
        yw: np.ndarray,
        groups: np.ndarray,
        XtX: np.ndarray,
        Xty: np.ndarray,
        d: np.ndarray,
        alpha_grid: Sequence[float],
        n_folds: int,
    ) -> Tuple[float, List[CVResult]]:
        """
        Game-grouped k-fold CV.

        Every play from a given game lands in the same fold. Plays within a
        game share weather, game script, personnel and officiating, so
        splitting them would leak and select too small a penalty.

        Fold training statistics are obtained by subtraction
        (XtX_train = XtX_full - XtX_fold), which is exact and avoids rebuilding
        the design matrix for every fold.
        """
        unique_games = np.unique(groups)
        if len(unique_games) < n_folds:
            n_folds = max(2, len(unique_games))

        # Deterministic assignment of games to folds.
        fold_of_game = {g: i % n_folds for i, g in enumerate(unique_games)}
        fold_ids = np.array([fold_of_game[g] for g in groups], dtype=np.int64)

        totals = np.zeros(len(alpha_grid), dtype=np.float64)
        counts = 0

        for fold in range(n_folds):
            test_mask = fold_ids == fold
            if not test_mask.any() or test_mask.all():
                continue

            # Training statistics come from the WEIGHTED matrices (the model is
            # fitted weighted); the held-out error is measured unweighted,
            # because every future play counts the same when predicting.
            Xw_te, yw_te = Xw[test_mask], yw[test_mask]
            XtX_tr = XtX - Xw_te.T @ Xw_te
            Xty_tr = Xty - Xw_te.T @ yw_te

            X_te, y_te = X[test_mask], y[test_mask]

            for j, a in enumerate(alpha_grid):
                beta = cls._solve(XtX_tr, Xty_tr, a, d)
                resid = y_te - X_te @ beta
                totals[j] += float(resid @ resid)
            counts += len(y_te)

        if counts == 0:
            return float(alpha_grid[len(alpha_grid) // 2]), []

        mses = totals / counts
        results = [CVResult(alpha=float(a), mse=float(m)) for a, m in zip(alpha_grid, mses)]
        best = min(results, key=lambda r: r.mse)
        return best.alpha, results

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def ratings(self) -> Tuple[pl.DataFrame, str]:
        """
        Per-team adjusted offensive and defensive ratings, in EPA per play.

        off_adj: EPA/play this offense would generate against a league average
                 defence on a neutral field. Higher is better.
        def_adj: EPA/play this defence would allow to a league average offence.
                 Lower (more negative) is better.
        net_adj: off_adj - def_adj. Higher is better.
        """
        net = self.off_effects - self.def_effects

        df = pl.DataFrame(
            {
                "team": self.teams,
                "off_adj": self.off_effects,
                "def_adj": self.def_effects,
                "net_adj": net,
                "off_plays": self.off_plays,
                "def_plays": self.def_plays,
            }
        )

        df = df.with_columns(
            [
                pl.col("off_adj").rank(method="min", descending=True).cast(pl.Int64).alias("off_rank"),
                pl.col("def_adj").rank(method="min", descending=False).cast(pl.Int64).alias("def_rank"),
                pl.col("net_adj").rank(method="min", descending=True).cast(pl.Int64).alias("rank"),
            ]
        ).sort("rank")

        df = df.select(
            [
                "rank",
                "team",
                "net_adj",
                "off_adj",
                "off_rank",
                "def_adj",
                "def_rank",
                "off_plays",
                "def_plays",
            ]
        )

        kind_label = "" if self.kind == "all" else f" {self.kind}"
        if self.half_life is None or not np.isfinite(self.half_life):
            recency = "All plays weighted equally."
        else:
            recency = (
                f"Recency-weighted with a {self.half_life:g}-week half-life "
                f"(effective sample {self.effective_n:,.0f} plays)."
            )

        summary = (
            f"Opponent-adjusted{kind_label} team ratings from a ridge model fitted "
            f"on {self.n_plays:,} plays (penalty alpha={self.alpha:g}, chosen by "
            f"game-grouped cross-validation). {recency} "
            "Ratings are EPA per play against a league average opponent on a "
            "neutral field: off_adj higher is better, def_adj lower is better, "
            "rank is by net_adj. "
            f"Estimated home-field advantage: {self.home_field:+.4f} EPA per play."
        )
        return df, summary

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False,
    ) -> Tuple[pl.DataFrame, str]:
        """
        Project EPA per play for both offences in a specific matchup.

        This is the payoff of opponent adjustment: the projection accounts for
        who each offence is actually facing, rather than assuming every
        opponent is average.
        """
        index = {t: i for i, t in enumerate(self.teams)}
        for team in (home_team, away_team):
            if team not in index:
                raise ValueError(
                    f"Unknown team {team!r}. Known teams: {', '.join(self.teams)}"
                )

        h, a = index[home_team], index[away_team]
        hfa = 0.0 if neutral_site else self.home_field

        home_off = self.intercept + hfa + self.off_effects[h] + self.def_effects[a]
        away_off = self.intercept + self.off_effects[a] + self.def_effects[h]

        df = pl.DataFrame(
            {
                "offense": [home_team, away_team],
                "defense": [away_team, home_team],
                "site": ["home", "away"] if not neutral_site else ["neutral", "neutral"],
                "proj_epa_per_play": [float(home_off), float(away_off)],
                "off_adj": [
                    float(self.off_effects[h]),
                    float(self.off_effects[a]),
                ],
                "opp_def_adj": [
                    float(self.def_effects[a]),
                    float(self.def_effects[h]),
                ],
            }
        )

        edge = float(home_off - away_off)
        favored = home_team if edge > 0 else away_team
        summary = (
            f"Projected EPA per play for {away_team} at {home_team}"
            + (" (neutral site)" if neutral_site else "")
            + f". Both projections use opponent-adjusted ratings, so each offence "
            "is evaluated against the specific defence it faces. Projected edge: "
            f"{favored} by {abs(edge):.4f} EPA per play. "
            "This is a per-play efficiency projection, not a point spread or a "
            "win probability."
        )
        return df, summary


@dataclass(frozen=True)
class HalfLifeResult:
    half_life: float
    mse: float
    n_eval: int


def select_half_life(
    df: pl.DataFrame,
    kind: Kind = "all",
    grid: Sequence[float] = DEFAULT_HALF_LIFE_GRID,
    min_train_weeks: int = 4,
    alpha: Optional[float] = None,
    min_plays: int = 20,
) -> Tuple[float, List[HalfLifeResult]]:
    """
    Choose a recency half-life by forward-chaining validation.

    For each candidate half-life, walk forward through the season: fit on
    weeks 1..k and score the model on week k+1, for every k from
    `min_train_weeks` to the second-to-last week. The winner is the half-life
    with the lowest total squared error on those held-out next weeks.

    This is the only honest way to tune recency. Ordinary k-fold -- even
    grouped by game -- asks the model to predict plays drawn from the *same*
    period it trained on, and down-weighting data can never help with that. Such
    a search always returns "no decay", regardless of whether decay would help
    for the thing we actually care about, which is predicting the future.

    `float("inf")` is in the default grid on purpose, so the procedure is able
    to conclude that recency weighting does not help.

    Returns (best_half_life, per-candidate results).
    """
    if "week" not in df.columns:
        raise ValueError("Forward-chaining selection requires a `week` column.")

    executor = MetricsExecutor(df)
    work = executor.df
    work = work.filter(
        executor._offensive_play_expr()
        & executor._filter_kind(kind)
        & pl.col("posteam").is_not_null()
        & pl.col("defteam").is_not_null()
    )
    if work.is_empty():
        raise ValueError(f"No plays available for kind={kind!r}.")

    weeks = sorted(work["week"].unique().to_list())
    cutoffs = [w for w in weeks[:-1] if w >= min_train_weeks]
    if not cutoffs:
        raise ValueError(
            f"Need more than {min_train_weeks} weeks of data to validate forward."
        )

    totals = {float(h): 0.0 for h in grid}
    counts = {float(h): 0 for h in grid}

    for cutoff in cutoffs:
        train = work.filter(pl.col("week") <= cutoff)
        test = work.filter(pl.col("week") == cutoff + 1)
        if test.is_empty():
            continue

        for h in grid:
            h = float(h)
            try:
                model = AdjustedRatings.fit(
                    train,
                    kind=kind,
                    alpha=alpha,
                    min_plays=min_plays,
                    half_life=None if not np.isfinite(h) else h,
                )
            except ValueError:
                # Not enough data yet for this cutoff; skip it for every
                # candidate equally by leaving the totals untouched.
                continue

            index = {t: i for i, t in enumerate(model.teams)}
            keep = test.filter(
                pl.col("posteam").is_in(model.teams) & pl.col("defteam").is_in(model.teams)
            )
            if keep.is_empty():
                continue

            off_i = np.array([index[t] for t in keep["posteam"].to_list()])
            def_i = np.array([index[t] for t in keep["defteam"].to_list()])
            home = (
                keep.select((pl.col("posteam") == pl.col("home_team")).cast(pl.Float64))
                .to_series()
                .to_numpy()
                if "home_team" in keep.columns
                else np.zeros(keep.height)
            )
            pred = (
                model.intercept
                + model.home_field * home
                + model.off_effects[off_i]
                + model.def_effects[def_i]
            )
            actual = keep["epa"].to_numpy()
            resid = actual - pred
            totals[h] += float(resid @ resid)
            counts[h] += len(actual)

    results = [
        HalfLifeResult(
            half_life=h,
            mse=(totals[h] / counts[h]) if counts[h] else float("inf"),
            n_eval=counts[h],
        )
        for h in sorted(totals, key=lambda x: (np.isinf(x), x))
    ]
    best = min(results, key=lambda r: r.mse)
    return best.half_life, results


def schedule_report(df: pl.DataFrame, model: AdjustedRatings) -> Tuple[pl.DataFrame, str]:
    """
    Compare raw and adjusted offensive ratings, and show the schedule faced.

    `opp_def_faced` is the play-weighted average adjusted defensive rating a
    team's offence lined up against. Positive means it faced defences that give
    up more EPA than average -- an easy slate. `rank_change` is how many places
    a team moves once that is accounted for.
    """
    executor = MetricsExecutor(df)
    work = executor.df
    mask = executor._offensive_play_expr() & executor._filter_kind(model.kind)
    work = work.filter(
        mask & pl.col("posteam").is_not_null() & pl.col("defteam").is_not_null()
    )

    def_lookup = pl.DataFrame(
        {"defteam": model.teams, "_opp_def": model.def_effects}
    )
    work = work.join(def_lookup, on="defteam", how="inner")

    raw = work.group_by("posteam").agg(
        [
            pl.col("epa").mean().alias("raw_epa"),
            pl.col("_opp_def").mean().alias("opp_def_faced"),
            pl.len().alias("plays"),
        ]
    ).rename({"posteam": "team"})

    adj = pl.DataFrame({"team": model.teams, "off_adj": model.off_effects})

    out = (
        raw.join(adj, on="team", how="inner")
        .with_columns(
            [
                pl.col("raw_epa").rank(method="min", descending=True).cast(pl.Int64).alias("raw_rank"),
                pl.col("off_adj").rank(method="min", descending=True).cast(pl.Int64).alias("adj_rank"),
            ]
        )
        .with_columns((pl.col("raw_rank") - pl.col("adj_rank")).alias("rank_change"))
        .sort("adj_rank")
        .select(
            [
                "adj_rank",
                "team",
                "off_adj",
                "raw_epa",
                "raw_rank",
                "rank_change",
                "opp_def_faced",
                "plays",
            ]
        )
    )

    biggest = out.sort("rank_change", descending=True).head(1)
    mover = biggest["team"][0] if biggest.height else "n/a"
    delta = int(biggest["rank_change"][0]) if biggest.height else 0

    summary = (
        "Raw versus opponent-adjusted offensive EPA per play. `opp_def_faced` is "
        "the average adjusted rating of the defences a team faced: positive means "
        "an easier slate, which inflates raw numbers. `rank_change` is places "
        f"gained once schedule is accounted for. Biggest riser: {mover} "
        f"({delta:+d} places)."
    )
    return out, summary
