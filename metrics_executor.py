from __future__ import annotations

from typing import Literal, Tuple

import polars as pl

Side = Literal["offense", "defense"]
Kind = Literal["all", "pass", "rush"]

# Internal column derived from the raw feed. We deliberately do NOT overwrite
# nflfastR's own `play_type`, so the source data stays inspectable.
PLAY_KIND = "play_kind"


class MetricsExecutor:
    """
    Core metrics engine for nflfastR-style play-by-play data.

    Usage:
        executor = MetricsExecutor(df)
        out_df, summary = executor.epa_per_play(side="offense", kind="pass")

    Aggregation policy
    ------------------
    Every metric here is a *rate* statistic and is aggregated with the MEAN,
    which is the nflfastR/industry standard. This is deliberate, not an
    oversight:

      * EPA is additive. mean(EPA) * plays == total expected points added, so
        the mean has a direct, interpretable relationship to scoring. No such
        relationship exists for the median.
      * Per-play EPA is heavily right-skewed: most plays are small negatives
        and a handful of explosive plays carry large positive EPA. That tail is
        the signal, not noise -- an offense that generates explosive plays is
        genuinely better. The median discards exactly that information and
        would rank a conservative, low-variance offense above an explosive one.
      * success_rate is mean(epa > 0), i.e. a proportion. A median of a 0/1
        indicator is just 0 or 1 and carries no information.

    Do not swap these to median. If you want to describe the *shape* of a
    team's EPA distribution, add an explicit dispersion metric (standard
    deviation, explosive-play rate, quantiles) as a separate statistic rather
    than changing the measure of central tendency.

    Expected input columns (missing columns raise a clear error rather than
    silently changing the answer):
        - season, season_type
        - posteam, defteam
        - epa (required for every metric)
        - pass / rush flags and/or play_type
        - qb_kneel, qb_spike (excluded from offensive plays when present)
        - qb_dropback, sack, qb_scramble, pass_attempt (QB metrics)
        - passer / passer_player_name, cpoe (QB metrics)
    """

    # ------------------------------------------------------------------
    # Initialization & normalization
    # ------------------------------------------------------------------

    def __init__(self, df: pl.DataFrame) -> None:
        # Work on a copy so we don't mutate upstream data.
        self.df = df.clone()

        self._normalize_play_kind()
        self._ensure_success_flag()

    def _normalize_play_kind(self) -> None:
        """
        Derive a clean `play_kind` column with values "pass", "rush" or null.

        Prefers nflfastR's binary `pass` / `rush` indicators, falling back to
        the textual `play_type` column.

        NOTE: every branch uses `pl.lit(...)`. A bare string inside `then()` is
        parsed by Polars as a COLUMN REFERENCE, and nflfastR play-by-play data
        contains columns literally named `pass` and `rush` -- so `then("pass")`
        silently yields the value of the `pass` column (1/0) instead of the
        string "pass".
        """
        df = self.df

        has_flag_pass = "pass" in df.columns
        has_flag_rush = "rush" in df.columns
        has_pt = "play_type" in df.columns

        if not (has_flag_pass or has_flag_rush or has_pt):
            # Nothing to derive from; leave the column absent so that any
            # pass/rush request fails loudly in _filter_kind().
            return

        expr = pl.lit(None, dtype=pl.Utf8)

        # Textual play_type first, so the more reliable binary flags below can
        # override it.
        if has_pt:
            pt = pl.col("play_type").cast(pl.Utf8).str.to_lowercase()
            expr = (
                pl.when(pt.str.contains("pass"))
                .then(pl.lit("pass"))
                .when(pt.str.contains("run") | pt.str.contains("rush"))
                .then(pl.lit("rush"))
                .otherwise(expr)
            )

        # Binary flags are authoritative in nflfastR: they correctly classify
        # scrambles as rushes, sacks as passes, and plays negated by penalty.
        if has_flag_pass:
            expr = (
                pl.when(pl.col("pass").cast(pl.Int64, strict=False) == 1)
                .then(pl.lit("pass"))
                .otherwise(expr)
            )
        if has_flag_rush:
            expr = (
                pl.when(pl.col("rush").cast(pl.Int64, strict=False) == 1)
                .then(pl.lit("rush"))
                .otherwise(expr)
            )

        self.df = self.df.with_columns(expr.alias(PLAY_KIND))

    def _ensure_success_flag(self) -> None:
        """
        Ensure a `success` column exists, defined (as in nflfastR) as epa > 0.
        """
        if "success" in self.df.columns or "epa" not in self.df.columns:
            return

        self.df = self.df.with_columns((pl.col("epa") > 0).alias("success"))

    # ------------------------------------------------------------------
    # Helpers / filters
    # ------------------------------------------------------------------

    @staticmethod
    def _team_col_for_side(side: Side) -> str:
        return "posteam" if side == "offense" else "defteam"

    def _require(self, *cols: str) -> None:
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {missing}.")

    def _offensive_play_expr(self) -> pl.Expr:
        """
        Restrict to genuine offensive scrimmage plays.

        Without this, "EPA per play" silently averages in punts, field goals,
        kickoffs, extra points, timeouts and no-play rows -- none of which
        belong in an offensive efficiency metric.
        """
        expr = pl.col("epa").is_not_null()

        if PLAY_KIND in self.df.columns:
            expr = expr & pl.col(PLAY_KIND).is_not_null()

        # Kneels and spikes are clock management, not offensive attempts.
        for col in ("qb_kneel", "qb_spike"):
            if col in self.df.columns:
                expr = expr & (
                    pl.col(col).cast(pl.Int64, strict=False).fill_null(0) != 1
                )

        return expr

    def _filter_kind(self, kind: Kind) -> pl.Expr:
        """
        Expression filtering to a kind of play. Raises rather than silently
        returning every play when the data cannot support the request.
        """
        if kind == "all":
            return pl.lit(True)

        if PLAY_KIND not in self.df.columns:
            raise ValueError(
                f"Cannot filter to kind={kind!r}: no `pass`/`rush` flags or "
                "`play_type` column available to classify plays."
            )

        return pl.col(PLAY_KIND) == kind

    def _dropback_expr(self) -> pl.Expr:
        """
        Expression defining QB dropbacks: pass attempts + sacks + scrambles.

        Sacks matter a great deal -- they carry strongly negative EPA and are
        substantially attributable to the quarterback. Excluding them inflates
        every passer's efficiency.
        """
        if "qb_dropback" in self.df.columns:
            return pl.col("qb_dropback").cast(pl.Int64, strict=False) == 1
        if "dropback" in self.df.columns:
            return pl.col("dropback").cast(pl.Int64, strict=False) == 1

        # Reconstruct from components when the flag is absent.
        parts = []
        for col in ("pass_attempt", "sack", "qb_scramble"):
            if col in self.df.columns:
                parts.append(pl.col(col).cast(pl.Int64, strict=False) == 1)

        if not parts:
            raise ValueError(
                "Cannot identify dropbacks: none of `qb_dropback`, `dropback`, "
                "`pass_attempt`, `sack` or `qb_scramble` are present."
            )

        expr = parts[0]
        for p in parts[1:]:
            expr = expr | p
        return expr

    def _passer_col(self) -> str:
        """
        Column identifying the dropback player.

        nflfastR's `passer` is populated on sacks and scrambles;
        `passer_player_name` is only populated on actual pass attempts. Prefer
        the former so sacks are attributed rather than dropped.
        """
        for col in ("passer", "passer_player_name"):
            if col in self.df.columns:
                return col
        raise ValueError(
            "Missing a passer column (`passer` or `passer_player_name`); "
            "cannot compute QB metrics."
        )

    @staticmethod
    def _rank(
        df: pl.DataFrame, metric_col: str, higher_is_better: bool
    ) -> pl.DataFrame:
        """
        Attach a competition rank (1 = best; ties share the better rank).
        """
        return df.with_columns(
            pl.col(metric_col)
            .rank(method="min", descending=higher_is_better)
            .cast(pl.Int64)
            .alias("rank")
        ).sort(["rank", metric_col])

    # ------------------------------------------------------------------
    # Team-level metrics
    # ------------------------------------------------------------------

    def epa_per_play(
        self,
        side: Side = "offense",
        kind: Kind = "all",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Team-level mean EPA per offensive play.

        Offense: rank 1 = highest EPA generated.
        Defense: rank 1 = lowest EPA allowed.
        """
        self._require("epa")
        team_col = self._team_col_for_side(side)
        self._require(team_col)

        filtered = self.df.filter(
            self._offensive_play_expr()
            & self._filter_kind(kind)
            & pl.col(team_col).is_not_null()
        )

        grouped = filtered.group_by(team_col).agg(
            [
                pl.len().alias("plays"),
                pl.col("epa").mean().alias("epa_per_play"),
            ]
        )

        higher_is_better = side == "offense"
        ranked = self._rank(grouped, "epa_per_play", higher_is_better)

        out_df = ranked.select(
            ["rank", pl.col(team_col).alias("team"), "epa_per_play", "plays"]
        )

        kind_label = "" if kind == "all" else f" {kind}"
        if side == "offense":
            summary = (
                f"Team-level offensive{kind_label} EPA per play (mean). "
                "Rank 1 is the highest (best) EPA/play. "
                "Limited to pass and rush plays with a valid EPA; kneels, "
                "spikes and special teams are excluded."
            )
        else:
            summary = (
                f"Team-level defensive{kind_label} EPA per play allowed (mean). "
                "Rank 1 is the lowest (best) EPA/play allowed. "
                "Limited to pass and rush plays with a valid EPA; kneels, "
                "spikes and special teams are excluded."
            )

        return out_df, summary

    def success_rate(
        self,
        side: Side = "offense",
        kind: Kind = "all",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Team-level success rate: the fraction of plays with positive EPA.

        Offense: rank 1 = highest success rate.
        Defense: rank 1 = LOWEST success rate allowed. A defense that lets the
        opponent succeed more often is worse, not better.
        """
        self._require("epa", "success")
        team_col = self._team_col_for_side(side)
        self._require(team_col)

        filtered = self.df.filter(
            self._offensive_play_expr()
            & self._filter_kind(kind)
            & pl.col(team_col).is_not_null()
        )

        grouped = filtered.group_by(team_col).agg(
            [
                pl.len().alias("plays"),
                pl.col("success").cast(pl.Float64).mean().alias("success_rate"),
            ]
        )

        higher_is_better = side == "offense"
        ranked = self._rank(grouped, "success_rate", higher_is_better)

        out_df = ranked.select(
            ["rank", pl.col(team_col).alias("team"), "success_rate", "plays"]
        )

        kind_label = "" if kind == "all" else f" {kind}"
        if side == "offense":
            summary = (
                f"Team-level offensive{kind_label} success rate "
                "(fraction of plays with positive EPA). "
                "Rank 1 is the highest (best) success rate."
            )
        else:
            summary = (
                f"Team-level defensive{kind_label} success rate allowed "
                "(fraction of opponent plays with positive EPA). "
                "Rank 1 is the lowest (best) success rate allowed."
            )

        return out_df, summary

    def epa_per_dropback(
        self,
        side: Side = "offense",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Team-level mean EPA per dropback (pass attempts + sacks + scrambles).
        """
        self._require("epa")
        team_col = self._team_col_for_side(side)
        self._require(team_col)

        filtered = self.df.filter(
            self._dropback_expr()
            & pl.col("epa").is_not_null()
            & pl.col(team_col).is_not_null()
        )

        grouped = filtered.group_by(team_col).agg(
            [
                pl.len().alias("dropbacks"),
                pl.col("epa").mean().alias("epa_per_dropback"),
            ]
        )

        higher_is_better = side == "offense"
        ranked = self._rank(grouped, "epa_per_dropback", higher_is_better)

        out_df = ranked.select(
            ["rank", pl.col(team_col).alias("team"), "epa_per_dropback", "dropbacks"]
        )

        if side == "offense":
            summary = (
                "Team-level offensive EPA per dropback (mean), including sacks "
                "and scrambles. Rank 1 is the highest (best) EPA/dropback."
            )
        else:
            summary = (
                "Team-level defensive EPA per dropback allowed (mean), including "
                "sacks and scrambles. Rank 1 is the lowest (best) allowed."
            )

        return out_df, summary

    # ------------------------------------------------------------------
    # QB-level metrics
    # ------------------------------------------------------------------

    def qb_epa_cpoe(
        self,
        metric: Literal["epa_per_dropback", "cpoe"] = "epa_per_dropback",
        min_dropbacks: int = 100,
    ) -> Tuple[pl.DataFrame, str]:
        """
        QB-level EPA per dropback and CPOE.

        Dropbacks include sacks and scrambles, so the EPA figure reflects the
        cost of taking sacks. QBs are grouped by season and passer, NOT by
        team, so a mid-season trade does not split one player into two
        under-threshold samples.

        Only QBs with at least `min_dropbacks` dropbacks are included.
        """
        self._require("season", "posteam", "epa")
        passer_col = self._passer_col()

        if min_dropbacks < 0:
            raise ValueError("min_dropbacks must be non-negative.")

        dropbacks = self.df.filter(
            self._dropback_expr()
            & pl.col("epa").is_not_null()
            & pl.col(passer_col).is_not_null()
        )

        aggs = [
            pl.len().alias("dropbacks"),
            pl.col("epa").mean().alias("epa_per_dropback"),
            # All teams the QB played for, e.g. "NYJ/PHI" after a trade.
            pl.col("posteam").drop_nulls().unique().sort().alias("_teams"),
        ]
        if "cpoe" in self.df.columns:
            aggs.append(pl.col("cpoe").mean().alias("cpoe"))

        qb_stats = dropbacks.group_by(["season", passer_col]).agg(aggs)

        if "cpoe" not in qb_stats.columns:
            qb_stats = qb_stats.with_columns(
                pl.lit(None, dtype=pl.Float64).alias("cpoe")
            )

        qb_stats = qb_stats.with_columns(
            pl.col("_teams").list.join("/").alias("team")
        ).drop("_teams")

        qb_filtered = qb_stats.filter(pl.col("dropbacks") >= min_dropbacks)

        metric_col = "cpoe" if metric == "cpoe" else "epa_per_dropback"
        if metric_col == "cpoe":
            # Ranking by an all-null column would be meaningless.
            qb_filtered = qb_filtered.filter(pl.col("cpoe").is_not_null())
            if qb_filtered.is_empty():
                raise ValueError(
                    "No CPOE values available in this dataset; cannot rank by cpoe."
                )

        ranked = self._rank(qb_filtered, metric_col, higher_is_better=True)

        out_df = ranked.select(
            [
                "rank",
                "season",
                "team",
                pl.col(passer_col).alias("qb"),
                "dropbacks",
                "epa_per_dropback",
                "cpoe",
            ]
        )

        if metric_col == "epa_per_dropback":
            summary = (
                "QB-level EPA per dropback (mean), including sacks and "
                "scrambles, ranked best to worst. "
                f"Only QBs with at least {min_dropbacks} dropbacks are included."
            )
        else:
            summary = (
                "QB-level completion percentage over expected (CPOE), ranked "
                "best to worst. "
                f"Only QBs with at least {min_dropbacks} dropbacks are included."
            )

        return out_df, summary
