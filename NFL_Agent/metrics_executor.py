from __future__ import annotations

from typing import Tuple, Literal

import polars as pl


Side = Literal["offense", "defense"]
Kind = Literal["all", "pass", "rush"]


class MetricsExecutor:
    """
    Core metrics engine for nflfastR-style play-by-play data.

    Public usage:
        executor = MetricsExecutor(df)
        out_df, summary = executor.execute("ranked epa per play")

    The input `df` is expected to be a Polars DataFrame with at least:
        - season, season_type
        - posteam, defteam
        - epa
        - play_type and/or pass / rush flags
        - success (optional; will derive if missing)
        - pass_attempt, passer_player_name (for QB metrics)
        - cpoe (for QB CPOE; may be null / missing)
    """

    # ------------------------------------------------------------------
    # Initialization & normalization
    # ------------------------------------------------------------------

    def __init__(self, df: pl.DataFrame) -> None:
        # Work on a copy so we don't mutate upstream data
        self.df = df.clone()

        # Basic normalizations used by multiple metrics
        self._normalize_teams()
        self._normalize_play_type()
        self._ensure_success_flag()

    def _normalize_teams(self) -> None:
        """
        Normalize team columns (posteam, defteam) to lowercase strings.
        """
        cols = []
        for name in ("posteam", "defteam"):
            if name in self.df.columns:
                cols.append(
                    pl.col(name)
                    .cast(pl.Utf8)
                    .str.to_lowercase()
                    .alias(name)
                )
        if cols:
            self.df = self.df.with_columns(cols)

    def _normalize_play_type(self) -> None:
        """
        Derive a clean `play_type` category that distinguishes at least:
          - "pass"
          - "rush"
          - None for everything else

        Uses a combination of:
          - boolean flags `pass`, `rush` if present
          - textual `play_type` column if present
        """
        df = self.df

        has_flag_pass = "pass" in df.columns
        has_flag_rush = "rush" in df.columns
        has_pt = "play_type" in df.columns

        if not (has_flag_pass or has_flag_rush or has_pt):
            # nothing we can do; leave as-is
            return

        expr = pl.lit(None).cast(pl.Utf8)

        # Derive from flags first (if available)
        if has_flag_pass:
            expr = pl.when(pl.col("pass") == 1).then("pass").otherwise(expr)
        if has_flag_rush:
            expr = pl.when(pl.col("rush") == 1).then("rush").otherwise(expr)

        # Fall back to textual play_type patterns if needed
        if has_pt:
            pt = pl.col("play_type").cast(pl.Utf8).str.to_lowercase()

            expr = (
                pl.when(expr.is_not_null())
                .then(expr)
                .when(pt.str.contains("pass"))
                .then("pass")
                .when(pt.str.contains("rush") | pt.str.contains("run"))
                .then("rush")
                .otherwise(expr)
            )

        self.df = self.df.with_columns(expr.alias("play_type"))

    def _ensure_success_flag(self) -> None:
        """
        Ensure there is a boolean `success` column.

        If missing, define:
            success := epa > 0
        """
        if "success" in self.df.columns:
            return

        if "epa" not in self.df.columns:
            # Can't derive; leave as-is
            return

        self.df = self.df.with_columns(
            (pl.col("epa") > 0).alias("success")
        )

    # ------------------------------------------------------------------
    # Helpers / filters
    # ------------------------------------------------------------------

    def _team_col_for_side(self, side: Side) -> str:
        """
        Return the team column name for offense or defense.
        """
        return "posteam" if side == "offense" else "defteam"

    def _filter_kind(self, kind: Kind) -> pl.Expr:
        """
        Expression to filter by kind of play: all, pass, or rush.
        Assumes normalized `play_type`.
        """
        if kind == "all":
            return pl.lit(True)

        if "play_type" not in self.df.columns:
            # No play_type info; can't filter by pass/rush
            return pl.lit(True)

        return pl.col("play_type") == kind

    def _dropback_expr(self) -> pl.Expr:
        """
        Expression that defines dropbacks, using `qb_dropback` or `dropback`
        or falling back to pass attempts.
        """
        if "qb_dropback" in self.df.columns:
            return pl.col("qb_dropback") == 1
        if "dropback" in self.df.columns:
            return pl.col("dropback") == 1
        # Fallback: use pass attempts
        if "pass_attempt" in self.df.columns:
            return pl.col("pass_attempt") == 1
        # Worst case: no filter
        return pl.lit(True)

    # ------------------------------------------------------------------
    # Team-level metrics
    # ------------------------------------------------------------------

    def _epa_per_play(
        self,
        side: Side = "offense",
        kind: Kind = "all",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Compute team-level EPA per play.

        For offense:
          - group by posteam
          - rank higher EPA as better (rank 1 is best)

        For defense:
          - group by defteam
          - rank lower EPA allowed as better (rank 1 is best)
        """
        df = self.df

        if "epa" not in df.columns:
            raise ValueError("epa column missing; cannot compute EPA per play.")

        team_col = self._team_col_for_side(side)
        if team_col not in df.columns:
            raise ValueError(f"{team_col} column missing; cannot compute team metrics.")

        mask = self._filter_kind(kind)
        filtered = df.filter(mask)

        grouped = (
            filtered
            .group_by(team_col)
            .agg(
                [
                    pl.count().alias("plays"),
                    pl.col("epa").mean().alias("epa_per_play"),
                ]
            )
        )

        # Sort order: offense -> descending; defense -> ascending (EPA allowed)
        ascending = side == "defense"
        sorted_df = grouped.sort("epa_per_play", descending=not ascending)

        ranked = sorted_df.with_columns(
            pl.col("epa_per_play")
            .rank(method="dense", descending=not ascending)
            .cast(pl.Int64)
            .alias("rank")
        ).sort("rank")

        out_df = ranked.select(
            [
                "rank",
                pl.col(team_col).alias("team"),
                "epa_per_play",
                "plays",
            ]
        )

        if side == "offense":
            summary = (
                "Computed team-level offensive EPA per play. "
                "Rank 1 is highest (best) EPA/play for offense. "
                "Includes a 'plays' column for sample size context."
            )
        else:
            summary = (
                "Computed team-level defensive EPA per play allowed. "
                "Rank 1 is lowest (best) EPA/play allowed for defense. "
                "Includes a 'plays' column for sample size context."
            )

        if kind != "all":
            summary = summary.replace("EPA per play", f"{kind} EPA per play")

        return out_df, summary

    def _success_rate(
        self,
        side: Side = "offense",
        kind: Kind = "all",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Compute team-level success rate:
            success_rate = mean(success)
        """
        df = self.df

        if "success" not in df.columns:
            raise ValueError("success column missing; cannot compute success rate.")

        team_col = self._team_col_for_side(side)
        if team_col not in df.columns:
            raise ValueError(f"{team_col} column missing; cannot compute team metrics.")

        mask = self._filter_kind(kind)
        filtered = df.filter(mask)

        grouped = (
            filtered
            .group_by(team_col)
            .agg(
                [
                    pl.count().alias("plays"),
                    pl.col("success").mean().alias("success_rate"),
                ]
            )
        )

        # For both offense and defense: higher success_rate is better
        sorted_df = grouped.sort("success_rate", descending=True)
        ranked = sorted_df.with_columns(
            pl.col("success_rate")
            .rank(method="dense", descending=True)
            .cast(pl.Int64)
            .alias("rank")
        ).sort("rank")

        out_df = ranked.select(
            [
                "rank",
                pl.col(team_col).alias("team"),
                "success_rate",
                "plays",
            ]
        )

        side_label = "offensive" if side == "offense" else "defensive"
        if kind == "all":
            kind_label = ""
        else:
            kind_label = f" {kind}"

        summary = (
            f"Computed team-level{kind_label} {side_label} success rate. "
            "success_rate is the fraction of plays with positive EPA. "
            "Rank 1 is highest (best) success_rate."
        )

        return out_df, summary

    def _epa_per_dropback(
        self,
        side: Side = "offense",
    ) -> Tuple[pl.DataFrame, str]:
        """
        Compute team-level EPA per dropback.
        """
        df = self.df

        if "epa" not in df.columns:
            raise ValueError("epa column missing; cannot compute EPA per dropback.")

        team_col = self._team_col_for_side(side)
        if team_col not in df.columns:
            raise ValueError(f"{team_col} column missing; cannot compute team metrics.")

        mask = self._dropback_expr()
        filtered = df.filter(mask)

        grouped = (
            filtered
            .group_by(team_col)
            .agg(
                [
                    pl.count().alias("dropbacks"),
                    pl.col("epa").mean().alias("epa_per_dropback"),
                ]
            )
        )

        ascending = side == "defense"
        sorted_df = grouped.sort("epa_per_dropback", descending=not ascending)
        ranked = sorted_df.with_columns(
            pl.col("epa_per_dropback")
            .rank(method="dense", descending=not ascending)
            .cast(pl.Int64)
            .alias("rank")
        ).sort("rank")

        out_df = ranked.select(
            [
                "rank",
                pl.col(team_col).alias("team"),
                "dropbacks",
                "epa_per_dropback",
            ]
        )

        if side == "offense":
            summary = (
                "Computed team-level offensive EPA per dropback. "
                "Rank 1 is highest (best) EPA/dropback."
            )
        else:
            summary = (
                "Computed team-level defensive EPA per dropback allowed. "
                "Rank 1 is lowest (best) EPA/dropback allowed."
            )

        return out_df, summary

    # ------------------------------------------------------------------
    # QB-level metrics
    # ------------------------------------------------------------------

    def _qb_epa_cpoe(
        self,
        metric: Literal["epa_per_pass", "cpoe"] = "epa_per_pass",
        min_dropbacks: int = 100,
    ) -> Tuple[pl.DataFrame, str]:
        """
        Compute QB-level EPA per pass and CPOE over the (already-filtered) dataset.

        Metric choices:
          - "epa_per_pass": rank QBs by mean EPA per pass play
          - "cpoe": rank QBs by completion percentage over expected

        Only QBs with at least `min_dropbacks` pass attempts are included.
        """
        df = self.df

        needed_cols = ["season", "posteam", "passer_player_name", "epa", "pass_attempt"]
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for QB metrics: {missing}")

        # Filter to pass attempts
        pass_plays = df.filter(pl.col("pass_attempt") == 1)

        # Group by QB (season + team + name)
        group_cols = ["season", "posteam", "passer_player_name"]
        qb_stats = (
            pass_plays
            .group_by(group_cols)
            .agg(
                [
                    pl.col("pass_attempt").sum().alias("dropbacks"),
                    pl.col("epa").mean().alias("epa_per_pass"),
                    # If cpoe column is missing or null, this will be null
                    pl.col("cpoe").mean().alias("cpoe"),
                ]
            )
        )

        # Filter out tiny samples
        qb_filtered = qb_stats.filter(pl.col("dropbacks") >= min_dropbacks)

        metric_col = "epa_per_pass" if metric == "epa_per_pass" else "cpoe"
        if metric_col not in qb_filtered.columns:
            raise ValueError(f"Metric column {metric_col} not found in QB stats.")

        # Rank: higher metric value is better
        ranked = (
            qb_filtered
            .sort(metric_col, descending=True)
            .with_columns(
                pl.col(metric_col)
                .rank(method="dense", descending=True)
                .cast(pl.Int64)
                .alias("rank")
            )
            .sort("rank")
        )

        out_df = ranked.select(
            [
                "rank",
                "season",
                pl.col("posteam").alias("team"),
                pl.col("passer_player_name").alias("qb"),
                "dropbacks",
                "epa_per_pass",
                "cpoe",
            ]
        )

        if metric_col == "epa_per_pass":
            summary = (
                "Computed QB-level EPA per pass play, ranked by epa_per_pass. "
                f"Only QBs with at least {min_dropbacks} dropbacks are included."
            )
        else:
            summary = (
                "Computed QB-level completion percentage over expected (CPOE), "
                "ranked by cpoe. "
                f"Only QBs with at least {min_dropbacks} dropbacks are included."
            )

        return out_df, summary

    # ------------------------------------------------------------------
    # Query router
    # ------------------------------------------------------------------

    def execute(self, query: str) -> Tuple[pl.DataFrame, str]:
        q = query.lower().strip()
    
        ALIASES = {
            "offensive epa": "ranked epa per play",
            "offense epa": "ranked epa per play",
            "epa offense": "ranked epa per play",
            "team epa": "ranked epa per play",
    
            "qb epa": "qb ranked by epa per pass",
            "quarterback epa": "qb ranked by epa per pass",
        }
    
        for k, v in ALIASES.items():
            if k in q:
                q = v
                break
    
        # ----- "EPA per pass" disambiguation -----
        # If QB intent -> QB EPA/pass, else -> TEAM EPA/dropback
        if "epa per pass" in q or "epa per pass play" in q:
            if any(k in q for k in ["qb", "quarterback", "passer"]):
                return self._qb_epa_cpoe(metric="epa_per_pass")
            return self._epa_per_dropback(side="offense")
    
        # ----- QB metrics -----
        if any(k in q for k in ["qb", "quarterback", "passer"]):
            if "cpoe" in q or "completion percentage over expected" in q:
                return self._qb_epa_cpoe(metric="cpoe")
            return self._qb_epa_cpoe(metric="epa_per_pass")
    
        # ----- Team EPA per play -----
        if "epa" in q and "play" in q:
            # Defensive EPA allowed
            if "def" in q or "allowed" in q:
                if "rush" in q:
                    return self._epa_per_play(side="defense", kind="rush")
                if "pass" in q:
                    return self._epa_per_play(side="defense", kind="pass")
                return self._epa_per_play(side="defense", kind="all")
    
            # Offensive
            if "rush" in q:
                return self._epa_per_play(side="offense", kind="rush")
            if "pass" in q:
                return self._epa_per_play(side="offense", kind="pass")
            return self._epa_per_play(side="offense", kind="all")
    
        # ----- Success rate -----
        if "success" in q:
            side: Side = "offense" if "off" in q or "offense" in q else "defense"
            if "pass" in q:
                return self._success_rate(side=side, kind="pass")
            if "rush" in q:
                return self._success_rate(side=side, kind="rush")
            return self._success_rate(side=side, kind="all")
    
        # ----- EPA per dropback -----
        if "dropback" in q:
            if "def" in q:
                return self._epa_per_dropback(side="defense")
            return self._epa_per_dropback(side="offense")
    
        # Fallback: offensive overall EPA per play
        return self._epa_per_play(side="offense", kind="all")