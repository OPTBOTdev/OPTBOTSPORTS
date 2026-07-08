"""Assemble the Perfect Window table (contracts/window_schema.py) as a JOIN over
audited sources — never a new parser.

  deduped spine (artifacts/window_spine_dedup.parquet)      keys/context/team-priors
+ OOF baseline  (player_windows_with_baseline_<season>)     mu/sigma block
+ rebuilt actuals-side per-window outcomes (final_windows)  y_* block [optional pass]
+ talent prior  (artifacts/talent_asof.parquet)             talent block
+ lines         (artifacts/lines.parquet)                   line_no
+ games_since_team_change                                   for switch-weighting (C3)

Every build ends with contracts.window_schema.validate() — failure aborts the write.
"""
from __future__ import annotations
import glob

import pandas as pd

from ..contracts import window_schema as ws


def _norm_strength(s):
    return s.astype(str).str.replace("_", "").str.lower().map(
        lambda v: "5v5" if v in ("5v5", "5v55v5") else v)


def games_since_team_change(spine: pd.DataFrame) -> pd.Series:
    g = spine[["playerId", "teamId", "gamePk", "date"]].drop_duplicates(
        ["playerId", "gamePk"]).sort_values(["playerId", "date"])
    changed = g.groupby("playerId")["teamId"].transform(lambda t: t != t.shift(1))
    grp = changed.groupby(g["playerId"]).cumsum()
    since = g.assign(_grp=grp).groupby(["playerId", "_grp"]).cumcount()
    return g[["playerId", "gamePk"]].assign(games_since_team_change=since.values)


SPINE_COLS = [  # trimmed load — full 59-col spine is unnecessary for v0
    "season", "gamePk", "window_id", "teamId", "playerId", "date",
    "strength_global", "seconds", "duration",
    "period", "period_time_bucket", "score_bucket", "start_regime",
    "lever_zone_start", "fo_loc_enum", "stoppage_class_at_start",
    "after_icing", "bench_rights", "long_change", "home_away", "rinkid",
    "skater_diff", "opponent_teamId_start", "n_eff_games_team",
    "team_xgf60_prior_ev", "team_xga60_prior_ev", "team_pace60_ev_prior",
    "opp_team_xgf60_prior_ev", "opp_team_xga60_prior_ev", "opp_team_pace60_ev_prior",
    "team_gsaa_per60_prior_eb", "opp_gsaa_per60_prior_eb",
    "team_goalie_tier", "opp_goalie_tier",
]


def build(spine_path: str, baseline_glob: str, talent_asof_path: str,
          lines_path: str, out_path: str, strength: str = "5v5") -> dict:
    spine = pd.read_parquet(spine_path, columns=SPINE_COLS)
    spine["strength_norm"] = _norm_strength(spine["strength_global"])
    spine = spine[spine.strength_norm == strength]

    KEYS = ["season", "gamePk", "window_id", "teamId", "playerId"]
    mu = pd.concat([pd.read_parquet(f, columns=KEYS +
                                    ["mu_xgf60", "mu_xga60", "sigma_xgf_w", "sigma_xga_w"])
                    for f in sorted(glob.glob(baseline_glob))], ignore_index=True)
    # the mu files were generated from the pre-dedupe spine and carry the same
    # ~13% duplicate rows — drop on keys or the left-join inflates exposure
    mu = mu.drop_duplicates(subset=KEYS)
    df = spine.merge(mu, on=["season", "gamePk", "window_id", "teamId", "playerId"],
                     how="left")

    tal = pd.read_parquet(talent_asof_path)  # as-of table: one row per (player, game)
    df = df.merge(tal[["playerId", "gamePk", "raw_off", "raw_def", "n_eff"]]
                  .rename(columns={"n_eff": "talent_n_eff"}),
                  on=["playerId", "gamePk"], how="left")

    lines = pd.read_parquet(lines_path)
    df = df.merge(lines[["gamePk", "teamId", "playerId", "line_no"]],
                  on=["gamePk", "teamId", "playerId"], how="left")

    gsc = games_since_team_change(spine)
    df = df.merge(gsc, on=["playerId", "gamePk"], how="left")
    df["schema_version"] = ws.SCHEMA_VERSION

    stats = {"rows": len(df),
             "mu_coverage": float(df["mu_xgf60"].notna().mean()),
             "talent_coverage": float(df["talent_n_eff"].notna().mean()),
             "line_coverage": float(df["line_no"].notna().mean())}
    df.to_parquet(out_path, index=False)
    return stats
