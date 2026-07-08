"""Destination scenario builder (generalizes the Matthews prototype).

Key audit fix baked in: a traded player's deployment priors describe his OLD team.
The scenario REPLACES the deployment/role block with the DESTINATION role template and
keeps only what genuinely travels with the player (bio, talent prior, style).

template = empirical distribution of context for the destination team's line-N slot,
built by bootstrap-resampling REAL windows played by that slot (preserves correlations
between context columns — never sample columns independently).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import numpy as np
import pandas as pd

TRAVELS_WITH_PLAYER = ["age", "height_cm", "weight_kg", "shootsCatches", "position",
                       "talent_off_shrunk", "talent_def_shrunk", "talent_se", "talent_n_eff",
                       "prior_off60_eb", "prior_def60_eb", "n_eff_games", "n_eff_minutes"]

BELONGS_TO_DESTINATION = ["team_xgf60_prior_ev", "team_xga60_prior_ev", "team_pace60_ev_prior",
                          "team_gsaa_per60_prior_eb", "team_goalie_tier",
                          "p_prior_oz_share_ev", "p_prior_dz_share_ev",
                          "share_vs_stars_ema_lag", "p_minutes_prior_ev",
                          "lever_zone_start", "score_bucket", "period", "period_time_bucket",
                          "start_regime", "home_away", "rinkid", "with_ids", "with_seconds",
                          "vs_ids", "vs_seconds", "mp_bucket"]


@dataclass
class Scenario:
    player_id: int
    as_of_date: str            # freeze date: NOTHING after this may be read
    dest_team: int
    line_no: int
    linemates: list[int] = field(default_factory=list)   # [] -> slot incumbents
    horizon_games: int = 40
    n_windows: int = 2000

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True, default=str)
                              .encode()).hexdigest()[:16]


def slot_incumbent(windows: pd.DataFrame, dest_team: int, line_no: int,
                   as_of_date: str, lines: pd.DataFrame) -> int:
    """Who held this line slot most recently before as_of_date."""
    pre = lines[(lines.teamId == dest_team) & (lines.date < as_of_date)
                & (lines.line_no == line_no)]
    if pre.empty:
        raise ValueError(f"no line-{line_no} history for team {dest_team} before {as_of_date}")
    return int(pre.sort_values("date").iloc[-1]["playerId"])


def build_windows(sc: Scenario, windows: pd.DataFrame, lines: pd.DataFrame,
                  player_row: pd.Series, rng_seed: int = 7) -> pd.DataFrame:
    """Synthetic windows = bootstrap of the slot's REAL pre-t0 windows,
    with the traveling-player block overwritten."""
    rng = np.random.default_rng(rng_seed)
    incumbent = slot_incumbent(windows, sc.dest_team, sc.line_no, sc.as_of_date, lines)
    # perfect_windows is already strength-filtered at build time (strength_norm);
    # do NOT filter on raw strength_global here — label drift ('5V5_5v5') burned us once.
    slot = windows[(windows.playerId == incumbent) & (windows.teamId == sc.dest_team)
                   & (windows.date < sc.as_of_date)]
    if len(slot) < 200:
        raise ValueError(f"slot support too thin: {len(slot)} windows for incumbent {incumbent}")

    take = slot.sample(n=sc.n_windows, replace=True, random_state=rng.integers(1 << 31))
    synth = take.copy().reset_index(drop=True)

    # overwrite the traveling-player block
    synth["playerId"] = sc.player_id
    for c in TRAVELS_WITH_PLAYER:
        if c in player_row.index:
            synth[c] = player_row[c]

    # optional explicit linemates: swap into with_ids keeping seconds structure
    if sc.linemates:
        def _swap(ids):
            ids = list(ids)
            for j, lm in enumerate(sc.linemates[: len(ids)]):
                ids[j] = lm
            return ids
        synth["with_ids"] = synth["with_ids"].map(_swap)

    synth["scenario_hash"] = sc.hash()
    return synth
