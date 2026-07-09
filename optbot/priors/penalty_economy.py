"""Penalty economy — the off-ice value line a 5v5-only model cannot see.

A player who draws penalties manufactures PP time; one who takes them donates it.
Invisible in on-ice 5v5 xG, so it ships as a SEPARATE value line:

    econ60(p, t0) = shrunk_delta60(p, t0) * NET_GOALS_PER_PENALTY

NET_GOALS_PER_PENALTY ~ 0.17: league PP converts ~20% per 2-min opportunity minus
~3% shorthanded counter. Deterministic economics, not a model.

Source: D:\\<yy>\\penalty_priors_per_game.csv (all seasons) — per-game lagged EMA
priors (leakage-audited in the legacy review) + per-game ACTUAL effective counts,
which makes a transport backtest possible: does penalty skill survive a team change?
"""
from __future__ import annotations
import glob

import pandas as pd

NET_GOALS_PER_PENALTY = 0.17
# NOTE: pen_neff_minutes_prior is in DECAYED-WEIGHT units (median ~5.5), not minutes.
# K tuned on pre-2024 moves, validated on 2024+ holdout: RMSE 0.2006 vs 0.2122
# zero-baseline (-5.5%), corr 0.31. Heavy shrinkage is correct at 40-GP horizons.
K_SHRINK = 20.0


def load_all(pattern: str = r"D:\20*\penalty_priors_per_game.csv") -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["gameid", "playerId"])
    df["gameid"] = df["gameid"].astype("int64")
    df["playerId"] = df["playerId"].astype("int64")
    # per-game files carry no date; derive order from gameid (monotone within season)
    return df.sort_values(["playerId", "gameid"]).reset_index(drop=True)


def snapshot(pen: pd.DataFrame, t0_gamePk: int) -> pd.DataFrame:
    """Frozen penalty-economy state per player strictly before a gamePk.
    gameid ordering is freeze-legal because priors are already lagged per game."""
    m = pen[pen.gameid < t0_gamePk]
    last = m.groupby("playerId").tail(1).copy()
    shrink = last["pen_neff_minutes_prior"] / (last["pen_neff_minutes_prior"] + K_SHRINK)
    last["pen_delta60_shrunk"] = shrink * last["pen_delta60_prior_ev"].fillna(0.0)
    last["pen_econ_goals60"] = last["pen_delta60_shrunk"] * NET_GOALS_PER_PENALTY
    return last[["playerId", "gameid", "pen_delta60_shrunk", "pen_econ_goals60",
                 "pen_taken60_prior_ev", "pen_drawn60_prior_ev",
                 "pen_neff_minutes_prior"]]


def actual_post(pen: pd.DataFrame, player_id: int, t0_gamePk: int,
                horizon_games: int = 40) -> dict:
    m = pen[(pen.playerId == player_id) & (pen.gameid >= t0_gamePk)] \
        .sort_values("gameid").head(horizon_games)
    minutes = m["ev_minutes"].sum()
    if minutes < 100:
        return {"ok": False}
    delta60 = 60.0 * (m["pen_effective_drawn"].sum() - m["pen_effective_taken"].sum()) / minutes
    return {"ok": True, "gp": int(len(m)), "minutes": float(minutes),
            "actual_delta60": float(delta60)}
