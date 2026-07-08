"""Trade/UFA ledger — the backtest's ground-truth population.

Schema (one row per qualifying player-move):
  player_id, player_name, move_date, from_team, to_team, move_type{trade,ufa,waiver},
  career_5v5_minutes_pre, post_move_gp

Sources, in preference order:
  1. data/ledger_manual.csv          (curated — always wins on conflict)
  2. NHLe API transactions scrape    (scripts/01_build_ledger.py fetches; cached to csv)

Qualification (configs/default.yaml): >=1000 career 5v5 minutes before the move and
>=20 GP after it. Both are computed from OUR data (player_game_obs), not trusted
from any external source — the ledger must agree with the data it will be scored on.
"""
from __future__ import annotations
import pandas as pd

COLUMNS = ["player_id", "player_name", "move_date", "from_team", "to_team",
           "move_type", "career_5v5_minutes_pre", "post_move_gp"]


def qualify(ledger: pd.DataFrame, obs: pd.DataFrame,
            min_minutes: float = 1000.0, min_post_gp: int = 20) -> pd.DataFrame:
    """Recompute qualification fields from player_game_obs and filter."""
    ev = obs[obs.strength_global == "5v5"][["playerId", "date", "toi_sec", "gamePk"]]
    rows = []
    for r in ledger.itertuples():
        mine = ev[ev.playerId == r.player_id]
        pre_min = mine[mine.date < r.move_date]["toi_sec"].sum() / 60.0
        post_gp = mine[mine.date >= r.move_date]["gamePk"].nunique()
        rows.append({**r._asdict(), "career_5v5_minutes_pre": pre_min, "post_move_gp": post_gp})
    q = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    return q[(q.career_5v5_minutes_pre >= min_minutes) & (q.post_move_gp >= min_post_gp)]


def actual_post_move(obs: pd.DataFrame, player_id: int, move_date: str,
                     horizon_games: int = 40) -> dict:
    """Ground truth: first N games' 5v5 on-ice rates after the move."""
    ev = obs[(obs.strength_global == "5v5") & (obs.playerId == player_id)
             & (obs.date >= move_date)].sort_values("date")
    games = ev.drop_duplicates("gamePk").head(horizon_games)["gamePk"]
    w = ev[ev.gamePk.isin(games)]
    m = w["toi_sec"].sum() / 60.0
    if m < 60:
        return {"ok": False}
    xgf60 = 60 * w["y_xgf_onice_w"].sum() / m
    xga60 = 60 * w["y_xga_onice_w"].sum() / m
    return {"ok": True, "gp": int(len(games)), "minutes": float(m),
            "xgf60": float(xgf60), "xga60": float(xga60),
            "xgf_pct": float(100 * xgf60 / (xgf60 + xga60))}
