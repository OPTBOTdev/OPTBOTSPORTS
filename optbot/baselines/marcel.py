"""Marcel-style projection — the industry-standard bar the CIN must beat.

Weighted mean of the last 3 seasons' 5v5 on-ice rates (5/4/3), regressed to league mean
by minutes, small age adjustment. Also provides naive carryover and Marcel+team-shift.
All computed strictly from games before t0.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

W = (5.0, 4.0, 3.0)


def _season_rates(obs: pd.DataFrame) -> pd.DataFrame:
    ev = obs[obs.strength_global == "5v5"]
    g = ev.groupby(["playerId", "season"]).agg(
        toi_sec=("toi_sec", "sum"),
        xgf_w=("y_xgf_onice_w", "sum"),
        xga_w=("y_xga_onice_w", "sum"),
        last_date=("date", "max"),
    ).reset_index()
    m = g["toi_sec"] / 60.0
    g["xgf60"] = 60.0 * g["xgf_w"] / m.clip(lower=1)
    g["xga60"] = 60.0 * g["xga_w"] / m.clip(lower=1)
    g["minutes"] = m
    return g


def project(obs: pd.DataFrame, player_id: int, t0: str,
            regression_minutes: float = 2400.0, age: float | None = None,
            age_knot: float = 27.0) -> dict:
    rates = _season_rates(obs[obs.date < t0])
    league_xgf = np.average(rates["xgf60"], weights=rates["minutes"])
    league_xga = np.average(rates["xga60"], weights=rates["minutes"])

    mine = rates[rates.playerId == player_id].sort_values("season").tail(3)
    if mine.empty:
        return {"xgf60": league_xgf, "xga60": league_xga,
                "xgf_pct": 100 * league_xgf / (league_xgf + league_xga), "basis": "league"}

    w = np.array(W[-len(mine):]) * mine["minutes"].values
    xgf = float(np.average(mine["xgf60"], weights=w))
    xga = float(np.average(mine["xga60"], weights=w))
    n = float(mine["minutes"].sum())
    a = n / (n + regression_minutes)
    xgf = a * xgf + (1 - a) * league_xgf
    xga = a * xga + (1 - a) * league_xga
    if age is not None:                       # mild symmetric age curve
        adj = -0.008 * max(age - age_knot, 0) + 0.004 * max(age_knot - age, 0)
        xgf *= (1 + adj); xga *= (1 - adj / 2)
    return {"xgf60": xgf, "xga60": xga,
            "xgf_pct": 100 * xgf / (xgf + xga), "basis": f"{len(mine)}season"}


def carryover(obs: pd.DataFrame, player_id: int, t0: str) -> dict:
    """Last-season-as-is — what the eye test uses."""
    rates = _season_rates(obs[obs.date < t0])
    mine = rates[rates.playerId == player_id].sort_values("season").tail(1)
    if mine.empty:
        return project(obs, player_id, t0)
    r = mine.iloc[0]
    return {"xgf60": float(r.xgf60), "xga60": float(r.xga60),
            "xgf_pct": float(100 * r.xgf60 / (r.xgf60 + r.xga60)), "basis": "carryover"}


def marcel_team_adjusted(obs: pd.DataFrame, player_id: int, t0: str,
                         from_team_xgfpct: float, to_team_xgfpct: float) -> dict:
    """The strawman a smart skeptic proposes: Marcel shifted halfway to the team delta."""
    m = project(obs, player_id, t0)
    m["xgf_pct"] = m["xgf_pct"] + 0.5 * (to_team_xgfpct - from_team_xgfpct)
    m["basis"] = "marcel+team"
    return m
