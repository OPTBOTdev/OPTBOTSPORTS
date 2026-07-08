"""EB-shrunken residual talent prior (replaces Phase C for the MVP).

Input : phaseC_player_game_observations.parquet  (OOF residuals per player-game,
        2017-18..2024-25, all strengths, n_eff attached — audited GOOD)
Output: talent_prior(player_id, as_of_date) with talent_off/def_shrunk, n_eff, SE.

Math:
  raw(p, t)    = sum_{g<t} w_g * resid60_g / sum w_g,   w_g = toi_g * decay^(games_between)
  n_eff(p, t)  = sum w_g   (in game units)
  shrunk(p, t) = n_eff/(n_eff+K) * raw(p, t)            # toward 0 = league-average residual
  se(p, t)     = max(sqrt(var_w / n_eff), SE_FLOOR)     # FLOOR, never 0 — cold-start fix

K is fit by out-of-time validation: predict each season's mean residual from the prior
as-of Oct 1 of that season; choose K minimizing weighted MSE. No hand-tuning.

Strictly lagged: the as-of computation uses games with date < t only. This is the same
expand-then-lag discipline as build_causal_priors.py, reimplemented small and tested.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SE_FLOOR_DEFAULT = 0.15


def _decayed_running(df: pd.DataFrame, value_col: str, weight_col: str,
                     half_life_games: float) -> pd.DataFrame:
    """Per player: lagged decayed mean/var/n_eff as-of each game date (value at row i
    excludes game i). df must be sorted by (playerId, date)."""
    lam = 0.5 ** (1.0 / half_life_games)
    out_mean = np.empty(len(df))
    out_var = np.empty(len(df))
    out_neff = np.empty(len(df))
    vals = df[value_col].to_numpy(dtype=float)      # hoisted: no per-row fetch
    wts = df[weight_col].to_numpy(dtype=float)
    for _, idx in df.groupby("playerId", sort=False).indices.items():
        sw = swx = swx2 = 0.0
        for i in idx:  # idx is positional & date-ordered
            # record LAGGED state before ingesting game i  (leak-proof by construction)
            out_neff[i] = sw
            out_mean[i] = swx / sw if sw > 0 else np.nan
            out_var[i] = max(swx2 / sw - (swx / sw) ** 2, 0.0) if sw > 0 else np.nan
            w = wts[i]
            x = vals[i]
            if np.isfinite(x) and w > 0:
                sw = lam * sw + w
                swx = lam * swx + w * x
                swx2 = lam * swx2 + w * x * x
            else:
                sw *= lam
                swx *= lam
                swx2 *= lam
    return pd.DataFrame({"raw": out_mean, "var": out_var, "n_eff": out_neff}, index=df.index)


def build_asof_table(obs: pd.DataFrame, half_life_games: float = 40.0,
                     strength: str = "5v5", demean_by_season: bool = True) -> pd.DataFrame:
    """One row per (playerId, gamePk): the talent state KNOWN BEFORE that game.

    demean_by_season: talent is LEAGUE-RELATIVE. The baseline mu carries a known
    global under-prediction (~1.1 xGF60 as of Jul 2026 rebuild), so raw residuals
    are biased positive for everyone. Subtracting the season's TOI-weighted league
    mean residual makes talent immune to any global mu miscalibration.
    NOTE: uses the season's own mean — fine for training-era data; for a strict
    t0 freeze the snapshot() date filter applies BEFORE this function is called,
    so the demeaning constant only sees pre-t0 games.
    """
    ev = obs[obs.strength_global == strength].copy()
    ev["w"] = ev["toi_sec"].clip(lower=0) / 3600.0   # game weight in TOI-hours
    if demean_by_season:
        for col in ("resid_xgf60_game", "resid_xga60_game"):
            lm = ev.groupby("season").apply(
                lambda g: np.average(g[col].fillna(0), weights=g["w"].clip(lower=1e-9)))
            ev[col] = ev[col] - ev["season"].map(lm)
    ev = ev.sort_values(["playerId", "date"]).reset_index(drop=True)
    off = _decayed_running(ev, "resid_xgf60_game", "w", half_life_games)
    dfn = _decayed_running(ev, "resid_xga60_game", "w", half_life_games)
    return pd.DataFrame({
        "playerId": ev.playerId, "season": ev.season, "gamePk": ev.gamePk, "date": ev.date,
        "raw_off": off["raw"], "var_off": off["var"],
        "raw_def": -dfn["raw"], "var_def": dfn["var"],   # sign: positive = suppresses xGA
        "n_eff": off["n_eff"],
        "y_off_game": ev["resid_xgf60_game"], "y_w": ev["w"],  # kept for K fitting
    })


def fit_K(asof: pd.DataFrame, k_grid=(500, 1000, 1500, 2000, 3000, 4500)) -> tuple[float, pd.DataFrame]:
    """Choose K by predicting each game's residual with the lagged shrunken prior."""
    rows = []
    m = asof.dropna(subset=["raw_off"])
    for K in k_grid:
        pred = m["n_eff"] / (m["n_eff"] + K / 60.0) * m["raw_off"]   # K given in minutes -> hours
        err = (m["y_off_game"] - pred)
        mse = float(np.average(err ** 2, weights=m["y_w"]))
        rows.append({"K_minutes": K, "weighted_mse": mse})
    tab = pd.DataFrame(rows)
    best = float(tab.loc[tab.weighted_mse.idxmin(), "K_minutes"])
    return best, tab


def snapshot(asof: pd.DataFrame, as_of_date: str, K_minutes: float,
             se_floor: float = SE_FLOOR_DEFAULT) -> pd.DataFrame:
    """Talent prior frozen at a date (trade t0 or season open). Latest state per player."""
    m = asof[asof.date < as_of_date]
    last = m.sort_values("date").groupby("playerId").tail(1).copy()
    Kh = K_minutes / 60.0
    shrink = last["n_eff"] / (last["n_eff"] + Kh)
    last["talent_off_shrunk"] = shrink * last["raw_off"].fillna(0.0)
    last["talent_def_shrunk"] = shrink * last["raw_def"].fillna(0.0)
    last["talent_se"] = np.maximum(
        np.sqrt(last["var_off"].fillna(last["var_off"].median()) / last["n_eff"].clip(lower=1e-3)),
        se_floor)
    last["talent_n_eff"] = last["n_eff"]
    return last[["playerId", "date", "talent_off_shrunk", "talent_def_shrunk",
                 "talent_se", "talent_n_eff"]].rename(columns={"date": "last_game_date"})
