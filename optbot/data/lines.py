"""Line/role table — promotes the phase_d cooccurrence idea to a first-class artifact.

One row per (gamePk, teamId, playerId): line_no (F) / pair_no (D), top-2 partners by
shared 5v5 seconds, toi_share. Built ONLY from on-ice cooccurrence within the game —
no external line-combo scraping, so it is freeze-safe by construction.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_lines(spine: pd.DataFrame) -> pd.DataFrame:
    """spine: deduped window spine (needs playerId, teamId, gamePk, date, seconds,
    strength_global, window_id). Returns the per-game role table."""
    ev = spine[spine.strength_global.astype(str).str.replace("_", "").str.lower()
               .isin(["5v5", "5v55v5"])]
    toi = (ev.groupby(["gamePk", "teamId", "playerId"], as_index=False)
             .agg(toi_sec=("seconds", "sum"), date=("date", "first")))
    # rank within team-game by TOI -> line slots (F: 4 lines x 3, D handled same way v0)
    toi["rank"] = toi.groupby(["gamePk", "teamId"])["toi_sec"] \
                     .rank(ascending=False, method="first")
    toi["line_no"] = np.clip(((toi["rank"] - 1) // 3) + 1, 1, 6).astype(int)
    toi["toi_share"] = toi["toi_sec"] / toi.groupby(["gamePk", "teamId"])["toi_sec"] \
                                           .transform("max").clip(lower=1)
    return toi[["gamePk", "teamId", "playerId", "date", "line_no", "toi_sec", "toi_share"]]


def top_partners(spine: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Per (gamePk, playerId): top-k teammates by shared window seconds."""
    ev = spine[["gamePk", "teamId", "window_id", "playerId", "seconds"]]
    pairs = ev.merge(ev, on=["gamePk", "teamId", "window_id"], suffixes=("", "_mate"))
    pairs = pairs[pairs.playerId != pairs.playerId_mate]
    pairs["shared"] = pairs[["seconds", "seconds_mate"]].min(axis=1)
    g = (pairs.groupby(["gamePk", "playerId", "playerId_mate"], as_index=False)
              .agg(shared=("shared", "sum")))
    g["rk"] = g.groupby(["gamePk", "playerId"])["shared"].rank(ascending=False, method="first")
    top = g[g.rk <= k].sort_values(["gamePk", "playerId", "rk"])
    return (top.groupby(["gamePk", "playerId"])["playerId_mate"]
               .apply(list).rename("top_partners").reset_index())
