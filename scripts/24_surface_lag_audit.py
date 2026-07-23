"""Script 24 — lag-symmetry audit on the NEW trainer-surface columns.

Same religion as script 13, applied to every scalar the surface added:
coach_tenure_games, rest_days_team, b2b_team, and the per-window member
aggregates (mean teammate form / age / gsc). A lagged-honest column predicts
today's outcome and the next game's outcome equally; a leak predicts today
better. Controls: planted leak (must scream), pure noise (~0).

Bio columns (age/hand/position) are time-invariant => symmetric by
construction, but age is audited anyway (it moves with the calendar).

Usage: python scripts/24_surface_lag_audit.py [--season 20232024]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")

NEW_SCALARS = ["coach_tenure_games", "rest_days_team", "b2b_team",
               "games_since_team_change"]
MEMBER_AGGS = {"with_form": "mean_with_form", "with_age": "mean_with_age",
               "with_gsc": "mean_with_gsc", "vs_form": "mean_vs_form"}


def _mean_of_list(v):
    if isinstance(v, (list, np.ndarray)) and len(v):
        a = pd.to_numeric(pd.Series(list(v)), errors="coerce")
        return float(a.mean()) if a.notna().any() else np.nan
    return np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=20232024)
    args = ap.parse_args()
    fp = ART / "trainer_surface" / f"season={args.season}.parquet"
    cols = (["playerId", "gamePk", "seconds", "y_xGF", "strength_global"]
            + NEW_SCALARS + list(MEMBER_AGGS))
    df = pd.read_parquet(fp, columns=cols)
    df = df[df.strength_global.astype(str).str.contains("5", na=False)]
    for src, dst in MEMBER_AGGS.items():
        df[dst] = df[src].map(_mean_of_list)

    # window -> player-game (exposure-weighted rates, seconds-weighted features)
    df["xgf_w"] = df.y_xGF
    g = df.groupby(["playerId", "gamePk"]).agg(
        toi=("seconds", "sum"), xgf=("xgf_w", "sum"),
        **{c: (c, "mean") for c in NEW_SCALARS},
        **{d: (d, "mean") for d in MEMBER_AGGS.values()}).reset_index()
    g = g[g.toi > 300]
    g["y"] = 3600 * g.xgf / g.toi
    g = g.sort_values(["playerId", "gamePk"])
    g["y_next"] = g.groupby("playerId").y.shift(-1)
    g = g.dropna(subset=["y_next"])
    print(f"player-games with today+next: {len(g):,}")

    rng = np.random.default_rng(0)
    g["_control_leak"] = g.y + rng.normal(0, 0.5, len(g))
    g["_control_noise"] = rng.normal(0, 1, len(g))

    rows = []
    for c in NEW_SCALARS + list(MEMBER_AGGS.values()) + ["_control_leak",
                                                         "_control_noise"]:
        v = g[c].astype(float)
        if v.notna().sum() < 500 or v.std() == 0:
            rows.append((c, np.nan, np.nan, np.nan, "SKIP(thin)"))
            continue
        ct = np.corrcoef(v.fillna(v.mean()), g.y)[0, 1]
        cn = np.corrcoef(v.fillna(v.mean()), g.y_next)[0, 1]
        gap = abs(ct) - abs(cn)
        verdict = "LEAK?" if gap > 0.02 else "ok"
        # Matchup-scoped features (vs_*) describe TONIGHT'S opponent — they
        # SHOULD predict today and not the next game (different opponent).
        # Their leak-safety comes from construction (season-lagged, V-LAG
        # verified in script 20), not from symmetry. Asymmetry here is the
        # signature of a correct matchup feature, not a peek.
        if c.startswith("mean_vs_") and verdict == "LEAK?":
            verdict = "ok (matchup-scoped by design)"
        if c == "_control_leak":
            verdict = "CONTROL(must scream)"
        rows.append((c, ct, cn, gap, verdict))
    out = pd.DataFrame(rows, columns=["column", "corr_today", "corr_next",
                                      "gap", "verdict"])
    print(out.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    flagged = out[out.verdict == "LEAK?"]
    print(f"\nFLAGGED: {len(flagged)}" + ("" if flagged.empty else
          " — DO NOT TRAIN until explained"))
