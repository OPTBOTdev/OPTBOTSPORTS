"""EMPIRICAL LAG AUDIT — prediction-symmetry test on every prior column.

A truly lagged prior knows nothing special about TODAY: corr(prior, y_today)
should approximately equal corr(prior, y_next_game). A leaky 'prior' that peeks
at today shows corr_today >> corr_next. We test every *prior*/*lag*/*eb* column
in the as-of player_priors table against on-ice xGF60, plus two controls:
  POSITIVE control: y_today itself relabeled as a 'prior' (must scream LEAK)
  NEGATIVE control: pure noise (must show ~0 gap)

Usage: python scripts/13_lag_symmetry_audit.py [--year 2023] [--top 25]
"""
import argparse

import numpy as np
import pandas as pd

OBS = r"D:\optbot\artifacts\player_game_obs_rebuilt.parquet"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    pri = pd.read_csv(rf"D:\{args.year}\player_rollups\player_priors_{args.year}.csv",
                      low_memory=False)
    obs = pd.read_parquet(OBS, columns=["playerId", "gamePk", "strength_global",
                                        "y_xgf60_game", "toi_sec"])
    ev = obs[(obs.strength_global == "5v5") & (obs.toi_sec > 300)]
    df = pri.merge(ev[["playerId", "gamePk", "y_xgf60_game"]],
                   on=["playerId", "gamePk"], how="inner")
    df = df.sort_values(["playerId", "gamePk"])
    df["y_next"] = df.groupby("playerId")["y_xgf60_game"].shift(-1)
    df = df.dropna(subset=["y_next"])
    print(f"rows with today+next outcome: {len(df):,}")

    rng = np.random.default_rng(0)
    df["_control_leak"] = df["y_xgf60_game"] + rng.normal(0, 0.5, len(df))
    df["_control_noise"] = rng.normal(0, 1, len(df))

    pats = ("prior", "_lag", "_eb", "ema")
    cols = [c for c in df.columns if df[c].dtype.kind in "if"
            and (any(p in c.lower() for p in pats) or c.startswith("_control"))]
    rows = []
    for c in cols:
        v = df[c]
        if v.notna().sum() < 5000 or v.std() == 0:
            continue
        ct = v.corr(df.y_xgf60_game)
        cn = v.corr(df.y_next)
        if np.isfinite(ct) and np.isfinite(cn):
            rows.append({"col": c, "corr_today": round(ct, 4),
                         "corr_next": round(cn, 4), "gap": round(ct - cn, 4)})
    res = pd.DataFrame(rows).sort_values("gap", ascending=False)
    res.to_csv(r"D:\optbot\artifacts\lag_symmetry_audit.csv", index=False)

    print("\n=== CONTROLS (method validation) ===")
    print(res[res.col.str.startswith("_control")].to_string(index=False))
    print(f"\n=== TOP {args.top} GAP (leak suspects if gap >> control-noise) ===")
    print(res[~res.col.str.startswith("_control")].head(args.top).to_string(index=False))
    real = res[~res.col.str.startswith("_control")]
    thresh = 0.05
    flagged = real[(real.gap > thresh) & (real.corr_today > 0.05)]
    print(f"\ncolumns tested: {len(real)} | flagged (gap>{thresh} & corr_today>0.05): "
          f"{len(flagged)}")
    print("VERDICT:", "ALL PRIORS SYMMETRIC — lag discipline empirically confirmed"
          if len(flagged) == 0 else f"INVESTIGATE {list(flagged.col.head(10))}")
