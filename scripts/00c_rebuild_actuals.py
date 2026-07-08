"""Fix 2: rebuild player-game ACTUALS from the per-game final_windows CSVs and
recompute residuals vs the OOF baseline mu.

Root cause being fixed: phaseC_player_game_observations.parquet shipped with
y_xgf_onice_w == 0 for every row (actuals join silently failed), so every residual
was just -mu — which is also why the Kalman bands covered 12% instead of 80%.

Sources per season: D:\\<yy>\\final_windows\\player_windows_train_<gamePk>_xg.csv
  (columns: playerId, window_id, seconds, strength..., xGF, xGA, GF, GA, SF, SA)
Baseline mu:        D:\\baseline_model_output\\player_windows_with_baseline_<season>.parquet

Output: D:\\optbot\\artifacts\\player_game_obs_rebuilt.parquet
  (same schema as the broken file, with REAL y and residuals + provenance column)

Usage: python scripts/00c_rebuild_actuals.py [--seasons 2023 2024] [--smoke N]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASON_DIRS = {  # season label -> game-files dir  (2017 lives in 2017_Final)
    20172018: r"D:\2017_Final", 20182019: r"D:\2018", 20192020: r"D:\2019",
    20202021: r"D:\2020", 20212022: r"D:\2021", 20222023: r"D:\2022",
    20232024: r"D:\2023", 20242025: r"D:\2024",
}
BASE = r"D:\baseline_model_output"
OUT = r"D:\optbot\artifacts\player_game_obs_rebuilt.parquet"

USE = ["playerId", "window_id", "seconds", "xGF", "xGA", "GF", "GA", "SF", "SA"]


def _norm_strength(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("_", "").str.lower().map(
        lambda v: "5v5" if v in ("5v5", "5v55v5") else ("pp" if v.startswith("pp") else
                                                        "pk" if v.startswith("pk") else v))


def season_actuals(season: int, limit: int | None = None) -> pd.DataFrame:
    d = SEASON_DIRS[season]
    files = sorted(glob.glob(f"{d}\\final_windows\\player_windows_train_*_xg.csv"))
    files = [f for f in files if "backup" not in f]
    if limit:
        files = files[:limit]
    if not files:
        print(f"WARN {season}: no final_windows files under {d}")
        return pd.DataFrame()
    rows = []
    for i, fp in enumerate(files):
        gamePk = int(Path(fp).stem.split("_")[3])
        df = pd.read_csv(fp)
        strength_col = next((c for c in ("strength_global", "strength") if c in df.columns), None)
        cols = [c for c in USE if c in df.columns]
        g = df[cols + ([strength_col] if strength_col else [])].copy()
        g["strength"] = _norm_strength(g[strength_col]) if strength_col else "5v5"
        agg = g.groupby(["playerId", "strength"], as_index=False).agg(
            toi_sec=("seconds", "sum"), y_xgf_onice_w=("xGF", "sum"),
            y_xga_onice_w=("xGA", "sum"), y_gf=("GF", "sum"), y_ga=("GA", "sum"),
            y_sf=("SF", "sum"), y_sa=("SA", "sum"))
        agg["gamePk"] = gamePk
        agg["season"] = season
        rows.append(agg)
        if (i + 1) % 200 == 0:
            print(f"  {season}: {i+1}/{len(files)} games")
    return pd.concat(rows, ignore_index=True)


def rebuild(seasons, smoke: int | None = None) -> pd.DataFrame:
    old = pd.read_parquet(f"{BASE}\\phaseC_player_game_observations.parquet")
    old["strength"] = _norm_strength(old["strength_global"])
    out = []
    for season in seasons:
        y = season_actuals(season, smoke)
        if y.empty:
            continue
        mu = old[old.season == season][["playerId", "gamePk", "date", "strength",
                                        "mu_xgf_onice_w", "mu_xga_onice_w", "n_eff_toi_games"]]
        m = mu.merge(y, on=["playerId", "gamePk", "strength"], how="inner")
        match = len(m) / max(len(mu), 1)
        print(f"{season}: joined {len(m):,}/{len(mu):,} mu-rows ({match:.1%})")
        if match < 0.9 and not smoke:
            print(f"WARN {season}: join under 90% — investigate before trusting")
        mins = m["toi_sec"].clip(lower=1) / 60.0
        m["y_xgf60_game"] = 60 * m["y_xgf_onice_w"] / mins
        m["y_xga60_game"] = 60 * m["y_xga_onice_w"] / mins
        m["mu_xgf60_game"] = 60 * m["mu_xgf_onice_w"] / mins
        m["mu_xga60_game"] = 60 * m["mu_xga_onice_w"] / mins
        m["resid_xgf60_game"] = m["y_xgf60_game"] - m["mu_xgf60_game"]
        m["resid_xga60_game"] = m["y_xga60_game"] - m["mu_xga60_game"]
        m["strength_global"] = m["strength"]
        m["provenance"] = "rebuilt_from_final_windows_v1"
        out.append(m)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int,
                    default=list(SEASON_DIRS))
    ap.add_argument("--smoke", type=int, default=None,
                    help="only N games/season — proves the join, prints residual centering")
    args = ap.parse_args()
    seasons = [s if s > 9999 else {2017: 20172018, 2018: 20182019, 2019: 20192020,
                                   2020: 20202021, 2021: 20212022, 2022: 20222023,
                                   2023: 20232024, 2024: 20242025}[s] for s in args.seasons]
    df = rebuild(seasons, args.smoke)
    if df.empty:
        sys.exit("nothing rebuilt")
    ev = df[df.strength == "5v5"]
    print("\n5v5 sanity: y_xgf60 mean %.3f | mu_xgf60 mean %.3f | resid mean %.3f (want ~0)"
          % (ev.y_xgf60_game.mean(), ev.mu_xgf60_game.mean(), ev.resid_xgf60_game.mean()))
    if not args.smoke:
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(OUT, index=False)
        print("wrote", OUT)
