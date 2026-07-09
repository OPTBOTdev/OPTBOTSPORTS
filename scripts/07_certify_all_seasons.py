"""Conservation + external certification, all 7 seasons vs MoneyPuck shots files.
Outputs artifacts/certification_report.json + prints per-season table.
"""
import json
from pathlib import Path

import pandas as pd

ART = Path(r"D:\optbot\artifacts")
MP = {20182019: 2018, 20192020: 2019, 20202021: 2020, 20212022: 2021,
      20222023: 2022, 20232024: 2023, 20242025: 2024}

if __name__ == "__main__":
    pw = pd.read_parquet(ART / "perfect_windows_v2.parquet",
                         columns=["season", "gamePk", "window_id", "teamId", "y_GF", "y_xGF"])
    out = []
    for season, yr in MP.items():
        try:
            sh = pd.read_csv(rf"D:\shots_{yr}.csv",
                             usecols=["game_id", "goal", "homeSkatersOnIce",
                                      "awaySkatersOnIce", "xGoal"])
        except FileNotFoundError:
            print(f"{season}: no shots file")
            continue
        sh5 = sh[(sh.homeSkatersOnIce == 5) & (sh.awaySkatersOnIce == 5)]
        w = pw[pw.season == season].copy()
        w["mp_id"] = w.gamePk - (yr * 1_000_000 + 0)  # e.g. 2024020001 - 2024000000
        w["mp_id"] = w.gamePk % 1_000_000
        tw = w.groupby(["mp_id", "window_id", "teamId"]).agg(
            GF=("y_GF", "max"), xGF=("y_xGF", "max")).reset_index()
        pg_w = tw.groupby("mp_id").agg(w_GF=("GF", "sum"), w_xGF=("xGF", "sum"))
        pg_m = sh5.groupby("game_id").agg(m_G=("goal", "sum"), m_xG=("xGoal", "sum"))
        j = pg_w.join(pg_m, how="inner")
        rec = {"season": season, "games": int(len(j)),
               "goal_exact_pct": round(100 * float((j.w_GF == j.m_G).mean()), 2),
               "goal_corr": round(float(j.w_GF.corr(j.m_G)), 4),
               "goal_mean_diff": round(float((j.w_GF - j.m_G).mean()), 4),
               "xg_corr": round(float(j.w_xGF.corr(j.m_xG)), 4),
               "xg_level_ratio": round(float(j.w_xGF.sum() / j.m_xG.sum()), 4)}
        out.append(rec)
        print(rec)
    (ART / "certification_report.json").write_text(json.dumps(out, indent=2))
    bad = [r for r in out if r["goal_exact_pct"] < 90 or r["xg_corr"] < 0.95]
    print("\nCERTIFIED" if not bad else f"\nATTENTION: {len(bad)} season(s) below bar: {bad}")
