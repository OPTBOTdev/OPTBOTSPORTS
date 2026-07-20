"""THE FINAL BACKTEST — extended ledger, MoneyPuck actuals, 8-season machinery.

Player-game table from MP attachments (mp_xgf/mp_xga per player-game) + extraction
(toi, team, date) -> extended ledger (moves 2019-07 .. 2025-10-01; the summer-2025
class gets full first-40 actuals from the new 2025-26 season) -> v0 vs Marcel vs
carryover, all scored on MP-scale on-ice xGF%, frozen-clock, destination-team-only.

Usage: python scripts/18_final_backtest.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.backtest.harness import run, headline  # noqa: E402
from optbot.cin.conformal import ConformalBands  # noqa: E402
from optbot.data.ledger import qualify  # noqa: E402

ART = Path(r"D:\optbot\artifacts")


def build_mp_playergame() -> pd.DataFrame:
    po = pd.read_parquet(ART / "people_outcomes_all.parquet",
                         columns=["season", "gamePk", "window_id", "playerId",
                                  "teamId", "seconds", "date", "strength_global"])
    # 5v5 ONLY: run-1 of this backtest summed PP/PK window credits into the
    # actuals (RMSE ballooned for ALL projectors incl. Marcel — the signature
    # of an actuals-definition bug, not a model failure). v0 predicts 5v5;
    # actuals must be 5v5.
    is55 = po.strength_global.astype(str).str.replace("_", "").str.lower() \
        .isin(["5v5", "5v55v5"])
    po = po[is55].drop(columns=["strength_global"])
    frames = []
    for d in sorted(glob.glob(str(ART / "mp_attach_*"))):
        if not Path(d).is_dir() or not Path(d).name.split("_")[-1].isdigit():
            continue                          # skips mp_attach_all.log etc.
        yr = int(Path(d).name.split("_")[-1])
        files = glob.glob(f"{d}/mp_pw_*.parquet")
        if files:
            mp = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            mp["season"] = yr * 10000 + (yr + 1)
            frames.append(mp)
    mp = pd.concat(frames, ignore_index=True)
    j = po.merge(mp[["gamePk", "window_id", "playerId", "mp_xgf", "mp_xga"]],
                 on=["gamePk", "window_id", "playerId"], how="left")
    j[["mp_xgf", "mp_xga"]] = j[["mp_xgf", "mp_xga"]].fillna(0.0)
    pg = j.groupby(["season", "gamePk", "playerId"], as_index=False).agg(
        toi_sec=("seconds", "sum"), y_xgf_onice_w=("mp_xgf", "sum"),
        y_xga_onice_w=("mp_xga", "sum"), date=("date", "first"))
    team = (j.groupby(["gamePk", "playerId", "teamId"], as_index=False)["seconds"]
             .sum().sort_values("seconds").groupby(["gamePk", "playerId"]).tail(1))
    pg = pg.merge(team[["gamePk", "playerId", "teamId"]], on=["gamePk", "playerId"])
    pg["strength_global"] = "5v5"
    pg["date"] = pd.to_datetime(pg["date"])
    return pg


def detect_moves(pg: pd.DataFrame, start="2019-07-01", end="2025-10-01"):
    ev = pg[["playerId", "teamId", "date", "season"]].drop_duplicates() \
        .sort_values(["playerId", "date"])
    ev["prev_team"] = ev.groupby("playerId")["teamId"].shift(1)
    ev["prev_season"] = ev.groupby("playerId")["season"].shift(1)
    mv = ev[(ev.prev_team.notna()) & (ev.teamId != ev.prev_team)
            & (ev.date >= start) & (ev.date <= end)]
    mv = mv.rename(columns={"prev_team": "from_team", "teamId": "to_team",
                            "date": "move_date", "playerId": "player_id"})
    mv["move_type"] = np.where(mv.season == mv.prev_season, "trade", "offseason")
    return mv[["player_id", "move_date", "from_team", "to_team", "move_type"]]


if __name__ == "__main__":
    print("building MP player-game table...")
    pg = build_mp_playergame()
    print(f"{len(pg):,} player-games, seasons {pg.season.min()}..{pg.season.max()}")
    pg.to_parquet(ART / "mp_playergame.parquet", index=False)

    ledger = detect_moves(pg)
    q = qualify(ledger, pg, 1000.0, 20)
    q.to_parquet(ART / "ledger_extended.parquet", index=False)
    print(f"extended ledger: {len(q)} qualifying moves "
          f"({q.move_type.value_counts().to_dict()})")
    by_yr = q.move_date.astype(str).str[:4].value_counts().sort_index()
    print("moves by year:", by_yr.to_dict())

    windows = pd.read_parquet(ART / "perfect_windows_v3.parquet")
    if "date" not in windows.columns:                  # merge-suffix casualty guard
        for cand in ("date_x", "date_y"):
            if cand in windows.columns:
                windows = windows.rename(columns={cand: "date"})
                break
    windows["date"] = pd.to_datetime(windows["date"])
    windows = windows[windows.mu_xgf60.notna()]        # env needs OOF mu
    lines = pd.read_parquet(ART / "lines.parquet")
    asof = pd.read_parquet(ART / "talent_asof.parquet")
    K = json.loads((ART / "talent_K.json").read_text())["K_minutes"]

    print("running frozen-clock backtest (v0 + marcel + carryover, MP actuals)...")
    bt = run(q, pg, windows, lines, asof, K)
    bt.to_parquet(ART / "backtest_final.parquet", index=False)
    h = headline(bt)
    print(json.dumps(h, indent=2))
    for col in ["move_type"]:
        sl = bt.dropna(subset=["pred_v0"]).groupby(col).apply(
            lambda g: float(np.sqrt(((g.pred_v0 - g.actual_xgf_pct) ** 2).mean())))
        print(f"RMSE by {col}:", sl.round(2).to_dict())
    yr = bt.dropna(subset=["pred_v0"]).copy()
    yr["year"] = yr.move_date.astype(str).str[:4]
    print("moves scored by year:", yr.year.value_counts().sort_index().to_dict())

    cb = ConformalBands(target=0.80).fit(bt.dropna(subset=["pred_v0"]),
                                         pred_col="pred_v0")
    cb.save(str(ART / "conformal_bands_final.json"))
    print("conformal coverage per bin:", {k: round(v, 3) for k, v in cb.coverage_.items()})
    print("FINALBT_DONE")
