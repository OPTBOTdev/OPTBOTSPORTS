"""Build the trade/UFA ledger 2021-2025.

Two-source strategy:
  1. data/ledger_manual.csv — curated rows (always win on conflict). Start here:
     the ~60 biggest trades + July-1 signings are 2 hours of manual entry and
     are the moves a GM will ask about by name in the meeting.
  2. Automated detection from OUR OWN data: a player whose teamId changes
     mid-season (trade) or across seasons (signing) in the window spine.
     This guarantees the ledger agrees with the data it is scored on.

Usage: python scripts/01_build_ledger.py [--min-minutes 1000] [--min-post-gp 20]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.data.ledger import qualify  # noqa: E402

SPINE = r"D:\optbot\artifacts\window_spine_dedup.parquet"
OBS = r"D:\optbot\artifacts\player_game_obs_rebuilt.parquet"
MANUAL = Path(__file__).resolve().parents[1] / "data" / "ledger_manual.csv"
OUT = r"D:\optbot\artifacts\ledger.parquet"


def detect_moves(spine: pd.DataFrame, first_season: int = 20212022) -> pd.DataFrame:
    ev = spine[["playerId", "teamId", "date", "season"]].drop_duplicates()
    ev = ev.sort_values(["playerId", "date"])
    ev["prev_team"] = ev.groupby("playerId")["teamId"].shift(1)
    ev["prev_season"] = ev.groupby("playerId")["season"].shift(1)
    mv = ev[(ev.prev_team.notna()) & (ev.teamId != ev.prev_team)
            & (ev.season >= first_season)]
    mv = mv.rename(columns={"prev_team": "from_team", "teamId": "to_team",
                            "date": "move_date", "playerId": "player_id"})
    mv["move_type"] = (mv.season == mv.prev_season).map({True: "trade", False: "offseason"})
    return mv[["player_id", "move_date", "from_team", "to_team", "move_type"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-minutes", type=float, default=1000)
    ap.add_argument("--min-post-gp", type=int, default=20)
    args = ap.parse_args()

    spine = pd.read_parquet(SPINE, columns=["playerId", "teamId", "date", "season"])
    auto = detect_moves(spine)
    if MANUAL.exists():
        manual = pd.read_csv(MANUAL)
        auto = auto[~auto.set_index(["player_id", "move_date"]).index.isin(
            manual.set_index(["player_id", "move_date"]).index)]
        ledger = pd.concat([manual, auto], ignore_index=True)
        print(f"manual: {len(manual)}  auto: {len(auto)}")
    else:
        ledger = auto
        print(f"auto only: {len(auto)}  (add data/ledger_manual.csv for curated names)")

    obs = pd.read_parquet(OBS)
    q = qualify(ledger, obs, args.min_minutes, args.min_post_gp)
    q.to_parquet(OUT, index=False)
    print(f"qualifying moves: {len(q)}  -> {OUT}")
    print(q.groupby("move_type").size())
