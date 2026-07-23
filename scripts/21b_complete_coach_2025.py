"""Script 21b — complete the coach table for 2025-26 and rebuild tenure.

The orphan game_coach_ids_2017_2025.csv stopped at 616 of 1,312 games for the
2025-26 season (it was built mid-season). This fetches head coaches for the
missing regular-season games from api-web right-rail (same source as the
original builder), takes team ids from our own boxscore JSONs, and rebuilds
artifacts/coach_table.parquet with tenure recomputed over the FULL history.

Usage: python scripts/21b_complete_coach_2025.py
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

ART = Path(r"D:\optbot\artifacts")
SRC = Path(r"C:\Users\lilli\Downloads\API\artifacts\game_coach_ids_2017_2025.csv")
BOX = Path(r"C:\Users\lilli\Downloads\API\API\Final\2025\raw\boxscore")
APIWEB = "https://api-web.nhle.com/v1"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_coaches(sess, game_pk):
    for attempt in range(4):
        r = sess.get(f"{APIWEB}/gamecenter/{game_pk}/right-rail", timeout=15)
        if r.status_code == 200:
            break
        time.sleep(2.0 * (attempt + 1))          # 429/5xx backoff
    else:
        return None, None
    gi = r.json().get("gameInfo") or {}

    def name(side):
        hc = (gi.get(side) or {}).get("headCoach")
        if isinstance(hc, dict):
            hc = hc.get("default")
        return hc.strip() if isinstance(hc, str) and hc.strip() else None
    return name("awayTeam"), name("homeTeam")


if __name__ == "__main__":
    done = ART / "game_coach_ids_completed.csv"
    w = pd.read_csv(done if done.exists() else SRC)   # resume from own output
    have = set(w[w.gamePk.astype(str).str.startswith("2025")].gamePk)
    all_2025 = sorted(int(p.stem) for p in BOX.glob("2025*.json")
                      if p.stem[4:6] == "02")          # regular season only
    missing = [g for g in all_2025 if g not in have]
    print(f"2025-26: {len(have)} present, {len(missing)} missing of {len(all_2025)}")

    sess = requests.Session()
    rows, fails = [], []
    for i, g in enumerate(missing):
        try:
            box = json.load(open(BOX / f"{g}.json", encoding="utf-8"))
            an, hn = fetch_coaches(sess, g)
            if not (an and hn):
                fails.append(g)
                continue
            rows.append({"season": 20252026, "gamePk": g,
                         "date": box.get("gameDate"),
                         "away_id": box["awayTeam"]["id"],
                         "away_headCoachName": an, "away_headCoachKey": slugify(an),
                         "home_id": box["homeTeam"]["id"],
                         "home_headCoachName": hn, "home_headCoachKey": slugify(hn)})
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            fails.append(g)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(missing)} ({len(fails)} fails)")
        time.sleep(0.35)

    new = pd.DataFrame(rows)
    combined = pd.concat([w, new], ignore_index=True)
    combined.to_csv(ART / "game_coach_ids_completed.csv", index=False)
    print(f"combined: {len(combined):,} games ({len(new)} newly fetched, "
          f"{len(fails)} fails: {fails[:10]})")

    c = pd.concat([
        combined[["gamePk", "home_id", "home_headCoachKey"]].rename(
            columns={"home_id": "teamId", "home_headCoachKey": "coach_id"}),
        combined[["gamePk", "away_id", "away_headCoachKey"]].rename(
            columns={"away_id": "teamId", "away_headCoachKey": "coach_id"}),
    ], ignore_index=True).dropna(subset=["coach_id"])
    c = c.sort_values(["teamId", "gamePk"])
    c["coach_tenure_games"] = c.groupby(["teamId", "coach_id"]).cumcount() + 1
    c[["gamePk", "teamId", "coach_id", "coach_tenure_games"]].to_parquet(
        ART / "coach_table.parquet", index=False)
    per = c.gamePk.astype(str).str[:4].value_counts().sort_index()
    print(f"coach_table rebuilt: {c.coach_id.nunique()} coaches, "
          f"{len(c):,} team-games\n{per}")
