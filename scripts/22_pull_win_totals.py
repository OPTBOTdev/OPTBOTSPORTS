"""Script 22 — season points-total futures (the roster-churn market), from Covers.

Pulls the archived preseason O/U points line + juice + actual result for every
team, per season. This is the cleanest backtestable market for our exact edge:
the line is set on trailing reputation; our roster-churn xG delta knows what
the trailing stats don't.

Output: artifacts/win_totals.parquet
Usage: python scripts/22_pull_win_totals.py [--first 2018] [--last 2026]
"""
import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests

ART = Path(r"D:\optbot\artifacts")
URL = "https://www.covers.com/sportsoddshistory/nhl-win/?y={y}&sa=nhl&t=pts"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def pull_season(y0: int) -> pd.DataFrame:
    y = f"{y0}-{y0+1}"
    r = requests.get(URL.format(y=y), headers=UA, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    best = max(tables, key=len)
    best.columns = [str(c).strip().lower().replace(" ", "_") for c in best.columns]
    best["season"] = y0 * 10000 + y0 + 1          # 20242025 style
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2018)
    ap.add_argument("--last", type=int, default=2026)
    args = ap.parse_args()
    parts = []
    for y0 in range(args.first, args.last + 1):
        try:
            t = pull_season(y0)
            parts.append(t)
            print(f"{y0}-{y0+1}: {len(t)} rows | cols: {t.columns.tolist()}")
        except Exception as e:                      # noqa: BLE001 — survey pull
            print(f"{y0}-{y0+1}: FAILED ({e})")
        time.sleep(1.0)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(ART / "win_totals.parquet", index=False)
    print(f"wrote win_totals.parquet: {len(df)} team-seasons")
