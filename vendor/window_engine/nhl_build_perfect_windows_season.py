#!/usr/bin/env python3
"""
Build "perfect windows" outputs for an NHL season using local API/Final dumps.

Pipeline per gamePk:
  1) build_pbp_with_onice.py  -> raw/pbpice/pbp_onice_<gamePk>.json
  2) perfect_windows.py       -> derived/windows/{windows,team_windows,player_windows,player_windows_train}_<gamePk>.*

Designed for resume/skip-existing.

Example (2025-26 season data lives in API/Final/2025):
  python nhl_build_perfect_windows_season.py --final-root API/Final/2025
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import List, Optional


def _read_gamepks(game_results_csv: str) -> List[int]:
    gamepks: List[int] = []
    if not os.path.exists(game_results_csv):
        return gamepks
    with open(game_results_csv, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Try common column names
            for k in ("gamePk", "gamepk", "GamePk", "game_id", "gameId"):
                if k in row and row[k]:
                    try:
                        gamepks.append(int(float(row[k])))
                    except Exception:
                        pass
                    break
    # De-dupe preserving order
    seen = set()
    out: List[int] = []
    for g in gamepks:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _run(cmd: List[str]) -> int:
    p = subprocess.run(cmd)
    return int(p.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build perfect_windows outputs for all games in an NHL season.")
    ap.add_argument("--final-root", default=os.path.join("API", "Final", "2025"), help="Season root like API/Final/2025")
    ap.add_argument("--raw-dir", default=None, help="Override raw dir (default: <final-root>/raw)")
    ap.add_argument("--standings-dir", default=None, help="Directory containing game_results_*.csv (default: <final-root>)")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: <final-root>/derived/windows)")
    ap.add_argument("--pbpice-dir", default=None, help="Where to write pbp_onice_*.json (default: <raw-dir>/pbpice)")
    ap.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N games (0 = all)")
    ap.add_argument("--start-at", type=int, default=0, help="Skip gamePks < this value")
    ap.add_argument("--only-missing", action="store_true", help="Only process games missing player_windows_train_<gamePk>.csv")
    ap.add_argument("--csv-only-player-train", action="store_true", help="Tell perfect_windows to only write player_windows_train_<gamePk>.csv")
    args = ap.parse_args(argv)

    final_root = str(args.final_root)
    raw_dir = str(args.raw_dir or os.path.join(final_root, "raw"))
    standings_dir = str(args.standings_dir or final_root)
    out_dir = str(args.out_dir or os.path.join(final_root, "derived", "windows"))
    pbpice_dir = str(args.pbpice_dir or os.path.join(raw_dir, "pbpice"))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(pbpice_dir, exist_ok=True)

    # Find season game results file inside standings dir
    game_results = None
    for fn in os.listdir(standings_dir) if os.path.isdir(standings_dir) else []:
        if fn.startswith("game_results_") and fn.endswith(".csv"):
            game_results = os.path.join(standings_dir, fn)
            break
    if not game_results:
        raise SystemExit(f"Could not find game_results_*.csv in {standings_dir}")

    gamepks = _read_gamepks(game_results)
    gamepks = [g for g in gamepks if g >= int(args.start_at)]
    if args.limit and args.limit > 0:
        gamepks = gamepks[: int(args.limit)]

    if not gamepks:
        print("No gamePks found.", flush=True)
        return 1

    print(f"Found {len(gamepks)} gamePks in {os.path.basename(game_results)}", flush=True)
    print(f"raw_dir={raw_dir}", flush=True)
    print(f"pbpice_dir={pbpice_dir}", flush=True)
    print(f"out_dir={out_dir}", flush=True)

    failures: List[int] = []
    built = skipped = 0

    for i, g in enumerate(gamepks, start=1):
        out_train = os.path.join(out_dir, f"player_windows_train_{g}.csv")
        if args.only_missing and os.path.exists(out_train) and not args.force:
            skipped += 1
            continue
        if (not args.force) and os.path.exists(out_train):
            skipped += 1
            continue

        pbp_onice_path = os.path.join(pbpice_dir, f"pbp_onice_{g}.json")
        if args.force or not os.path.exists(pbp_onice_path):
            rc = _run([sys.executable, "build_pbp_with_onice.py", "--game", str(g), "--raw", raw_dir, "--out", pbpice_dir])
            if rc != 0 or not os.path.exists(pbp_onice_path):
                print(f"[{i}/{len(gamepks)}] game {g}: FAILED build_pbp_with_onice", flush=True)
                failures.append(g)
                continue

        cmd = [
            sys.executable,
            "perfect_windows.py",
            "--in",
            pbp_onice_path,
            "--out_dir",
            out_dir,
            "--csv",
            "--standings_dir",
            standings_dir,
        ]
        if args.csv_only_player_train:
            cmd.append("--csv_only_player_train")
        rc2 = _run(cmd)
        if rc2 != 0 or (not os.path.exists(out_train)):
            print(f"[{i}/{len(gamepks)}] game {g}: FAILED perfect_windows", flush=True)
            failures.append(g)
            continue

        built += 1
        if built % 10 == 0:
            print(f"[progress] built={built} skipped={skipped} failures={len(failures)} (last={g})", flush=True)

    print(f"\nDone. built={built} skipped={skipped} failures={len(failures)}", flush=True)
    if failures:
        print("Failures:", failures[:50], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





