#!/usr/bin/env python3
"""
Populate `API/Final/<year>` for an NHL season and (optionally) build "windows".

This script is meant to be the "get to the point" entrypoint:
1) Build standings + game_results for a season (REG only, completed games only)
2) Dump per-game raw JSON + wide CSVs into `API/Final/<year>/{raw,csv}/...`
3) Build per-game windows via `build_windows_and_shots.py` into
   `API/Final/<year>/derived/windows/`

Examples
--------
Populate 2025-26 season (to-date):
  python nhl_make_season_final.py --season 20252026

Populate + build windows:
  python nhl_make_season_final.py --season 20252026 --windows

Resume safely (skip anything already downloaded/built):
  python nhl_make_season_final.py --season 20252026 --windows --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Iterable, List, Optional


def _run(cmd: List[str]) -> int:
    # Stream output so user can see progress in real time.
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd)


def _season_to_year(season: str) -> str:
    if not (isinstance(season, str) and len(season) == 8 and season.isdigit()):
        raise ValueError("season must be 8 digits like 20252026")
    return season[:4]


def _read_game_ids_from_game_results_csv(path: str) -> List[int]:
    ids: List[int] = []
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            gid = row.get("gamePk") or row.get("game_id") or row.get("id")
            try:
                ids.append(int(gid))
            except Exception:
                continue
    # stable unique
    return sorted(set(ids))


def _chunks(xs: List[int], n: int) -> Iterable[List[int]]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Populate API/Final/<year> for a season, and optionally build windows.")
    ap.add_argument("--season", required=True, help="Season like 20252026")
    ap.add_argument("--windows", action="store_true", help="Also build windows outputs for each completed game.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads/builds when output files already exist.")
    ap.add_argument("--max-games", type=int, default=0, help="If >0, limit to first N completed games (for quick tests).")
    ap.add_argument("--pause", type=float, default=0.0, help="Optional sleep seconds between games (0 for none).")
    args = ap.parse_args(argv)

    season = args.season
    year = _season_to_year(season)

    out_root = os.path.join("API", "Final", year)
    raw_dir = os.path.join(out_root, "raw")
    csv_dir = os.path.join(out_root, "csv")
    derived_dir = os.path.join(out_root, "derived")
    windows_dir = os.path.join(derived_dir, "windows")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(derived_dir, exist_ok=True)
    if args.windows:
        os.makedirs(windows_dir, exist_ok=True)

    # 1) Build season standings + game results (completed REG games only)
    rc = _run([sys.executable, "rankings.py", "--season", season, "--out", out_root])
    if rc != 0:
        print("Failed to build standings/game_results. Aborting.", file=sys.stderr)
        return rc

    game_results_csv = os.path.join(out_root, f"game_results_{season}.csv")
    game_ids = _read_game_ids_from_game_results_csv(game_results_csv)
    if not game_ids:
        print(f"No completed games found in {game_results_csv}. Nothing to do.")
        return 0

    if args.max_games and args.max_games > 0:
        game_ids = game_ids[: args.max_games]

    print(f"Season {season}: {len(game_ids)} completed REG games found.", flush=True)

    # 2) Dump raw+csv for each game
    for i, gid in enumerate(game_ids, start=1):
        pbp_raw = os.path.join(raw_dir, "pbp", f"{gid}.json")
        shifts_raw = os.path.join(raw_dir, "shiftcharts", f"{gid}.json")
        box_raw = os.path.join(raw_dir, "boxscore", f"{gid}.json")

        need_dump = True
        if args.skip_existing:
            # Only skip if ALL core raw files exist.
            need_dump = not (os.path.exists(pbp_raw) and os.path.exists(shifts_raw) and os.path.exists(box_raw))

        if need_dump:
            print(f"[{i}/{len(game_ids)}] dump game {gid}", flush=True)
            rc = _run([sys.executable, "nhl_dump_everything.py", "--game", str(gid), "--out", out_root, "--dump-raw"])
            if rc != 0:
                print(f"  Warning: dumper failed for game {gid} (rc={rc}); continuing.", file=sys.stderr)
        else:
            print(f"[{i}/{len(game_ids)}] skip dump (exists) game {gid}", flush=True)

        # 3) Build windows (optional)
        if args.windows:
            out_panel = os.path.join(windows_dir, f"panel_windows_{gid}.csv")
            out_shots = os.path.join(windows_dir, f"shots_master_{gid}.csv")
            need_windows = True
            if args.skip_existing:
                need_windows = not (os.path.exists(out_panel) and os.path.exists(out_shots))
            if need_windows:
                print(f"  build windows {gid}", flush=True)
                rc = _run(
                    [
                        sys.executable,
                        "build_windows_and_shots.py",
                        "--game",
                        str(gid),
                        "--raw",
                        raw_dir,
                        "--out",
                        windows_dir,
                        "--season",
                        season,
                    ]
                )
                if rc != 0:
                    print(f"  Warning: windows builder failed for game {gid} (rc={rc}); continuing.", file=sys.stderr)
            else:
                print(f"  skip windows (exists) {gid}", flush=True)

        if args.pause and args.pause > 0:
            import time

            time.sleep(args.pause)

    print(f"Done. Outputs under: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





