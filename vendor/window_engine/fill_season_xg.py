"""Season driver for the xG-fill chain — the step that never had one (F1).

Per game: build_training_from_raw -> score_xg (D:/XG models) -> fill_player_windows_xg
Produces player_windows_train_<g>_xg.csv next to the train CSVs.

Usage: python fill_season_xg.py --year 2025 [--limit N]
Idempotent: skips games whose _xg.csv already exists.
"""
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True).returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models-root", default="D:/XG/artifacts_xg")
    args = ap.parse_args()
    root = os.path.join(HERE, "API", "Final", str(args.year))
    win_dir = os.path.join(root, "derived", "windows")
    shots_dir = os.path.join(root, "derived", "shots")
    os.makedirs(shots_dir, exist_ok=True)
    season_tag = f"{args.year}{args.year + 1}"
    pbpice = os.path.join(root, "raw", "pbpice")
    if not os.path.isdir(pbpice):
        pbpice = os.path.join(root, "raw", f"pbp_built_{season_tag}")

    trains = sorted(glob.glob(os.path.join(win_dir, "player_windows_train_*.csv")))
    trains = [t for t in trains if not t.endswith("_xg.csv")]
    if args.limit:
        trains = trains[: args.limit]
    done = fail = skip = 0
    for i, t in enumerate(trains):
        g = os.path.basename(t).split("_")[3].split(".")[0]
        out_xg = t.replace(".csv", "_xg.csv")
        if os.path.exists(out_xg):
            skip += 1
            continue
        shots = os.path.join(shots_dir, f"shots_train_{g}.csv")
        scored = shots.replace(".csv", "_scored.csv")
        onice = os.path.join(pbpice, f"pbp_onice_{g}.json")
        steps = []
        if not os.path.exists(scored):
            if not os.path.exists(shots):
                steps.append([sys.executable, "build_training_from_raw.py", "--game", g,
                              "--raw", os.path.join(root, "raw"), "--out", shots_dir])
            steps.append([sys.executable, "score_xg.py", "--shots", shots,
                          "--models_root", args.models_root, "--out", scored])
        steps.append([sys.executable, "fill_player_windows_xg.py", "--windows", t,
                      "--shots", scored, "--onice", onice, "--out", out_xg])
        rc = 0
        for s in steps:
            rc = run(s)
            if rc != 0:
                break
        if rc == 0 and os.path.exists(out_xg):
            done += 1
        else:
            fail += 1
            print(f"[{i+1}/{len(trains)}] {g}: FAILED (rc={rc})", flush=True)
        if (done + fail) % 100 == 0:
            print(f"[{i+1}/{len(trains)}] done={done} fail={fail} skip={skip}", flush=True)
    print(f"FILL_DONE done={done} fail={fail} skip={skip}")
