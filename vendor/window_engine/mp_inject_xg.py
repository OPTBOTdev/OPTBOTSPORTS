"""Inject MoneyPuck xGoal into our scored-shots files (the MP-xG switch).

Strategy: our scored files keep ALL features (style counts, slots, etc. feed the
fill step unchanged); ONLY the per-shot `xg` value is replaced by MP's xGoal,
matched per shot on (gamePk, shooterId, |time - sec_game| <= tol). Unmatched
shots keep our in-house xg (counted + reported). Original xg preserved in
`xg_inhouse`; provenance column `xg_source` added.

Usage: python mp_inject_xg.py --year 2024 [--scored-dir D:/2024/shots_final]
                              [--tol 3] [--dry-run N]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_mp(year: int) -> pd.DataFrame:
    mp = pd.read_csv(rf"D:\shots_{year}.csv",
                     usecols=["game_id", "shooterPlayerId", "time", "period",
                              "xGoal", "goal"])
    mp = mp.dropna(subset=["shooterPlayerId"])
    mp["gamePk"] = (year * 1_000_000 + mp.game_id).astype("int64")
    mp["shooterPlayerId"] = mp.shooterPlayerId.astype("int64")
    return mp


def inject_game(fp: str, mp_g: pd.DataFrame, tol: int) -> tuple[int, int]:
    df = pd.read_csv(fp)
    if "xg_source" in df.columns:      # already injected
        return -1, -1
    df["xg_inhouse"] = df["xg"]
    matched = 0
    mp_by_shooter = {pid: g[["time", "xGoal"]].values
                     for pid, g in mp_g.groupby("shooterPlayerId")}
    xg_new = df["xg"].to_numpy(dtype=float).copy()
    src = np.array(["inhouse"] * len(df), dtype=object)
    sid = df["shooterId"].to_numpy()
    sec = df["sec_game"].to_numpy(dtype=float)
    used = {pid: np.zeros(len(v), dtype=bool) for pid, v in mp_by_shooter.items()}
    for i in range(len(df)):
        arr = mp_by_shooter.get(sid[i])
        if arr is None:
            continue
        d = np.abs(arr[:, 0] - sec[i])
        d[used[sid[i]]] = 1e9                # one MP shot matches one of ours
        j = int(np.argmin(d))
        if d[j] <= tol:
            xg_new[i] = arr[j, 1]
            src[i] = "moneypuck"
            used[sid[i]][j] = True
            matched += 1
    df["xg"] = xg_new
    df["xg_source"] = src
    df.to_csv(fp, index=False)
    return matched, len(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--scored-dir", default=None)
    ap.add_argument("--tol", type=int, default=3)
    ap.add_argument("--dry-run", type=int, default=None,
                    help="only N games, report match rate, DO NOT write")
    args = ap.parse_args()
    sd = args.scored_dir or rf"D:\{args.year}\shots_final"
    files = sorted(glob.glob(os.path.join(sd, "shots_train_*_scored.csv")))
    files = [f for f in files if "backup" not in f]
    mp = load_mp(args.year)
    tot_m = tot_n = games = 0
    for i, fp in enumerate(files[: args.dry_run] if args.dry_run else files):
        g = int(os.path.basename(fp).split("_")[2])
        mp_g = mp[mp.gamePk == g]
        if args.dry_run:
            df = pd.read_csv(fp)
            m, n = 0, len(df)
            mbs = {pid: gg[["time", "xGoal"]].values
                   for pid, gg in mp_g.groupby("shooterPlayerId")}
            for sid_v, sec_v in zip(df.shooterId, df.sec_game):
                arr = mbs.get(sid_v)
                if arr is not None and np.abs(arr[:, 0] - sec_v).min() <= args.tol:
                    m += 1
        else:
            m, n = inject_game(fp, mp_g, args.tol)
            if m < 0:
                continue
        tot_m += m
        tot_n += n
        games += 1
        if games % 200 == 0:
            print(f"{games} games | running match {tot_m/max(tot_n,1):.2%}", flush=True)
    print(f"{'DRY ' if args.dry_run else ''}DONE year={args.year} games={games} "
          f"match={tot_m}/{tot_n} ({tot_m/max(tot_n,1):.2%})")
