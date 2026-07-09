"""ZONE & CONTEXT DEPTH AUDIT — 'what zone does a player LIVE in during long windows?'

Uses local pbp_onice (event zones + on-ice + sec_game) vs our windows, full season.

  Z1 ZONE-INFO DECAY   how fast does zone_start stop describing reality? Event-zone
                       agreement with the starting zone, bucketed by seconds since
                       window start. THE test of whether zone context is honest.
  Z2 GOAL-ENDS-WINDOW  distance from each goal to its window's end (score constancy)
  Z3 MID-WINDOW RESETS faceoffs occurring strictly inside windows (context resets
                       our start-features never see), by window-length bucket
  Z4 RESIDENCE         in windows >120s: share of event-activity in the started
                       zone — how misleading zone_start is for long windows
  Z5 BOUNDARY LOCK     % of window starts that coincide with an official faceoff
                       second; % of ends on stoppage/goal/period/shift-change events
  Z6 THIRD REFEREE     at every goal: pbp_onice's own on-ice set vs our GF-credited
                       set (independent of shiftcharts used in script 10)

Usage: python scripts/11_zone_context_audit.py [--year 2023] [--limit N]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

API = Path(r"C:\Users\lilli\Downloads\API\API\Final")
WIN = {2021: r"D:\2021", 2022: r"D:\2022", 2023: r"D:\2023", 2024: r"D:\2024"}
ART = Path(r"D:\optbot\artifacts")
BUCKETS = [(0, 10), (10, 30), (30, 60), (60, 120), (120, 300), (300, 9999)]


def load_onice(year, tag, gamePk):
    f = API / str(year) / "raw" / f"pbp_built_{tag}" / f"pbp_onice_{gamePk}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def zscore(zone, ev_team, ref_team):
    """+1 = offensive zone for ref_team, -1 = defensive, 0 = neutral."""
    if zone not in ("O", "D", "N"):
        return None
    v = {"O": 1, "D": -1, "N": 0}[zone]
    return v if ev_team == ref_team else -v


def audit_game(year, tag, gamePk, acc):
    wf = Path(WIN[year]) / "final_windows" / f"player_windows_train_{gamePk}_xg.csv"
    if not wf.exists():
        return
    ours = pd.read_csv(wf, usecols=["playerId", "teamId", "window_id", "period",
                                    "start_sec", "end_sec", "duration", "seconds",
                                    "strength_global", "zone_start", "GF"])
    ours = ours[ours.period <= 3]
    d = load_onice(year, tag, gamePk)
    if d is None:
        return
    home_id = d["home"]["teamId"]
    evs = [e for e in d["events"] if e.get("period", 9) <= 3 and "sec_game" in e]

    is55 = ours.strength_global.astype(str).str.lower().str.replace("_", "") \
        .isin(["5v5", "5v55v5"])
    w = ours[is55]
    wins = w.drop_duplicates(["window_id", "teamId"])
    # one reference row per window per team, with that team's zone_start
    zmap = {"OZ": 1, "DZ": -1, "NZ": 0, "O": 1, "D": -1, "N": 0}
    wlist = [(r.window_id, r.teamId, r.start_sec, r.end_sec,
              zmap.get(str(r.zone_start), None), r.end_sec - r.start_sec)
             for r in wins.itertuples()]
    # index windows by home team only (avoid double counting; away is mirror)
    wlist_h = [x for x in wlist if x[1] == home_id]

    ZONED = ("shot-on-goal", "goal", "missed-shot", "blocked-shot", "hit",
             "giveaway", "takeaway", "faceoff")
    for e in evs:
        et = e.get("type")
        if et not in ZONED:
            continue
        det = e.get("details", {}) or {}
        z = det.get("zoneCode")
        team = det.get("eventOwnerTeamId")
        t = e["sec_game"]
        for wid, wteam, ws, we, z0, dur in wlist_h:
            if ws < t < we:
                if et == "faceoff" and t > ws + 1:
                    acc["z3"].append(dur)                       # mid-window reset
                zs = zscore(z, team, wteam)
                if zs is not None and z0 is not None and z0 != 0:
                    age = t - ws
                    for lo, hi in BUCKETS:
                        if lo <= age < hi:
                            acc["z1"].append((lo, z0 * zs))     # agreement w/ start zone
                            break
                    if dur > 120:
                        acc["z4"].append((wid, z0 * zs))
                break

    # Z2 + Z6: goals
    for e in evs:
        if e.get("type") != "goal":
            continue
        det = e.get("details", {}) or {}
        t = e["sec_game"]
        team = det.get("eventOwnerTeamId")
        cand = w[(w.teamId == team) & (w.start_sec <= t) & (w.end_sec >= t)]
        if len(cand):
            acc["z2"].append(float(cand.end_sec.iloc[0] - t))
            onice = e.get("onice") or {}
            side = "home" if team == home_id else "away"
            official = set(onice.get(side, []))
            credited = set(cand[cand.GF > 0].playerId)
            if official:
                acc["z6_exact"].append(credited == (official & set(ours.playerId)))
    acc["z5_starts"].append(0)  # placeholder increments below
    # Z5: window starts on faceoff seconds
    fo_secs = {e["sec_game"] for e in evs if e.get("type") == "faceoff"}
    stops = {e["sec_game"] for e in evs if e.get("type") in
             ("stoppage", "goal", "penalty", "period-end", "shift_change")}
    for wid, wteam, ws, we, z0, dur in wlist_h:
        acc["z5_start_fo"].append(ws in fo_secs or (ws + 1) in fo_secs or (ws - 1) in fo_secs)
        acc["z5_end_stop"].append(we in stops or (we + 1) in stops or (we - 1) in stops)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = f"{args.year}{args.year + 1}"
    files = sorted(glob.glob(str(API / str(args.year) / "raw" / f"pbp_built_{tag}"
                                 / "pbp_onice_*.json")))
    if args.limit:
        files = files[: args.limit]
    acc = {"z1": [], "z2": [], "z3": [], "z4": [], "z5_start_fo": [],
           "z5_end_stop": [], "z5_starts": [], "z6_exact": []}
    for i, f in enumerate(files):
        gamePk = int(Path(f).stem.split("_")[-1])
        try:
            audit_game(args.year, tag, gamePk, acc)
        except Exception as e:
            print(f"{gamePk} FAILED: {type(e).__name__}: {e}")
        if (i + 1) % 150 == 0:
            print(f"{i+1}/{len(files)}")

    print("\n=========== ZONE & CONTEXT DEPTH AUDIT ===========")
    z1 = pd.DataFrame(acc["z1"], columns=["bucket", "agree"])
    print("Z1 ZONE-INFO DECAY (mean agreement of event-zone with zone_start; "
          "+1 perfect, 0 = no info left):")
    for lo, hi in BUCKETS:
        m = z1[z1.bucket == lo]
        if len(m):
            print(f"   {lo:>3}-{hi:<4}s after window start: {m.agree.mean():+.3f}  (n={len(m):,})")
    z2 = np.array(acc["z2"])
    print(f"Z2 GOAL->WINDOW-END distance: median {np.median(z2):.0f}s | "
          f"<=2s {np.mean(z2 <= 2):.1%} | p95 {np.percentile(z2, 95):.0f}s")
    z3 = np.array(acc["z3"])
    tot_w = len(acc["z5_start_fo"])
    print(f"Z3 MID-WINDOW FACEOFFS: {len(z3):,} resets inside {tot_w:,} windows "
          f"({len(z3)/max(tot_w,1):.2f}/window) | in windows >120s: "
          f"{np.mean(z3 > 120):.1%} of resets")
    z4 = pd.DataFrame(acc["z4"], columns=["wid", "agree"])
    if len(z4):
        per_w = z4.groupby("wid")["agree"].mean()
        print(f"Z4 LONG-WINDOW RESIDENCE (>120s, {len(per_w):,} windows): "
              f"mean start-zone agreement {per_w.mean():+.3f} | "
              f"windows where activity NET-OPPOSED start zone: {(per_w < 0).mean():.1%}")
    print(f"Z5 BOUNDARY LOCK: starts on faceoff {np.mean(acc['z5_start_fo']):.1%} | "
          f"ends on stoppage-class event {np.mean(acc['z5_end_stop']):.1%}")
    print(f"Z6 THIRD REFEREE (pbp_onice at goals): exact credited-set match "
          f"{np.mean(acc['z6_exact']):.1%}  (n={len(acc['z6_exact']):,})")
    pd.to_pickle(acc, ART / f"zone_audit_{args.year}.pkl")
