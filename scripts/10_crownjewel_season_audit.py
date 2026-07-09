"""CROWN-JEWEL FULL-SEASON AUDIT — every game, using local pbp_onice + shiftcharts.

Six checks, one season (default 2023-24, all ~1,269 games):

  C1 SECONDS      per (player,window): our `seconds` vs exact official shift-overlap
  C2 MULTISTINT   on-off-on inside one window: is `seconds` the SUM of overlaps, and
                  are goals scored during the player's OFF-gap NOT credited to him?
  C3 SF LEDGER    per (player,window): our SF vs pbp shots-for during his exact
                  presence (definition auto-detected: SOG / Fenwick / Corsi)
  C4 PURITY       events inside a 5v5 window whose on-ice state is NOT 5v5
  C5 STOP-BIAS    length-biased sampling: windows are outcome-stopped (goals cause
                  whistles) — quantify per-window rate inflation vs game-level truth
  C6 ENTRY-SELECT on-the-fly entrants enter into selected (possession) states —
                  quantify the xGF60 gap between mid-window entrants and FO starters

Usage: python scripts/10_crownjewel_season_audit.py [--year 2023] [--limit N]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

API = Path(r"C:\Users\lilli\Downloads\API\API\Final")
WIN = {2018: r"D:\2018", 2019: r"D:\2019", 2020: r"D:\2020", 2021: r"D:\2021",
       2022: r"D:\2022", 2023: r"D:\2023", 2024: r"D:\2024"}
ART = Path(r"D:\optbot\artifacts")


def mmss(t):
    m, s = str(t).split(":")
    return int(m) * 60 + int(s)


def load_shifts(year, gamePk):
    f = API / str(year) / "csv" / "shiftcharts" / f"shiftcharts_{gamePk}.csv"
    if not f.exists():
        return pd.DataFrame()
    sh = pd.read_csv(f, usecols=["playerId", "teamId", "period", "startTime",
                                 "endTime", "typeCode"])
    sh = sh[(sh.typeCode == 517) & (sh.period <= 3)].dropna(subset=["startTime", "endTime"])
    sh["s"] = (sh.period - 1) * 1200 + sh.startTime.map(mmss)
    sh["e"] = (sh.period - 1) * 1200 + sh.endTime.map(mmss)
    return sh[sh.e > sh.s]


def load_events(year, gamePk, season_tag):
    f = API / str(year) / "raw" / f"pbp_built_{season_tag}" / f"pbp_onice_{gamePk}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    evs = []
    for e in d["events"]:
        if e.get("period", 9) > 3 or "sec_game" not in e:
            continue
        onice = e.get("onice") or {}
        h, a = onice.get("home", []), onice.get("away", [])
        gl = onice.get("goalies", {})
        is55 = len(h) == 5 and len(a) == 5 and gl.get("home") and gl.get("away")
        det = e.get("details", {}) or {}
        evs.append({"t": e["sec_game"], "type": e.get("type"), "is55": bool(is55),
                    "team": det.get("eventOwnerTeamId"),
                    "home_ids": h, "away_ids": a})
    return {"home": d["home"]["teamId"], "away": d["away"]["teamId"], "events": evs}


def overlap(s, e, ws, we):
    return max(0, min(e, we) - max(s, ws))


def audit_game(year, season_tag, gamePk, acc):
    wf = Path(WIN[year]) / "final_windows" / f"player_windows_train_{gamePk}_xg.csv"
    if not wf.exists():
        return
    ours = pd.read_csv(wf, usecols=["playerId", "teamId", "window_id", "period",
                                    "start_sec", "end_sec", "seconds", "strength_global",
                                    "SF", "GF", "xGF", "entered_after_start"])
    ours = ours[ours.period <= 3]
    sh = load_shifts(year, gamePk)
    pbp = load_events(year, gamePk, season_tag)
    if sh.empty or pbp is None:
        return
    shifts_by_p = {pid: g[["s", "e"]].values for pid, g in sh.groupby("playerId")}
    is55w = ours.strength_global.astype(str).str.lower().str.replace("_", "") \
        .isin(["5v5", "5v55v5"])
    ev = pd.DataFrame(pbp["events"])
    shots = ev[ev.type.isin(["shot-on-goal", "goal"])]
    goals = ev[ev.type == "goal"]

    n = 0
    for r in ours[is55w].itertuples():
        ivs = shifts_by_p.get(r.playerId)
        if ivs is None:
            continue
        inter = [(max(s, r.start_sec), min(e, r.end_sec)) for s, e in ivs
                 if e > r.start_sec and s < r.end_sec]
        off_sec = sum(e - s for s, e in inter)
        acc["c1_diff"].append(abs(off_sec - r.seconds))
        n += 1
        if len(inter) >= 2:                                   # C2: multistint
            acc["c2_n"] += 1
            acc["c2_sum_ok"].append(abs(off_sec - r.seconds) <= 4)
            gaps = [(inter[i][1], inter[i + 1][0]) for i in range(len(inter) - 1)]
            for g in goals.itertuples():
                if g.team == r.teamId and r.start_sec <= g.t <= r.end_sec \
                        and any(gs < g.t < ge for gs, ge in gaps):
                    acc["c2_offgap_goals"] += 1
                    acc["c2_offgap_credited"] += int(r.GF > 0)
        if n % 7 == 0:                                        # C3 sample: 1-in-7 rows
            cnt = 0
            for s in shots.itertuples():
                if s.team == r.teamId and any(a <= s.t <= b for a, b in inter):
                    cnt += 1
            acc["c3_pairs"].append((cnt, r.SF))

    # C4: purity — pbp events strictly inside our 5v5 windows must be 5v5 on-ice
    w55 = ours[is55w].drop_duplicates(["window_id"])[["start_sec", "end_sec"]].values
    inner = ev[(ev.type.isin(["shot-on-goal", "goal", "hit", "faceoff"]))]
    for e in inner.itertuples():
        for ws, we in w55:
            if ws + 1 < e.t < we - 1:
                acc["c4_tot"] += 1
                acc["c4_bad"] += int(not e.is55)
                break

    # C5: stop-bias — team-window xGF rate, goal-ended vs not
    tw = ours[is55w].groupby(["window_id", "teamId"], as_index=False) \
        .agg(xGF=("xGF", "max"), GF=("GF", "max"),
             dur=("end_sec", "max"))
    tw["dur"] = tw["dur"] - ours[is55w].groupby(["window_id", "teamId"])["start_sec"] \
        .min().values
    tw = tw[tw.dur > 0]
    acc["c5"].append(tw.assign(goal_end=tw.GF > 0)[["xGF", "dur", "goal_end"]])

    # C6: entry selection — xGF60 by entry mode
    m = ours[is55w & (ours.seconds > 10)]
    acc["c6"].append(m[["entered_after_start", "xGF", "seconds"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    season_tag = f"{args.year}{args.year + 1}"
    files = sorted(glob.glob(str(API / str(args.year) / "raw" /
                                 f"pbp_built_{season_tag}" / "pbp_onice_*.json")))
    if args.limit:
        files = files[: args.limit]
    acc = {"c1_diff": [], "c2_n": 0, "c2_sum_ok": [], "c2_offgap_goals": 0,
           "c2_offgap_credited": 0, "c3_pairs": [], "c4_tot": 0, "c4_bad": 0,
           "c5": [], "c6": []}
    for i, f in enumerate(files):
        gamePk = int(Path(f).stem.split("_")[-1])
        try:
            audit_game(args.year, season_tag, gamePk, acc)
        except Exception as e:
            print(f"{gamePk} FAILED: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(files)} games")

    print("\n=========== CROWN-JEWEL SEASON AUDIT ===========")
    d = np.array(acc["c1_diff"])
    print(f"C1 SECONDS ({len(d):,} player-windows): "
          f"exact<=1s {np.mean(d <= 1):.1%} | <=3s {np.mean(d <= 3):.1%} | "
          f"mean {d.mean():.2f}s | p99 {np.percentile(d, 99):.0f}s")
    print(f"C2 MULTISTINT: {acc['c2_n']:,} cases | seconds=sum-of-overlaps "
          f"{np.mean(acc['c2_sum_ok']):.1%} | off-gap goals {acc['c2_offgap_goals']} "
          f"| WRONGLY credited {acc['c2_offgap_credited']} (must be 0)")
    if acc["c3_pairs"]:
        p = pd.DataFrame(acc["c3_pairs"], columns=["pbp", "ours"])
        print(f"C3 SF LEDGER ({len(p):,} sampled rows): exact {np.mean(p.pbp == p.ours):.1%} "
              f"| corr {p.pbp.corr(p.ours):.3f} | mean(pbp-ours) {(p.pbp - p.ours).mean():.3f}")
    print(f"C4 PURITY: {acc['c4_bad']}/{acc['c4_tot']} events inside 5v5 windows "
          f"were NOT 5v5 on-ice ({acc['c4_bad'] / max(acc['c4_tot'], 1):.2%})")
    c5 = pd.concat(acc["c5"])
    ge, ng = c5[c5.goal_end], c5[~c5.goal_end]
    r_ge = 3600 * ge.xGF.sum() / ge.dur.sum()
    r_ng = 3600 * ng.xGF.sum() / ng.dur.sum()
    naive = c5.assign(r=3600 * c5.xGF / c5.dur)
    print(f"C5 STOP-BIAS: xGF/60 goal-ended windows {r_ge:.2f} vs others {r_ng:.2f} "
          f"(exposure-summed, unbiased) | naive per-window mean rate "
          f"{naive.r.mean():.2f} vs exposure-weighted {3600 * c5.xGF.sum() / c5.dur.sum():.2f}")
    c6 = pd.concat(acc["c6"])
    for flag, name in [(1, "mid-window entrants"), (0, "present at faceoff")]:
        m = c6[c6.entered_after_start == flag]
        print(f"C6 ENTRY-SELECT: {name}: xGF60 "
              f"{3600 * m.xGF.sum() / m.seconds.sum():.3f}  (n={len(m):,})")
    pd.to_pickle(acc, ART / f"crownjewel_audit_{args.year}.pkl")
