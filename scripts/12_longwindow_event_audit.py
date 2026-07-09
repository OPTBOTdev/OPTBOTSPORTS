"""LONG-WINDOW EVENT ATTRIBUTION MICROSCOPE — the churn audit.

Only windows >120s (where players flow through). For EVERY event, verify the right
humans got it, using official shifts as the referee:

  L1 ON-ICE LEDGER   per (player, long-window): team shots-for during HIS exact
                     presence intervals vs our SF (FULL population, not sampled);
                     same for GF.
  L2 PERSONAL LEDGER hits/giveaways/takeaways by actor during his presence vs our
                     hits_personal/giveaways_committed/takeaways_forced.
  L3 SOURCE ANOMALY  events whose ACTOR was not on ice at event time per official
                     shifts (upstream NHL data disease — count it, don't inherit it).
  L4 PEOPLE BLUR     at each event moment, what share of the ACTUAL on-ice teammates
                     appear in the focal player's top-5 with-list? (the blend cost
                     that stint-ization removes) — long vs short windows.

Usage: python scripts/12_longwindow_event_audit.py [--year 2023] [--limit N]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

API = Path(r"C:\Users\lilli\Downloads\API\API\Final")
WIN = {2023: r"D:\2023"}
ART = Path(r"D:\optbot\artifacts")
LONG = 120


def mmss(t):
    m, s = str(t).split(":")
    return int(m) * 60 + int(s)


def parse_people(ids_str, sec_str, k=5):
    if not isinstance(ids_str, str) or not ids_str:
        return []
    try:
        ids = np.array(ids_str.split("|"), dtype=np.int64)
        secs = np.array(str(sec_str).split("|"), dtype=np.float64)
    except ValueError:
        return []
    n = min(len(ids), len(secs))
    order = np.argsort(-secs[:n], kind="stable")
    return list(ids[:n][order][:k])


def audit_game(year, tag, gamePk, acc):
    wf = Path(WIN[year]) / "final_windows" / f"player_windows_train_{gamePk}_xg.csv"
    sf = API / str(year) / "csv" / "shiftcharts" / f"shiftcharts_{gamePk}.csv"
    pf = API / str(year) / "raw" / f"pbp_built_{tag}" / f"pbp_onice_{gamePk}.json"
    if not (wf.exists() and sf.exists() and pf.exists()):
        return
    ours = pd.read_csv(wf, usecols=["playerId", "teamId", "window_id", "period",
                                    "start_sec", "end_sec", "seconds", "duration",
                                    "strength_global", "SF", "GF", "hits_personal",
                                    "giveaways_committed", "takeaways_forced",
                                    "teammates_onice_ids_w", "teammates_onice_sec_w"])
    ours = ours[(ours.period <= 3) & (ours.duration > 0)]
    is55 = ours.strength_global.astype(str).str.lower().str.replace("_", "") \
        .isin(["5v5", "5v55v5"])
    ours = ours[is55]
    sh = pd.read_csv(sf, usecols=["playerId", "teamId", "period", "startTime",
                                  "endTime", "typeCode"])
    sh = sh[(sh.typeCode == 517) & (sh.period <= 3)].dropna(subset=["startTime", "endTime"])
    sh["s"] = (sh.period - 1) * 1200 + sh.startTime.map(mmss)
    sh["e"] = (sh.period - 1) * 1200 + sh.endTime.map(mmss)
    sh = sh[sh.e > sh.s]
    shifts_by_p = {pid: g[["s", "e"]].values for pid, g in sh.groupby("playerId")}
    d = json.loads(pf.read_text(encoding="utf-8"))
    evs = [e for e in d["events"] if e.get("period", 9) <= 3 and "sec_game" in e]

    PERSONAL = {"hit": ("hittingPlayerId", "hits_personal"),
                "giveaway": ("playerId", "giveaways_committed"),
                "takeaway": ("playerId", "takeaways_forced")}
    shots = [(e["sec_game"], (e.get("details") or {}).get("eventOwnerTeamId"))
             for e in evs if e.get("type") in ("shot-on-goal", "goal")]
    goals = [(e["sec_game"], (e.get("details") or {}).get("eventOwnerTeamId"))
             for e in evs if e.get("type") == "goal"]
    personal = []
    for e in evs:
        et = e.get("type")
        if et in PERSONAL:
            det = e.get("details") or {}
            pid = det.get(PERSONAL[et][0])
            if pid:
                personal.append((e["sec_game"], pid, PERSONAL[et][1]))

    lw = ours[ours.duration > LONG]
    for r in lw.itertuples():
        ivs = shifts_by_p.get(r.playerId)
        if ivs is None:
            continue
        inter = [(max(s, r.start_sec), min(e, r.end_sec)) for s, e in ivs
                 if e > r.start_sec and s < r.end_sec]
        if not inter:
            continue
        def present(t):
            return any(a <= t <= b for a, b in inter)
        sf_true = sum(1 for t, tm in shots if tm == r.teamId
                      and r.start_sec <= t <= r.end_sec and present(t))
        gf_true = sum(1 for t, tm in goals if tm == r.teamId
                      and r.start_sec <= t <= r.end_sec and present(t))
        acc["l1_sf"].append((sf_true, r.SF))
        acc["l1_gf"].append((gf_true, r.GF))
        for col in ("hits_personal", "giveaways_committed", "takeaways_forced"):
            true_n = sum(1 for t, pid, c in personal
                         if pid == r.playerId and c == col
                         and r.start_sec <= t <= r.end_sec and present(t))
            acc["l2"].append((true_n, getattr(r, col)))

    # L3: actor on ice at event time?
    for t, pid, _ in personal:
        ivs = shifts_by_p.get(pid)
        if ivs is not None:
            acc["l3"].append(any(s <= t <= e for s, e in ivs))

    # L4: with-list event-time coverage, long vs short
    ours = ours.copy()
    ours["with5"] = [parse_people(a, b) for a, b in
                     zip(ours.teammates_onice_ids_w, ours.teammates_onice_sec_w)]
    onice_at = {}
    for e in evs:
        if e.get("type") in ("shot-on-goal", "goal", "hit"):
            oi = e.get("onice") or {}
            onice_at[e["sec_game"]] = {d["home"]["teamId"]: set(oi.get("home", [])),
                                       d["away"]["teamId"]: set(oi.get("away", []))}
    for r in ours.itertuples():
        for t, sets in onice_at.items():
            if r.start_sec < t < r.end_sec:
                mates = sets.get(r.teamId, set()) - {r.playerId}
                if r.playerId in sets.get(r.teamId, set()) and mates and r.with5:
                    cov = len(mates & set(r.with5)) / len(mates)
                    acc["l4"].append((r.duration > LONG, cov))
                break


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
    acc = {"l1_sf": [], "l1_gf": [], "l2": [], "l3": [], "l4": []}
    for i, f in enumerate(files):
        gamePk = int(Path(f).stem.split("_")[-1])
        try:
            audit_game(args.year, tag, gamePk, acc)
        except Exception as e:
            print(f"{gamePk} FAILED: {type(e).__name__}: {e}")
        if (i + 1) % 150 == 0:
            print(f"{i+1}/{len(files)}")

    print("\n========= LONG-WINDOW EVENT MICROSCOPE (windows >120s) =========")
    for name, key in [("SF", "l1_sf"), ("GF", "l1_gf")]:
        p = pd.DataFrame(acc[key], columns=["true", "ours"])
        print(f"L1 {name}: {len(p):,} player-long-windows | exact "
              f"{np.mean(p.true == p.ours):.2%} | corr {p.true.corr(p.ours):.3f} | "
              f"bias {np.mean(p.ours - p.true):+.4f}")
    p = pd.DataFrame(acc["l2"], columns=["true", "ours"])
    print(f"L2 PERSONAL: {len(p):,} player-window-stats | exact "
          f"{np.mean(p.true == p.ours):.2%} | bias {np.mean(p.ours - p.true):+.4f}")
    l3 = np.array(acc["l3"])
    print(f"L3 SOURCE ANOMALY: {np.mean(~l3):.2%} of {len(l3):,} personal events have "
          f"an actor NOT on official ice at event time (upstream NHL disease)")
    l4 = pd.DataFrame(acc["l4"], columns=["is_long", "coverage"])  # 'cov' shadows .cov()
    for flag, nm in [(False, "short (<=120s)"), (True, "long (>120s)")]:
        m = l4[l4.is_long == flag]
        if len(m):
            print(f"L4 PEOPLE BLUR {nm}: top-5 with-list covers {m.coverage.mean():.1%} of "
                  f"actual on-ice teammates at event moments (n={len(m):,})")
    pd.to_pickle(acc, ART / f"longwindow_audit_{args.year}.pkl")
