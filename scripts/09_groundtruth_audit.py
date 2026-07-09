"""THE GROUND-TRUTH AUDIT — grade our windows against the NHL's own records.

For a stratified sample of games (default 10/season x 7 seasons):
  download official shift charts (exact per-player intervals) + play-by-play
  (exact event times), then grade four layers of our stack:

  G1 TOI        per-player game seconds (our windows) vs official shift totals
  G2 PRESENCE   for each 5v5 goal: the on-ice set from official shifts vs the
                players our windows credit (y_GF>0) — exact set match, incl.
                the 'on for 3 seconds then subbed' cases
  G3 EVENTS     hits/blocks/giveaways/takeaways per player-game vs pbp counts
  G4 BOUNDS     every 5v5 goal falls inside exactly one of our windows for
                the scoring team, that window scores GF>=1

Writes artifacts/groundtruth_audit_report.json and prints letter grades.
Usage: python scripts/09_groundtruth_audit.py [--per-season 10]
"""
import argparse
import glob
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")
CACHE = ART / "nhl_api_cache"
SEASON_DIRS = {20182019: r"D:\2018", 20192020: r"D:\2019", 20202021: r"D:\2020",
               20212022: r"D:\2021", 20222023: r"D:\2022", 20232024: r"D:\2023",
               20242025: r"D:\2024"}


def fetch(url: str, cache_name: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / cache_name
    if fp.exists():
        return json.loads(fp.read_text())
    req = urllib.request.Request(url, headers={"User-Agent": "optbot-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    fp.write_text(json.dumps(data))
    time.sleep(0.4)
    return data


def mmss(t: str) -> int:
    m, s = str(t).split(":")
    return int(m) * 60 + int(s)


def official_shifts(gamePk: int) -> pd.DataFrame:
    d = fetch(f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gamePk}",
              f"shifts_{gamePk}.json")
    rows = [r for r in d.get("data", []) if r.get("typeCode") == 517 and r.get("startTime")]
    sh = pd.DataFrame(rows)
    if sh.empty:
        return sh
    sh["start_g"] = (sh.period - 1) * 1200 + sh.startTime.map(mmss)
    sh["end_g"] = (sh.period - 1) * 1200 + sh.endTime.map(mmss)
    sh.loc[sh.end_g < sh.start_g, "end_g"] += 0  # data quirk guard
    return sh[["playerId", "teamId", "period", "start_g", "end_g"]]


def official_goals_5v5(gamePk: int) -> list[dict]:
    d = fetch(f"https://api-web.nhle.com/v1/gamecenter/{gamePk}/play-by-play",
              f"pbp_{gamePk}.json")
    out = []
    for p in d.get("plays", []):
        if p.get("typeDescKey") == "goal" and p.get("situationCode") == "1551":
            per = p["periodDescriptor"]["number"]
            if per > 3:
                continue                       # OT is 3v3 in regular season
            out.append({"t_g": (per - 1) * 1200 + mmss(p["timeInPeriod"]),
                        "period": per,
                        "team": p["details"]["eventOwnerTeamId"],
                        "scorer": p["details"].get("scoringPlayerId")})
    ev = {"hit": ("hittingPlayerId", "y_hits"), "blocked-shot": ("blockingPlayerId", "y_blocks"),
          "giveaway": ("playerId", "y_giveaways"), "takeaway": ("playerId", "y_takeaways")}
    counts = {}
    for p in d.get("plays", []):
        k = p.get("typeDescKey")
        if k in ev and p.get("situationCode") == "1551" and p["periodDescriptor"]["number"] <= 3:
            pid = p.get("details", {}).get(ev[k][0])
            if pid:
                counts.setdefault((pid, ev[k][1]), 0)
                counts[(pid, ev[k][1])] += 1
    return out, counts


def our_game(season: int, gamePk: int) -> pd.DataFrame:
    f = Path(SEASON_DIRS[season]) / "final_windows" / f"player_windows_train_{gamePk}_xg.csv"
    if not f.exists():
        return pd.DataFrame()
    df = pd.read_csv(f)
    # start_sec/end_sec are ALREADY game-elapsed (verified: P2 spans 1200-2400).
    # v1 of this audit added (period-1)*1200 on top — exiling all P2/P3 windows
    # and producing spurious 34% G4. The windows were right; the audit was wrong.
    df["start_g"] = df.start_sec
    df["end_g"] = df.end_sec
    return df[df.period <= 3]


def audit_game(season: int, gamePk: int) -> dict | None:
    ours = our_game(season, gamePk)
    if ours.empty:
        return None
    sh = official_shifts(gamePk)
    goals, ev_counts = official_goals_5v5(gamePk)
    if sh.empty:
        return None
    res = {"gamePk": gamePk, "season": season}

    # G1: per-player total seconds (all strengths, regulation only — our CSVs and
    # official shifts must cover the same periods for a fair comparison)
    ours_toi = ours.groupby("playerId")["seconds"].sum()
    sh = sh[sh.period <= 3]
    off_toi = sh.assign(d=sh.end_g - sh.start_g).groupby("playerId")["d"].sum()
    j = pd.concat([ours_toi.rename("ours"), off_toi.rename("official")], axis=1).dropna()
    res["g1_players"] = int(len(j))
    res["g1_mean_abs_diff_s"] = float((j.ours - j.official).abs().mean())
    res["g1_within_30s_pct"] = float(((j.ours - j.official).abs() <= 30).mean())

    # G2 + G4: goal credit sets
    g2_exact, g2_prec, g2_rec, g4_ok = [], [], [], []
    is55 = ours.strength_global.astype(str).str.replace("_", "").str.lower().isin(["5v5", "5v55v5"])
    for g in goals:
        t = g["t_g"]
        on_official = set(sh[(sh.teamId == g["team"]) & (sh.start_g <= t - 1) & (sh.end_g >= t)]
                          .playerId) & set(ours.playerId)   # skaters we model
        win = ours[is55 & (ours.teamId == g["team"]) & (ours.start_g <= t) & (ours.end_g >= t)]
        credited = set(win[win.GF > 0].playerId)
        g4_ok.append(len(win) > 0 and win.GF.max() >= 1)
        if on_official:
            g2_exact.append(credited == on_official)
            inter = len(credited & on_official)
            g2_prec.append(inter / len(credited) if credited else 0.0)
            g2_rec.append(inter / len(on_official))
    res["g2_goals"] = len(g2_exact)
    res["g2_exact_set_pct"] = float(np.mean(g2_exact)) if g2_exact else np.nan
    res["g2_precision"] = float(np.mean(g2_prec)) if g2_prec else np.nan
    res["g2_recall"] = float(np.mean(g2_rec)) if g2_rec else np.nan
    res["g4_goal_in_window_pct"] = float(np.mean(g4_ok)) if g4_ok else np.nan

    # G3: micro-event counts per player-game (5v5)
    ours55 = ours[is55]
    hits_ours = ours55.groupby("playerId")[["hits_personal", "blocks_personal",
                                            "giveaways_committed", "takeaways_forced"]].sum()
    colmap = {"y_hits": "hits_personal", "y_blocks": "blocks_personal",
              "y_giveaways": "giveaways_committed", "y_takeaways": "takeaways_forced"}
    match, tot = 0, 0
    for (pid, ycol), n in ev_counts.items():
        if pid in hits_ours.index:
            tot += 1
            match += int(hits_ours.loc[pid, colmap[ycol]] == n)
    res["g3_events_checked"] = tot
    res["g3_exact_pct"] = float(match / tot) if tot else np.nan
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-season", type=int, default=10)
    args = ap.parse_args()
    rng = np.random.default_rng(42)
    results = []
    for season, d in SEASON_DIRS.items():
        files = [f for f in glob.glob(f"{d}\\final_windows\\player_windows_train_*_xg.csv")
                 if "backup" not in f]
        picks = rng.choice(len(files), min(args.per_season, len(files)), replace=False)
        for i in picks:
            gamePk = int(Path(files[i]).stem.split("_")[3])
            try:
                r = audit_game(season, gamePk)
                if r:
                    results.append(r)
                    print(f"{gamePk}: G1 {r['g1_within_30s_pct']:.0%}/30s "
                          f"G2 exact {r.get('g2_exact_set_pct', float('nan')):.0%} "
                          f"({r['g2_goals']} goals) G3 {r.get('g3_exact_pct', float('nan')):.0%} "
                          f"G4 {r.get('g4_goal_in_window_pct', float('nan')):.0%}")
            except Exception as e:
                print(f"{gamePk}: FAILED {type(e).__name__}: {e}")
    df = pd.DataFrame(results)
    (ART / "groundtruth_audit_report.json").write_text(df.to_json(orient="records", indent=2))
    print("\n================ FINAL GRADES ================")
    print(f"games audited: {len(df)}  |  5v5 goals checked: {int(df.g2_goals.sum())}")
    print(f"G1 TOI:      mean abs diff {df.g1_mean_abs_diff_s.mean():.1f}s | "
          f"within 30s {df.g1_within_30s_pct.mean():.1%}")
    print(f"G2 PRESENCE: exact on-ice set match {df.g2_exact_set_pct.mean():.1%} | "
          f"precision {df.g2_precision.mean():.1%} | recall {df.g2_recall.mean():.1%}")
    print(f"G3 EVENTS:   exact count match {df.g3_exact_pct.mean():.1%} "
          f"({int(df.g3_events_checked.sum())} player-events)")
    print(f"G4 BOUNDS:   goal-in-window {df.g4_goal_in_window_pct.mean():.1%}")
    def grade(x, a, b):
        return "A" if x >= a else "B" if x >= b else "C/INVESTIGATE"
    print(f"\nGRADES: G1 {grade(df.g1_within_30s_pct.mean(), .95, .85)} | "
          f"G2 {grade(df.g2_exact_set_pct.mean(), .90, .75)} | "
          f"G3 {grade(df.g3_exact_pct.mean(), .85, .70)} | "
          f"G4 {grade(df.g4_goal_in_window_pct.mean(), .97, .90)}")
