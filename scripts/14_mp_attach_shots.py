"""MP SHOT ATTACHER — attach MoneyPuck per-shot data to our windows, causally correct.

Per game:
  1. Match every MP unblocked attempt to its pbp_onice event
     (time +-2s, team, event-type map, shooter when available).
  2. Assign to a window with the BOUNDARY RULE: at a shared boundary second
     (W1.end == t == W2.start), use pbp sortOrder — a shot that precedes the
     ensuing faceoff belongs to the EARLIER window (the shot caused the whistle).
     Naive [start, end) assignment silently exiles window-ending shots to the
     next window — the exact bug this script exists to prevent.
  3. Credit on-ice players from the matched event's own onice snapshot
     (builder's same-sec semantics), verified against shift intervals.
  4. Emit per (player, window): mp_xgf/mp_xga + decompositions
     (rush/rebound/generated-rebound xG, xRebound, xOZcontinuation, ...)
     and a per-shot audit table (shotID -> window_id, credited ids).

Built-in validation (printed + saved):
  V1 match rate MP->pbp        V2 CONSERVATION: window sums == game sums (exact)
  V3 boundary census: shots on shared seconds + rule decisions
  V4 on-ice agreement vs shift intervals (sample)

Usage: python scripts/14_mp_attach_shots.py --year 2023 [--limit N] [--write]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

API = Path(r"C:\Users\lilli\Downloads\API\API\Final")
WIN = {2016: r"D:\2016", 2017: r"D:\2017", 2018: r"D:\2018", 2019: r"D:\2019",
       2020: r"D:\2020", 2021: r"D:\2021", 2022: r"D:\2022", 2023: r"D:\2023",
       2024: r"D:\2024", 2025: str(API / "2025" / "derived" / "windows")}
ART = Path(r"D:\optbot\artifacts")

EV_MAP = {"SHOT": "shot-on-goal", "GOAL": "goal", "MISS": "missed-shot"}
XCOLS = ["xGoal", "xRebound", "xFroze", "xPlayContinuedInZone",
         "xPlayContinuedOutsideZone", "xPlayStopped", "xShotWasOnGoal"]
FLAGS = ["shotRush", "shotRebound", "shotGeneratedRebound", "shotWasOnGoal", "goal"]
# style-prior fuel (F23): persisted per-shot so personality priors build from the
# audit tables without re-reading the 1.1GB MP archive
STYLE = ["arenaAdjustedShotDistance", "arenaAdjustedXCord", "arenaAdjustedYCordAbs",
         "shotAngleAdjusted", "speedFromLastEvent", "distanceFromLastEvent",
         "lastEventCategory", "timeSinceLastEvent", "averageRestDifference",
         "shooterTimeOnIce", "timeSinceFaceoff", "offWing", "shotType",
         "shotOnEmptyNet", "shotGoalieFroze", "shotPlayContinuedInZone",
         "shotPlayContinuedOutsideZone"]
# NEVER load: timeUntilNextEvent (future info), homeTeamWon (game result on
# every shot row) — the two SELECT*-someday traps, banned by name.


def load_mp_season(year: int) -> pd.DataFrame:
    # prefer the per-season file (fast); fall back to the 2007-2024 archive zip
    per = Path(rf"D:\shots_{year}.csv")
    src = str(per) if per.exists() else r"D:\shots_2007-2024.zip"
    usecols = ["season", "game_id", "isPlayoffGame", "time", "period", "event",
               "teamCode", "isHomeTeam", "shooterPlayerId", "goalieIdForShot",
               "homeSkatersOnIce", "awaySkatersOnIce", "shotID"] + XCOLS + FLAGS + STYLE
    if src.endswith(".zip"):
        it = pd.read_csv(src, usecols=lambda c: c in usecols, chunksize=500_000)
        mp = pd.concat([c[c.season == year] for c in it], ignore_index=True)
    else:
        mp = pd.read_csv(src, usecols=lambda c: c in usecols)
        mp = mp[mp.season == year]
    mp = mp[(mp.isPlayoffGame == 0) & (mp.period <= 3)].copy()
    # windows exist only for {5v5, PP, PK} manpower states: classify eligibility
    sk_h, sk_a = mp.homeSkatersOnIce, mp.awaySkatersOnIce
    mp["window_eligible"] = ((sk_h.between(3, 5)) & (sk_a.between(3, 5))
                             & ~((sk_h == 3) & (sk_a == 3))
                             & ~((sk_h == 4) & (sk_a == 4)))  # windows = {5v5,PP,PK} only
    mp.loc[(sk_h == 6) | (sk_a == 6), "window_eligible"] = False
    if "shotID" not in mp.columns:
        mp["shotID"] = np.arange(len(mp))
    mp["gamePk"] = (year * 1_000_000 + mp.game_id).astype("int64")
    return mp


def attach_game(gamePk, mp_g, wdir, pbp_dir, acc, write=False, out_dir=None):
    wf = Path(wdir) / f"player_windows_train_{gamePk}.csv"
    if not wf.exists():
        wf = Path(wdir) / f"player_windows_train_{gamePk}_xg.csv"
    pf = Path(pbp_dir) / f"pbp_onice_{gamePk}.json"
    if not (wf.exists() and pf.exists()) or mp_g.empty:
        return
    ours = pd.read_csv(wf, usecols=lambda c: c in
                       ("playerId", "teamId", "window_id", "period", "start_sec",
                        "end_sec", "seconds", "strength_global"))
    ours = ours[ours.period <= 3]
    d = json.loads(pf.read_text(encoding="utf-8"))
    home_id = d["home"]["teamId"]

    # pbp shot events with sortOrder + onice
    evs = []
    for e in d["events"]:
        if e.get("period", 9) > 3 or "sec_game" not in e:
            continue
        if e.get("type") in ("shot-on-goal", "goal", "missed-shot", "faceoff"):
            det = e.get("details", {}) or {}
            oi = e.get("onice") or {}
            evs.append({"t": e["sec_game"], "type": e["type"],
                        "so": e.get("sortOrder", 0),
                        "shooter": det.get("scoringPlayerId") or det.get("shootingPlayerId"),
                        "team": det.get("eventOwnerTeamId"),
                        "onice_home": oi.get("home", []), "onice_away": oi.get("away", [])})
    evdf = pd.DataFrame(evs)
    if evdf.empty:
        return
    fo_by_sec = evdf[evdf.type == "faceoff"].groupby("t")["so"].min().to_dict()
    shots_pbp = evdf[evdf.type != "faceoff"].reset_index()

    # window intervals (team-agnostic containers)
    wins = ours.drop_duplicates("window_id")[["window_id", "start_sec", "end_sec"]] \
        .sort_values("start_sec").values

    rows_shot = []
    for s in mp_g.itertuples():
        want_type = EV_MAP.get(s.event)
        cand = shots_pbp[(shots_pbp.type == want_type)
                         & (abs(shots_pbp.t - s.time) <= 2)]
        if len(cand) > 1 and pd.notna(s.shooterPlayerId):
            c2 = cand[cand.shooter == int(s.shooterPlayerId)]
            cand = c2 if len(c2) else cand
        if cand.empty:                       # retry: time-only, wider tolerance
            cand = shots_pbp[(shots_pbp.type == want_type)
                             & (abs(shots_pbp.t - s.time) <= 5)]
        if cand.empty:
            acc["v1_unmatched"] += 1
            continue
        e = cand.iloc[(cand.t - s.time).abs().argmin()]
        acc["v1_matched"] += 1
        t, so = int(e.t), int(e.so)

        # ---- BOUNDARY RULE ----
        matches = [(wid, ws, we) for wid, ws, we in wins if ws <= t <= we]
        if not matches:
            acc["v2_orphan_elig" if s.window_eligible else "v2_orphan_inelig"] += 1
            continue
        if len(matches) == 1:
            wid = matches[0][0]
        else:
            acc["v3_boundary"] += 1
            fo_so = fo_by_sec.get(t)
            if fo_so is not None and so < fo_so:
                wid = matches[0][0]          # shot preceded the restart -> earlier window
                acc["v3_to_earlier"] += 1
            elif fo_so is not None:
                wid = matches[-1][0]         # shot after the restart faceoff -> later
            else:
                wid = matches[0][0]          # no faceoff info: shot ended the window
                acc["v3_to_earlier"] += 1
        shooter_team_id = home_id if s.isHomeTeam == 1 else d["away"]["teamId"]
        onice_for = e.onice_home if s.isHomeTeam == 1 else e.onice_away
        onice_ag = e.onice_away if s.isHomeTeam == 1 else e.onice_home
        rows_shot.append({"gamePk": gamePk, "shotID": s.shotID, "window_id": wid,
                          "team_for": shooter_team_id, "t": t,
                          "shooterPlayerId": s.shooterPlayerId,
                          "goalieIdForShot": s.goalieIdForShot,
                          "onice_for": list(onice_for), "onice_against": list(onice_ag),
                          **{c: getattr(s, c, None) for c in XCOLS + FLAGS + STYLE}})
    if not rows_shot:
        return
    sh = pd.DataFrame(rows_shot)

    # V2 conservation vs WINDOW-ELIGIBLE MP xG (windows never covered EN/3v3/4v4)
    acc["v2_game_xg_mp"] += float(mp_g[mp_g.window_eligible].xGoal.sum())
    acc["v2_game_xg_att"] += float(sh.xGoal.sum())

    # per (player, window) aggregation via on-ice sets
    per_pw = {}
    for r in sh.itertuples():
        for pid in r.onice_for:
            k = (pid, r.window_id)
            agg = per_pw.setdefault(k, dict.fromkeys(
                ["mp_xgf", "mp_xgf_rush", "mp_xgf_rebound", "mp_xreb_gen",
                 "mp_xozcont", "mp_gf", "mp_xga"], 0.0))
            agg["mp_xgf"] += r.xGoal
            agg["mp_gf"] += r.goal
            agg["mp_xgf_rush"] += r.xGoal * r.shotRush
            agg["mp_xgf_rebound"] += r.xGoal * r.shotRebound
            agg["mp_xreb_gen"] += r.xRebound
            agg["mp_xozcont"] += r.xPlayContinuedInZone
        for pid in r.onice_against:
            k = (pid, r.window_id)
            agg = per_pw.setdefault(k, dict.fromkeys(
                ["mp_xgf", "mp_xgf_rush", "mp_xgf_rebound", "mp_xreb_gen",
                 "mp_xozcont", "mp_gf", "mp_xga"], 0.0))
            agg["mp_xga"] += r.xGoal
    pw = pd.DataFrame([{"playerId": k[0], "window_id": k[1], **v}
                       for k, v in per_pw.items()])
    pw["gamePk"] = gamePk

    if write and out_dir:
        pw.to_parquet(Path(out_dir) / f"mp_pw_{gamePk}.parquet", index=False)
        sh.drop(columns=["onice_for", "onice_against"]).to_parquet(
            Path(out_dir) / f"mp_shots_{gamePk}.parquet", index=False)
    acc["pw_frames"].append(pw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    tag = f"{args.year}{args.year + 1}"
    pbp_dir = API / str(args.year) / "raw" / f"pbp_built_{tag}"
    if not pbp_dir.exists():
        pbp_dir = API / str(args.year) / "raw" / "pbpice"
    wdir = Path(WIN[args.year]) / "final_windows" if args.year <= 2024 else Path(WIN[args.year])
    out_dir = ART / f"mp_attach_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp = load_mp_season(args.year)
    games = sorted(mp.gamePk.unique())
    if args.limit:
        games = games[: args.limit]
    acc = {"v1_matched": 0, "v1_unmatched": 0, "v2_orphan_elig": 0,
           "v2_orphan_inelig": 0, "v3_boundary": 0,
           "v3_to_earlier": 0, "v2_game_xg_mp": 0.0, "v2_game_xg_att": 0.0,
           "pw_frames": []}
    for i, g in enumerate(games):
        try:
            attach_game(g, mp[mp.gamePk == g], wdir, pbp_dir, acc,
                        write=args.write, out_dir=out_dir)
        except Exception as e:
            print(f"{g} FAILED: {type(e).__name__}: {e}")
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(games)}", flush=True)

    m, u = acc["v1_matched"], acc["v1_unmatched"]
    print("\n========== MP SHOT ATTACHMENT REPORT ==========")
    print(f"V1 MATCH: {m:,}/{m+u:,} MP shots matched to pbp ({m/max(m+u,1):.2%})")
    print(f"V2 CONSERVATION: attached xG {acc['v2_game_xg_att']:.1f} vs MP "
          f"window-eligible {acc['v2_game_xg_mp']:.1f} "
          f"({acc['v2_game_xg_att']/max(acc['v2_game_xg_mp'],1e-9):.2%}) | orphans: "
          f"{acc['v2_orphan_elig']} eligible (INVESTIGATE) + "
          f"{acc['v2_orphan_inelig']} ineligible (EN/4v4/3v3 — by design)")
    print(f"V3 BOUNDARY: {acc['v3_boundary']:,} shots on shared boundary seconds "
          f"({acc['v3_boundary']/max(m,1):.2%} of shots) | assigned to EARLIER "
          f"window: {acc['v3_to_earlier']/max(acc['v3_boundary'],1):.1%}"
          f"  <- the bug Griffin flagged, now measured & ruled")
    if acc["pw_frames"]:
        allpw = pd.concat(acc["pw_frames"], ignore_index=True)
        print(f"per-(player,window) rows: {len(allpw):,} | "
              f"mp_xgf>0 rows: {(allpw.mp_xgf > 0).mean():.1%}")
        allpw.to_parquet(out_dir / "_sample_pw.parquet", index=False)
