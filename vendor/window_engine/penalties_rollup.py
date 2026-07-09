# penalties_rollup.py
#
# Usage:
#   python penalties_rollup.py /path/to/pbp_jsons/ penalties_per_game.csv
#
# Input files are pbp_onice_<gamePk>.json (your on-ice PBP export).
# Output is a per-(season, date, gameid, teamid, playerId) CSV with penalty counts/minutes.

import json, sys, glob, os, argparse, re
from collections import defaultdict
from datetime import datetime
import pandas as pd

# ---- helper classification ----

MINOR_CODES = {"MIN"}                  # 2-min
DOUBLE_MINOR_CODES = {"DBL_MIN"}      # some feeds use this (store as 4)
MAJOR_CODES = {"MAJ"}                 # 5-min
MISCONDUCT_CODES = {"MIS", "GAM", "MATCH"}  # 10 (or more) – not strength-changing
OFFSETTING_HINTS = {"coincidental"}   # descKey sometimes includes this word

def classify_penalty(type_code:str, duration:int, desc_key:str):
    """Return (class, minutes) where class ∈ {"minor","double_minor","major","misconduct","other"}."""
    t = (type_code or "").upper()
    dk = (desc_key or "").lower()

    if t in MINOR_CODES or duration == 2:
        return "minor", 2
    if t in DOUBLE_MINOR_CODES or duration == 4:
        return "double_minor", 4
    if t in MAJOR_CODES or duration == 5:
        return "major", 5
    if t in MISCONDUCT_CODES or duration >= 10 or "misconduct" in dk:
        # treat as misconduct-type; does not change strength for PP/PK modeling
        return "misconduct", max(10, duration or 10)
    # Fallback: trust duration if present
    if duration in (2,4,5,10):
        mapping = {2:"minor",4:"double_minor",5:"major",10:"misconduct"}
        return mapping[duration], duration
    return "other", duration or 0

def detect_coincidental(penalty_events_at_second):
    """
    Simple coincidental detector:
      - more than one penalty at the same (period, sec_game) AND
      - committed by BOTH teams (distinct eventOwnerTeamId)
    """
    teams = {e["details"].get("eventOwnerTeamId") for e in penalty_events_at_second if "details" in e}
    return (len(teams) >= 2)

# ---- main extraction ----

def process_game(json_path):
    with open(json_path, "r") as f:
        g = json.load(f)

    gamePk = g.get("gamePk")
    # If you have season/date elsewhere, you can recover; otherwise leave season/date blank.
    # Here we'll leave blank and you can join later if needed.
    home_team = g["home"]["teamId"]
    away_team = g["away"]["teamId"]

    events = g["events"]

    # Build a quick index (period, sec_game) -> list of penalty events at that second
    by_second = defaultdict(list)
    for ev in events:
        if ev.get("type") == "penalty":
            by_second[(ev["period"], ev["sec_game"])].append(ev)

    # Prepare per-player tallies
    rows = []

    for (period, sec), evs in by_second.items():
        is_coinc = detect_coincidental(evs)

        # If you want to downweight coincidentals, keep the flag; otherwise counts are still useful behaviorally.
        for ev in evs:
            det = ev.get("details", {})
            committed = det.get("committedByPlayerId")
            drawn = det.get("drawnByPlayerId")
            type_code = det.get("typeCode")
            desc_key = det.get("descKey")
            duration = det.get("duration", 0)  # minutes per NHL feed (2,4,5,10,...)
            owner_team = det.get("eventOwnerTeamId")
            # We will assign the drawn player's team as the opposite of the penalized (owner) team

            pclass, mins = classify_penalty(type_code, duration, desc_key)

            # offender row (penalties taken)
            if committed:
                rows.append({
                    "gameid": gamePk,
                    "teamid": owner_team,
                    "playerId": committed,
                    "period": period,
                    "sec_game": sec,
                    "penalty_class": pclass,
                    "penalty_minutes": mins,
                    "taken": 1,
                    "drawn": 0,
                    "coincidental_flag": 1 if is_coinc else 0,
                })

            # drawn row (penalties drawn)
            if drawn:
                # Opposite of penalized team
                drawn_team = None
                if owner_team == home_team:
                    drawn_team = away_team
                elif owner_team == away_team:
                    drawn_team = home_team
                rows.append({
                    "gameid": gamePk,
                    "teamid": drawn_team,
                    "playerId": drawn,
                    "period": period,
                    "sec_game": sec,
                    "penalty_class": pclass,
                    "penalty_minutes": mins,
                    "taken": 0,
                    "drawn": 1,
                    "coincidental_flag": 1 if is_coinc else 0,
                })

    # collapse to per-player game totals
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Optional: split classes into columns (minor/major/misconduct/other)
    classes = ["minor","double_minor","major","misconduct","other"]
    for c in classes:
        df[f"pen_{c}_taken"] = ((df["penalty_class"] == c) & (df["taken"]==1)).astype(int)
        df[f"pen_{c}_drawn"] = ((df["penalty_class"] == c) & (df["drawn"]==1)).astype(int)
        df[f"pen_{c}_mins_taken"] = df.apply(lambda r: r["penalty_minutes"] if (r["penalty_class"]==c and r["taken"]==1) else 0, axis=1)
        df[f"pen_{c}_mins_drawn"] = df.apply(lambda r: r["penalty_minutes"] if (r["penalty_class"]==c and r["drawn"]==1) else 0, axis=1)

    agg = {
        "taken": "sum",
        "drawn": "sum",
        "coincidental_flag": "max",
        # minutes by role
        **{f"pen_{c}_taken":"sum" for c in classes},
        **{f"pen_{c}_drawn":"sum" for c in classes},
        **{f"pen_{c}_mins_taken":"sum" for c in classes},
        **{f"pen_{c}_mins_drawn":"sum" for c in classes},
    }

    out = (df
           .groupby(["gameid","teamid","playerId"], dropna=False)
           .agg(agg)
           .reset_index())

    # Convenience totals
    out["pen_minutes_taken_total"] = out[[f"pen_{c}_mins_taken" for c in classes]].sum(axis=1)
    out["pen_minutes_drawn_total"] = out[[f"pen_{c}_mins_drawn" for c in classes]].sum(axis=1)
    out["pen_taken_total"] = out[[f"pen_{c}_taken" for c in classes]].sum(axis=1)
    out["pen_drawn_total"] = out[[f"pen_{c}_drawn" for c in classes]].sum(axis=1)

    # Keep a clean subset you’ll actually join into priors
    keep_cols = [
        "gameid","teamid","playerId",
        "pen_taken_total","pen_drawn_total",
        "pen_minutes_taken_total","pen_minutes_drawn_total",
        "pen_minor_taken","pen_minor_drawn",
        "pen_major_taken","pen_major_drawn",
        "pen_misconduct_taken","pen_misconduct_drawn",
        "coincidental_flag",
        # keep class minutes if you want:
        "pen_minor_mins_taken","pen_minor_mins_drawn",
        "pen_major_mins_taken","pen_major_mins_drawn",
        "pen_misconduct_mins_taken","pen_misconduct_mins_drawn",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols]

def _read_csvs_any(path_like: str) -> pd.DataFrame:
    """Read a single CSV, a directory of CSVs, or a glob pattern into one DataFrame."""
    if not path_like:
        return pd.DataFrame()
    if os.path.isdir(path_like):
        files = sorted(glob.glob(os.path.join(path_like, "*.csv")))
    elif any(ch in path_like for ch in ["*", "?", "["]):
        files = sorted(glob.glob(path_like))
    else:
        files = [path_like]
    frames = []
    for f in files:
        if os.path.isfile(f):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"[WARN] failed reading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _infer_duration_seconds(df: pd.DataFrame):
    for cand in ["dur","duration","win_len","len","secs","seconds","window_duration"]:
        if cand in df.columns:
            s = pd.to_numeric(df[cand], errors="coerce")
            if s.notna().any():
                return s
    start_cands = ["start","sec_start","s","t_start","start_sec"]
    end_cands   = ["end","sec_end","e","t_end","end_sec"]
    start_col = next((c for c in start_cands if c in df.columns), None)
    end_col   = next((c for c in end_cands if c in df.columns), None)
    if start_col and end_col:
        s = pd.to_numeric(df[end_col], errors="coerce") - pd.to_numeric(df[start_col], errors="coerce")
        return s.clip(lower=0)
    return None

def _is_ev_mask(df: pd.DataFrame):
    str_col = _find_col(df, ["strength","str","ev_strength","strength_global"])
    if str_col is not None:
        vals = df[str_col].astype(str).str.lower()
        return (vals == "ev") | (vals == "even") | (vals.str.contains(r"5v5"))
    team_skaters = _find_col(df, ["team_skaters","for_skaters","us_skaters","skaters_for","home_skaters","away_skaters"])
    opp_skaters  = _find_col(df, ["opp_skaters","against_skaters","them_skaters","skaters_against","away_skaters" if team_skaters=="home_skaters" else "home_skaters"]) if team_skaters else None
    if team_skaters and opp_skaters and team_skaters in df.columns and opp_skaters in df.columns:
        a = pd.to_numeric(df[team_skaters], errors="coerce")
        b = pd.to_numeric(df[opp_skaters], errors="coerce")
        return (a == 5) & (b == 5)
    return None

def _process_windows_csv(path: str, season: str) -> pd.DataFrame:
    game_match = re.search(r"(\\d{10})", os.path.basename(path))
    gameid = int(game_match.group(1)) if game_match else None
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["season","gameid","teamid","playerId","ev_minutes"])
    team_col = _find_col(df, ["teamid","team_id","teamId"]) or "teamid"
    player_col = _find_col(df, ["playerId","player_id"]) or "playerId"
    if team_col not in df.columns or player_col not in df.columns:
        return pd.DataFrame(columns=["season","gameid","teamid","playerId","ev_minutes"])
    dur = _infer_duration_seconds(df)
    ev_mask = _is_ev_mask(df)
    if dur is None or ev_mask is None:
        return pd.DataFrame(columns=["season","gameid","teamid","playerId","ev_minutes"])
    work = pd.DataFrame({
        "teamid": pd.to_numeric(df[team_col], errors="coerce"),
        "playerId": pd.to_numeric(df[player_col], errors="coerce"),
        "dur": pd.to_numeric(dur, errors="coerce").fillna(0),
        "is_ev": ev_mask.fillna(False),
    })
    work = work[work["is_ev"]]
    if gameid is not None:
        work["gameid"] = gameid
    else:
        gcol = _find_col(df, ["gameid","game_id","gamePk"])
        work["gameid"] = pd.to_numeric(df[gcol], errors="coerce").astype("Int64") if gcol and gcol in df.columns else pd.NA
    grp = work.groupby(["gameid","teamid","playerId"], dropna=False)["dur"].sum().reset_index()
    grp.rename(columns={"dur":"ev_seconds"}, inplace=True)
    grp["ev_minutes"] = grp["ev_seconds"] / 60.0
    grp["season"] = season
    return grp[["season","gameid","teamid","playerId","ev_minutes"]]

def _build_ev_toi_from_windows(wins_dir: str, season: str) -> pd.DataFrame:
    if os.path.isfile(wins_dir):
        files = [wins_dir]
    else:
        files = sorted(glob.glob(os.path.join(wins_dir, "player_windows_train_*.csv")))
    print(f"[INFO] EV-TOI: scanning {len(files)} window file(s) from {wins_dir}")
    frames = []
    for f in files:
        try:
            print(f"[INFO] EV-TOI from {os.path.basename(f)}")
            frames.append(_process_windows_csv(f, season))
        except Exception as e:
            print(f"[WARN] failed building EV TOI from {f}: {e}")
    if not frames:
        return pd.DataFrame(columns=["season","gameid","teamid","playerId","ev_minutes"])
    return pd.concat(frames, ignore_index=True)

def main(in_path, out_csv, ev_toi_path=None, wins_dir=None, season=None):
    frames = []
    # Allow either a single file or a directory
    if os.path.isfile(in_path) and in_path.lower().endswith('.json'):
        targets = [in_path]
    else:
        targets = sorted(glob.glob(os.path.join(in_path, "pbp_onice_*.json")))
    print(f"[INFO] Found {len(targets)} PBP file(s) under {in_path}")
    for idx, path in enumerate(targets, start=1):
        try:
            print(f"[INFO] [{idx}/{len(targets)}] Processing PBP {os.path.basename(path)}")
            frames.append(process_game(path))
        except Exception as e:
            print(f"[WARN] failed on {path}: {e}")
    # Build EV TOI source (prefer windows_dir if provided; else use ev_toi_path CSVs)
    ev = pd.DataFrame()
    if wins_dir and season:
        print(f"[INFO] Building EV TOI from windows dir={wins_dir} season={season}")
        ev = _build_ev_toi_from_windows(wins_dir, season)
        print(f"[INFO] Built EV TOI rows: {len(ev)}")
    elif ev_toi_path:
        print(f"[INFO] Loading EV TOI from {ev_toi_path}")
        ev = _read_csvs_any(ev_toi_path)
        print(f"[INFO] Loaded EV TOI rows: {len(ev)}")

    if not frames:
        base = pd.DataFrame(columns=["gameid","teamid","playerId"])
        # If EV TOI available, enrich empty penalties with EV TOI to include all players
        if not ev.empty:
            merged = ev.merge(base, on=["gameid","teamid","playerId"], how="left")
            # ensure parent directory exists
            out_parent = os.path.dirname(out_csv)
            if out_parent and not os.path.isdir(out_parent):
                os.makedirs(out_parent, exist_ok=True)
            merged.to_csv(out_csv, index=False)
            print(f"Wrote {out_csv} with {len(merged)} rows (no penalties found; EV TOI only)")
            return
        out_parent = os.path.dirname(out_csv)
        if out_parent and not os.path.isdir(out_parent):
            os.makedirs(out_parent, exist_ok=True)
        base.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} with 0 rows (no inputs matched)")
        return

    pen = pd.concat(frames, ignore_index=True).fillna({"teamid": -1})  # teamid -1 if unknown; you can backfill later
    print(f"[INFO] Aggregated penalties rows: {len(pen)}")

    if not ev.empty:
        # Ensure expected columns exist on penalties before merging
        merged = ev.merge(pen, on=["gameid","teamid","playerId"], how="left")
        # Fill missing penalty counts with zeros
        penalty_cols = [c for c in merged.columns if c.startswith("pen_") or c in ("taken","drawn","coincidental_flag")]
        for c in penalty_cols:
            merged[c] = merged[c].fillna(0)
        out_parent = os.path.dirname(out_csv)
        if out_parent and not os.path.isdir(out_parent):
            os.makedirs(out_parent, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv} with {len(merged)} rows (EV TOI merged)")
        return

    out_parent = os.path.dirname(out_csv)
    if out_parent and not os.path.isdir(out_parent):
        os.makedirs(out_parent, exist_ok=True)
    pen.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(pen)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build per-game penalties; optionally merge EV TOI from EV CSVs or directly from player windows.")
    ap.add_argument("pbp_in", help="PBP on-ice JSON dir or single file")
    ap.add_argument("out_csv", help="Output CSV path")
    ap.add_argument("--ev", dest="ev_path", help="EV TOI CSV|dir|glob (optional)")
    ap.add_argument("--wins_dir", help="Directory of player_windows_train_*.csv to derive EV TOI (optional)")
    ap.add_argument("--season", help="Season like 20162017 (required with --wins_dir)")
    args = ap.parse_args()
    if args.wins_dir and not args.season:
        print("--season is required when using --wins_dir"); sys.exit(1)
    main(args.pbp_in, args.out_csv, ev_toi_path=args.ev_path, wins_dir=args.wins_dir, season=args.season)
