#!/usr/bin/env python3
"""
Build training CSVs from api-web raw dumps (shots + panel) with engineered
features for xG and causal modeling.

Adds team/opponent abbreviations, per-game goalie SV%, and REG-season league
ranks for model context pulled from standings_by_date_<season>.csv:
  * REG games: rank ON the game date (same as print_standings_by_date.py)
  * PLAYOFF games: FINAL REG-SEASON rank (max date in the CSV)

Run:
  python build_training_from_raw.py --game 2024010086 \
      --raw artifacts/dumps/raw --out artifacts/training --only-team-abbr TOR
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import bisect
from datetime import datetime

SECONDS_PER_PERIOD = 20 * 60
MAX_SECONDS = SECONDS_PER_PERIOD * 6
NET_X = 89.0

# ---- Rush heuristics (tunable)
TSF_RUSH = 5
TSF_BURST = 15
TRANS_DT = 8
MIN_DX   = 50
MIN_VX   = 6.0

EVENTS_CONTEXT = {
    "faceoff", "giveaway", "takeaway", "hit",
    "blocked-shot", "shot-blocked", "shot blocked",
    "missed-shot", "shot-missed", "missed shot",
    "shot-on-goal", "shot on goal", "goal",
    "icing", "offside", "stoppage", "puck-out-of-play", "goalie-stopped",
    "penalty",
}

# ==== Empty-Net xG heuristic (distance + light angle penalty) ====
def en_xg(distance_ft: float, abs_angle_deg: Optional[float] = None) -> float:
    """
    Logistic in feet; ~0.5 at ~64 ft, ~0.75 at ~48 ft, ~0.25 at ~80 ft.
    Angle softly penalizes wide looks (max 10% at 90°). Clipped to [0.02, 0.98].
    """
    z = 0.09 * (distance_ft - 64.0)
    p = 1.0 / (1.0 + math.exp(z))
    if abs_angle_deg is not None:
        penalty = 1.0 - 0.10 * min(max(abs_angle_deg, 0.0), 90.0) / 90.0
        p *= penalty
    return float(max(0.02, min(0.98, p)))

# -------------------- IO --------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_csv_with_fields(rows: List[Dict[str, Any]], path: str, fields_in_order: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_in_order, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row_out = {k: r.get(k, "") for k in fields_in_order}
            w.writerow(row_out)

# -------------------- xG loader --------------------

def load_xg_index(jsonl_path: str) -> dict:
    """
    Build {(gamePk, period, sec_game, sortOrder): xg} -> float
    from a JSONL scored by score_xg.py
    """
    idx = {}
    if not jsonl_path or not os.path.exists(jsonl_path):
        return idx
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    key = (int(r["gamePk"]), int(r["period"]), int(r["sec_game"]), int(r["sortOrder"]))
                    idx[key] = float(r["xg"])
                except Exception:
                    # ignore malformed rows
                    pass
    except Exception:
        return idx
    return idx

# -------------------- Paths --------------------

@dataclass
class Inputs:
    game_pk: int
    raw_dir: str
    out_dir: str

    def _resolve_multi(self, cands: List[str]) -> str:
        for rel in cands:
            p = os.path.join(self.raw_dir, rel)
            if os.path.exists(p):
                return p
        return os.path.join(self.raw_dir, cands[0])

    @property
    def pbp_path(self) -> str:
        gid = str(self.game_pk)
        return self._resolve_multi([
            os.path.join("pbp", f"{gid}.json"),
            os.path.join("playbyplay", f"{gid}.json"),
            f"pbp_{gid}.json",
        ])

    @property
    def box_path(self) -> str:
        gid = str(self.game_pk)
        return self._resolve_multi([
            os.path.join("boxscore", f"{gid}.json"),
            os.path.join("box", f"{gid}.json"),
            f"box_{gid}.json",
        ])

    @property
    def shifts_path(self) -> str:
        gid = str(self.game_pk)
        return self._resolve_multi([
            os.path.join("shiftcharts", f"{gid}.json"),
            os.path.join("shifts", f"{gid}.json"),
            f"shifts_{gid}.json",
        ])

# -------------------- Time helpers --------------------

def sec_from_period_clock(period: int, mmss: str) -> int:
    try:
        m, s = mmss.split(":")
        return (period - 1) * SECONDS_PER_PERIOD + int(m) * 60 + int(s)
    except Exception:
        return 0

# -------------------- Box helpers --------------------

def get_home_away_ids_from_box(box: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    for root in ("teamByGameStats", None):
        src = box.get(root, {}) if root else box
        h = (src or {}).get("homeTeam", {}) or {}
        a = (src or {}).get("awayTeam", {}) or {}
        hid = h.get("id") or h.get("teamId")
        aid = a.get("id") or a.get("teamId")
        if hid and aid:
            return int(hid), int(aid)
    return None, None

def build_player_maps_from_box(box: Dict[str, Any]):
    pid_to_tid: Dict[int, int] = {}
    pid_to_abbr: Dict[int, str] = {}
    pid_to_pos: Dict[int, str] = {}
    pgs = (box.get("playerByGameStats") or {})
    for side in ("homeTeam", "awayTeam"):
        team = (box.get(side) or {})
        tid = team.get("id") or team.get("teamId")
        abbr = team.get("abbrev") or team.get("triCode")
        stats = (pgs.get(side) or {})
        for group, code in (("forwards","F"),("defense","D"),("goalies","G")):
            for p in (stats.get(group) or []):
                pid = p.get("playerId")
                if isinstance(pid, int) and isinstance(tid, int):
                    pid_to_tid[pid] = int(tid)
                    pid_to_abbr[pid] = str(abbr)
                    pid_to_pos[pid] = code
    return pid_to_tid, pid_to_abbr, pid_to_pos

# -------------------- Goalie SV merge --------------------

GOALIE_FILE_PAT = re.compile(r"(?i)^(?:[a-z]{3}_)?goalies_.*\.json$")
TEAM_FROM_NAME  = re.compile(r"(?i)^([a-z]{3})_goalies_.*\.json$")

def _choose_team_goalie_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    starters = [r for r in rows if bool(r.get("starter"))]
    if starters:
        return starters[0]
    rows2 = sorted(rows, key=lambda r: int(r.get("toi_seconds", 0)), reverse=True)
    if rows2 and int(rows2[0].get("toi_seconds", 0)) > 0:
        return rows2[0]
    rows3 = sorted(rows, key=lambda r: int(r.get("shots_all", 0)), reverse=True)
    return rows3[0]

# New: map (gamePk, goalie_id) -> SV info so we can vary opp_sv per shot by faced goalie
def load_goalie_sv_by_game_goalie(raw_dir: str, debug: bool = False, debug_gid: Optional[int] = None) -> Dict[Tuple[int, int], Dict[str, Any]]:
    sv_by_goalie: Dict[Tuple[int, int], Dict[str, Any]] = {}
    if not os.path.isdir(raw_dir):
        return sv_by_goalie
    all_names = sorted(os.listdir(raw_dir))
    for fname in all_names:
        if not GOALIE_FILE_PAT.match(fname):
            continue
        fpath = os.path.join(raw_dir, fname)
        try:
            data = load_json(fpath)
        except Exception:
            continue
        rows = data if isinstance(data, list) else []
        for r in rows:
            gid = r.get("gamePk") or r.get("game_id") or r.get("game")
            gpid = r.get("goalie_id") or r.get("player_id")
            if not (isinstance(gid, int) and isinstance(gpid, int)):
                continue
            shots = int(r.get("shots_all", 0))
            saves = int(r.get("saves_all", 0))
            sv_game = (float(saves) / float(shots)) if shots > 0 else None
            pre_raw = r.get("pregame_sv_all")
            try:
                sv_pre = float(pre_raw) if pre_raw is not None and pre_raw != "" else None
            except Exception:
                sv_pre = None
            key = (gid, gpid)
            cur = sv_by_goalie.get(key)
            # prefer entries where played is true, else higher TOI
            better = False
            if cur is None:
                better = True
            else:
                cur_played = bool(cur.get("played"))
                new_played = bool(r.get("played"))
                if new_played and not cur_played:
                    better = True
                elif (int(r.get("toi_seconds", 0)) > int(cur.get("toi_seconds", 0))):
                    better = True
            if better:
                sv_by_goalie[key] = {
                    "sv_game": sv_game,
                    "sv_pregame_all": sv_pre,
                    "team_abbr": (r.get("team_abbr") or r.get("team")),
                    "played": r.get("played"),
                    "toi_seconds": int(r.get("toi_seconds", 0)),
                }
    return sv_by_goalie

def load_goalie_sv_by_game_team(raw_dir: str, debug: bool = False, debug_gid: Optional[int] = None) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Scan --raw for *goalies*.json files and build mapping
    (gamePk, TEAM_ABBR) -> {sv_game, goalie_id, goalie_name}.
    """
    sv_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not os.path.isdir(raw_dir):
        if debug:
            print(f"[goalie-debug] raw_dir does not exist or is not a directory: {raw_dir}")
        return sv_map

    all_names = sorted(os.listdir(raw_dir))
    if debug:
        print(f"[goalie-debug] scanning directory: {raw_dir}")
        print(f"[goalie-debug] found {len(all_names)} entries")
    for fname in all_names:
        if not GOALIE_FILE_PAT.match(fname):
            if debug:
                print(f"[goalie-debug] skipping non-matching name: {fname}")
            continue
        fpath = os.path.join(raw_dir, fname)
        if debug:
            print(f"[goalie-debug] considering file: {fname}")
        try:
            data = load_json(fpath)
        except Exception as e:
            if debug:
                print(f"[goalie-debug]   failed to load JSON: {e}; skipping {fname}")
            continue
        # normalize to list
        if not isinstance(data, list):
            if debug:
                tname = type(data).__name__
                print(f"[goalie-debug]   top-level JSON is {tname}, expected list; skipping {fname}")
            rows = []
        else:
            rows = data
        # If file name is like "tor_goalies_*.json" capture team hint (optional)
        team_hint = None
        m = TEAM_FROM_NAME.match(fname)
        if m:
            team_hint = m.group(1).upper()
        # group rows by (gamePk, team_abbr)
        by_key: Dict[Tuple[int,str], List[Dict[str,Any]]] = defaultdict(list)
        for r in rows:
            gid = r.get("gamePk") or r.get("game_id") or r.get("game")
            ab  = (r.get("team_abbr") or r.get("team") or team_hint)
            if not (isinstance(gid, int) and isinstance(ab, str) and len(ab) in (3,4)):
                continue
            by_key[(gid, ab.upper())].append(r)
        added_keys = 0
        for key, lst in by_key.items():
            if debug and (debug_gid is None or key[0] == debug_gid):
                gid_dbg, abbr_dbg = key
                print(f"[goalie-debug] candidates for game={gid_dbg} team={abbr_dbg} from {fname}: {len(lst)}")
                for i, rr in enumerate(lst, 1):
                    print(f"[goalie-debug]   cand[{i}] id={rr.get('goalie_id')} name={rr.get('goalie_name')} shots={rr.get('shots_all')} saves={rr.get('saves_all')} toi_s={rr.get('toi_seconds')} starter={rr.get('starter')} played={rr.get('played')} pregame_sv_all={rr.get('pregame_sv_all')}")
            pick = _choose_team_goalie_row(lst)
            if not pick:
                continue
            shots = int(pick.get("shots_all", 0))
            saves = int(pick.get("saves_all", 0))
            sv = (float(saves) / float(shots)) if shots > 0 else None
            # cumulative pregame SV (if present in source JSON)
            pre_raw = pick.get("pregame_sv_all")
            try:
                sv_pregame_all = float(pre_raw) if pre_raw is not None and pre_raw != "" else None
            except Exception:
                sv_pregame_all = None
            sv_map[key] = {
                "sv_game": sv,
                "sv_pregame_all": sv_pregame_all,
                "goalie_id": pick.get("goalie_id"),
                "goalie_name": pick.get("goalie_name"),
                "shots": shots,
                "saves": saves,
            }
            added_keys += 1
            if debug and (debug_gid is None or key[0] == debug_gid):
                print(f"[goalie-debug]   chosen id={pick.get('goalie_id')} name={pick.get('goalie_name')} shots={shots} saves={saves} toi_s={pick.get('toi_seconds')} starter={pick.get('starter')} played={pick.get('played')} sv_game={sv} sv_pregame_all={sv_pregame_all}")
        if debug:
            print(f"[goalie-debug]   loaded {added_keys} keys from {fname}")
    return sv_map

# -------------------- Shifts → on-ice index --------------------

def build_onice_index(shift_rows: List[Dict[str, Any]]):
    """
    Build per-second on-ice sets (deduped) and player TOI.
    - Treat intervals as half-open [start, end): player is on from start sec up to but not including end sec.
    - Use sets so duplicate shift rows for the same player/time do not double-count.
    """
    onice_sets: List[set[int]] = [set() for _ in range(MAX_SECONDS + 1)]
    toi: Dict[int, int] = defaultdict(int)

    for r in shift_rows:
        pid = r.get("playerId") or r.get("player_id")
        period = r.get("period") or r.get("periodNumber")
        start = r.get("startTime") or r.get("start_time")
        end   = r.get("endTime") or r.get("end_time")
        if not (isinstance(pid, int) and period and start and end):
            continue

        s = sec_from_period_clock(int(period), str(start))
        e = sec_from_period_clock(int(period), str(end))
        if e <= s:
            continue

        s = max(0, min(s, MAX_SECONDS))
        e = max(0, min(e, MAX_SECONDS))

        toi[pid] += (e - s)
        # Dedup by using a set:
        for t in range(s, e):
            onice_sets[t].add(pid)

    # Freeze sets into sorted lists for downstream compatibility
    onice: List[List[int]] = [sorted(list(s)) for s in onice_sets]
    return onice, toi

# -------------------- PBP parsing --------------------

def detect_pbp_source(doc: Dict[str, Any]) -> str:
    return "statsapi" if "liveData" in (doc or {}) else "apiweb"

def iter_apiweb_plays(gc: Dict[str, Any]):
    for p in (gc or {}).get("plays", []) or []:
        pd = p.get("periodDescriptor") or {}
        sec = sec_from_period_clock(int(pd.get("number", 1)), p.get("timeInPeriod", "00:00"))
        so  = int(p.get("sortOrder", 0))
        yield sec, so, p

# --- NEW: earliest penalty order per second (api-web PBP) ---
def penalty_min_sort_by_sec_apiweb(gc: Dict[str, Any]) -> Dict[int, int]:
    """Return {sec_game: earliest_penalty_sortOrder} for the game.
    Used to detect shots that occur before any penalty at the same second.
    """
    min_so: Dict[int, int] = {}
    for sec, so, p in iter_apiweb_plays(gc):
        kind = (p.get("typeDescKey") or "").lower()
        if kind != "penalty":
            continue
        cur = min_so.get(sec)
        if cur is None or so < cur:
            min_so[sec] = int(so)
    return min_so

# -------------------- Penalties & goals → manpower --------------------

def collect_penalties_and_goals_apiweb(gc: Dict[str, Any]):
    pens, goals = [], []
    for sec, _so, p in iter_apiweb_plays(gc):
        kind = (p.get("typeDescKey") or "").lower()
        det = p.get("details") or {}
        tid = p.get("teamId") or det.get("eventOwnerTeamId")
        if kind == "goal":
            if isinstance(tid, int):
                goals.append({"team": int(tid), "sec": sec})
            continue
        if kind != "penalty":
            continue
        sev = (det.get("penaltySeverity") or "").lower()
        mins = det.get("duration") or det.get("penaltyMinutes")
        if isinstance(mins, int):
            dur = 300 if mins >= 5 else (240 if mins >= 4 else 120)
            tag = "major" if mins >= 5 else ("double-minor" if mins >= 4 else "minor")
        else:
            tag = "major" if sev == "major" else ("double-minor" if "double" in sev else "minor")
            dur = 300 if tag == "major" else (240 if tag == "double-minor" else 120)
        pens.append({"team": int(det.get("committedByTeamId") or tid) if isinstance(det.get("committedByTeamId") or tid, int) else None,
                     "sec": sec, "kind": tag, "dur": dur})
    return pens, sorted(goals, key=lambda x: x["sec"])

def manpower_timeline_from_penalties(gc: Dict[str, Any], home_id: int, away_id: int) -> List[Tuple[int,int]]:
    horizon = MAX_SECONDS
    home = [5] * (horizon + 1)
    away = [5] * (horizon + 1)
    pens, goals = collect_penalties_and_goals_apiweb(gc)
    by_sec: Dict[int, List[dict]] = defaultdict(list)
    for p in pens:
        by_sec[p["sec"].__int__() if hasattr(p["sec"], "__int__") else p["sec"]].append(p)
    active = {home_id: [], away_id: []}
    gi = 0
    for s in range(horizon + 1):
        if s in by_sec:
            mh = sum(1 for p in by_sec[s] if p["team"] == home_id and p["kind"] in ("minor","double-minor"))
            ma = sum(1 for p in by_sec[s] if p["team"] == away_id and p["kind"] in ("minor","major","double-minor"))
            coincident = (mh > 0 and mh == ma)
            for p in by_sec[s]:
                tid = p["team"]
                if p["kind"] == "double-minor":
                    active.setdefault(tid, []).append({"kind":"minor","remain":120,"coincident":coincident})
                    active[tid].append({"kind":"minor","remain":120,"coincident":coincident})
                else:
                    active.setdefault(tid, []).append({"kind":p["kind"],"remain":p["dur"],"coincident":coincident})
        h_red = sum(1 for pen in active.get(home_id, []) if pen["kind"] in ("minor","major") and not pen.get("coincident"))
        a_red = sum(1 for pen in active.get(away_id, []) if pen["kind"] in ("minor","major") and not pen.get("coincident"))
        h_sk = max(3, 5 - h_red)
        a_sk = max(3, 5 - a_red)
        while gi < len(goals) and goals[gi]["sec"] == s:
            gteam = goals[gi]["team"]
            if h_sk != a_sk:
                adv = home_id if h_sk > a_sk else away_id
                dis = away_id if adv == home_id else home_id
                if gteam == adv:
                    bucket = active.get(dis, [])
                    idx = next((i for i,pen in enumerate(bucket) if pen["kind"]=="minor" and not pen.get("coincident") and pen["remain"]>0), None)
                    if idx is not None:
                        bucket[idx]["remain"] = 0
            gi += 1
        home[s], away[s] = h_sk, a_sk
        for tid in (home_id, away_id):
            nxt = []
            for pen in active.get(tid, []):
                if pen["remain"] > 0:
                    pen["remain"] -= 1
                if pen["remain"] > 0:
                    nxt.append(pen)
            active[tid] = nxt
    return list(zip(home, away))

# -------------------- Shots --------------------

def extract_shots_apiweb(gc: Dict[str, Any]):
    rows = []
    for sec, so, p in iter_apiweb_plays(gc):
        t_raw = (p.get("typeDescKey") or "").lower()
        t = (
            "goal" if t_raw == "goal"
            else "shot-on-goal" if ("shot" in t_raw and "goal" in t_raw)
            else "missed-shot" if ("miss" in t_raw)
            else "blocked-shot" if ("block" in t_raw)
            else t_raw
        )
        # keep unblocked attempts only: goals + SOG + MISS
        if t not in ("goal", "shot-on-goal", "missed-shot"):
            continue
        d = p.get("details") or {}
        row = {
            "gamePk": gc.get("id") or gc.get("gamePk"),
            "period": (p.get("periodDescriptor") or {}).get("number"),
            "time": p.get("timeInPeriod"),
            "sec": sec,
            "sortOrder": so,
            "x": d.get("xCoord"),
            "y": d.get("yCoord"),
            "shooterId": d.get("shootingPlayerId") or d.get("scoringPlayerId") or d.get("shooterId"),
            "goalieId": d.get("goalieInNetId") or d.get("goalieId"),
            "shotType": d.get("shotType"),
            "eventType": t,
            "isGoal": 1 if t == "goal" else 0,
            "on_target": 1 if t in ("goal","shot-on-goal") else 0,
            "is_unblocked": 1,
        }
        if t == "goal":
            row["attempt_outcome"] = "GOAL"
        elif t == "shot-on-goal":
            row["attempt_outcome"] = "SAVED"
        else:
            row["attempt_outcome"] = "MISSED"
        rows.append(row)
    return rows

# -------------------- Geometry --------------------

def flip_xy_to_attacking(xv: Optional[float], yv: Optional[float]):
    x = float(xv) if xv is not None else 0.0
    y = float(yv) if yv is not None else 0.0
    sgn = 1.0 if x >= 0 else -1.0
    return abs(x), y * sgn

def geom(xv: Optional[float], yv: Optional[float]):
    xf, yf = flip_xy_to_attacking(xv, yv)
    dx = max(NET_X - xf, 1e-6)
    dist = (dx*dx + yf*yf) ** 0.5
    ang_abs = math.degrees(math.atan2(abs(yf), dx))
    ang_signed = math.degrees(math.atan2(yf, dx))
    return xf, yf, dist, ang_abs, ang_signed

# -------------------- TSF (play-aware) --------------------

def build_faceoff_index(gc: Dict[str, Any]) -> List[Tuple[int,int]]:
    faceoffs: List[Tuple[int,int]] = []
    for sec, so, p in iter_apiweb_plays(gc):
        if (p.get("typeDescKey") or "").lower() == "faceoff":
            faceoffs.append((sec, so))
    faceoffs.sort(key=lambda t: (t[0], t[1]))
    return faceoffs

def tsf_for_play(sec_shot: int, so_shot: int, faceoffs: List[Tuple[int,int]]) -> int:
    if not faceoffs:
        return 9999
    secs = [s for (s, _) in faceoffs]
    i = bisect.bisect_right(secs, sec_shot) - 1
    while i >= 0:
        fo_sec, fo_so = faceoffs[i]
        if fo_sec < sec_shot or (fo_sec == sec_shot and fo_so < so_shot):
            return sec_shot - fo_sec
        i -= 1
    return 9999

# ---- Previous event maps -----------------------

def build_prev_same_team_map(gc: Dict[str, Any]):
    prev_by_team: Dict[int, Dict[str, Any]] = {}
    prev_map: Dict[Tuple[int,int], Optional[Dict[str, Any]]] = {}
    last_period = None
    for sec, so, p in iter_apiweb_plays(gc):
        pd = (p.get("periodDescriptor") or {})
        period = int(pd.get("number", 1))
        if period != last_period:
            prev_by_team = {}
            last_period = period
        det  = p.get("details") or {}
        team = p.get("teamId") or det.get("eventOwnerTeamId")
        x = det.get("xCoord"); y = det.get("yCoord")
        kind = (p.get("typeDescKey") or "").lower()
        if isinstance(team, int):
            prev_map[(sec, so)] = prev_by_team.get(team)
            if x is not None and y is not None:
                prev_by_team[team] = {"sec": int(sec), "x": float(x), "y": float(y), "type": kind}
        else:
            prev_map[(sec, so)] = None
    return prev_map

def build_prev_global_any_map(gc: Dict[str, Any]):
    last = None
    out = {}
    last_period = None
    for sec, so, p in iter_apiweb_plays(gc):
        pd = p.get("periodDescriptor") or {}
        period = int(pd.get("number", 1))
        if period != last_period:
            last = None
            last_period = period
        kind = (p.get("typeDescKey") or "").lower()
        det  = p.get("details") or {}
        team = p.get("teamId") or det.get("eventOwnerTeamId")
        out[(sec, so)] = last
        if kind in EVENTS_CONTEXT:
            last = {"sec": int(sec), "type": kind, "team": int(team) if isinstance(team, int) else None}
    return out

def build_prev_global_xy_map(gc: Dict[str, Any]):
    last = None
    out = {}
    last_period = None
    for sec, so, p in iter_apiweb_plays(gc):
        pd = p.get("periodDescriptor") or {}
        period = int(pd.get("number", 1))
        if period != last_period:
            last = None
            last_period = period
        det  = p.get("details") or {}
        kind = (p.get("typeDescKey") or "").lower()
        team = p.get("teamId")or det.get("eventOwnerTeamId")
        x = det.get("xCoord"); y = det.get("yCoord")
        out[(sec, so)] = last
        if kind in EVENTS_CONTEXT and x is not None and y is not None and isinstance(team, int):
            last = {"sec": int(sec), "x": float(x), "y": float(y),
                    "type": kind, "team": int(team)}
    return out

# -------------------- Standings snapshot helpers --------------------

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

def _extract_game_date_iso(pbp: Dict[str, Any], box: Dict[str, Any]) -> Optional[str]:
    """
    Try a bunch of likely spots & fallback to regex to get YYYY-MM-DD.
    """
    cand = (
        (box.get("gameDate"))
        or ((box.get("gameInfo") or {}).get("gameDate"))
        or (pbp.get("gameDate"))
        or (pbp.get("gameDateISO"))
        or (pbp.get("startTimeUTC"))
        or (pbp.get("start_time"))
        or (pbp.get("game") or {}).get("startTimeUTC")
    )
    if isinstance(cand, str):
        m = DATE_RE.search(cand)
        if m:
            return m.group(1)
    # sometimes date is nested as startTimeUTC-like fields inside gameInfo
    for k in ("startTimeUTC","start_time","gameDate"):
        v = (box.get("gameInfo") or {}).get(k)
        if isinstance(v, str):
            m = DATE_RE.search(v)
            if m:
                return m.group(1)
    return None

def _find_standings_by_date_csv(raw_dir: str, season: str) -> Optional[str]:
    """
    Prefer <raw>/standings/standings_by_date_<season>.csv
    Fallback to artifacts/standings/standings_by_date_<season>.csv
    """
    fname = f"standings_by_date_{season}.csv"
    p1 = os.path.join(raw_dir, "standings", fname)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("artifacts", "standings", fname)
    if os.path.exists(p2):
        return p2
    return None

def _rank_from_row(row: Dict[str, Any]) -> Optional[int]:
    v = row.get("league_rank_unique") or row.get("league_rank") or ""
    try:
        r = int(v)
        return r if r > 0 else None
    except Exception:
        return None

def _get_snapshot_ranks(standings_csv: str, date_iso: Optional[str], use_final: bool,
                        needed_team_ids: List[int]) -> Dict[int, Optional[int]]:
    """
    If use_final=True -> get rank at the MAX date in CSV.
    Else -> get rank at date_iso.
    Only returns entries for needed_team_ids; missing -> None.
    """
    by_date: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    all_dates: set[str] = set()
    with open(standings_csv, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            d = row.get("date")
            try:
                tid = int(row.get("team_id") or 0)
            except Exception:
                continue
            if not d or tid == 0:
                continue
            by_date[d][tid] = row
            all_dates.add(d)
    if not all_dates:
        return {tid: None for tid in needed_team_ids}

    target_date: Optional[str]
    if use_final:
        target_date = max(all_dates)  # lex works for ISO yyyy-mm-dd
    else:
        target_date = date_iso

    ranks: Dict[int, Optional[int]] = {tid: None for tid in needed_team_ids}
    snap = by_date.get(target_date or "", {})
    for tid in needed_team_ids:
        row = snap.get(tid)
        ranks[tid] = _rank_from_row(row) if row else None
    return ranks

# -------------------- Build --------------------

def build(game_pk: int, raw_dir: str, out_dir: str, only_team_abbr: Optional[str] = None, debug_goalies: bool = False, debug_strength: bool = False, shots_train_only: bool = False, xg_jsonl: Optional[str] = None) -> None:
    ensure_out_dir(out_dir)
    inp = Inputs(game_pk, raw_dir, out_dir)

    pbp = load_json(inp.pbp_path)
    box = load_json(inp.box_path)
    shifts = load_json(inp.shifts_path)
    shift_rows = (shifts.get("data") if isinstance(shifts, dict) and "data" in shifts else shifts) or []

    if "liveData" in (pbp or {}):
        raise RuntimeError("Expect api-web PBP at raw/pbp/{id}.json; your file looks like statsapi.")

    # Determine playoff flag
    def _game_type_from_box(b: Dict[str, Any]) -> str:
        gt = b.get("gameType") or (b.get("gameInfo") or {}).get("gameType")
        if isinstance(gt, int):
            return "PRE" if gt == 1 else ("REG" if gt == 2 else ("PST" if gt == 3 else str(gt)))
        if isinstance(gt, str):
            v = gt.strip().upper()
            if v in ("1","PRE","PRESEASON"): return "PRE"
            if v in ("2","REG","REGULAR"):   return "REG"
            if v in ("3","PST","PLAYOFFS","POSTSEASON"): return "PST"
            return v
        return ""
    game_type_code = _game_type_from_box(box)
    game_is_playoff = 1 if game_type_code == "PST" else 0

    # Goalie SV% maps
    goalie_sv_by_game_team = load_goalie_sv_by_game_team(raw_dir, debug=debug_goalies, debug_gid=(game_pk if debug_goalies else None))
    goalie_sv_by_game_goalie = load_goalie_sv_by_game_goalie(raw_dir, debug=debug_goalies, debug_gid=(game_pk if debug_goalies else None))

    home_id, away_id = get_home_away_ids_from_box(box)
    if home_id is None or away_id is None:
        raise RuntimeError("Could not determine home/away team IDs from box JSON.")

    home_abbr = (box.get("homeTeam") or {}).get("abbrev") or (box.get("homeTeam") or {}).get("triCode")
    away_abbr = (box.get("awayTeam") or {}).get("abbrev") or (box.get("awayTeam") or {}).get("triCode")

    # Load xG index if provided
    xg_idx: Dict[Tuple[int,int,int,int], float] = load_xg_index(xg_jsonl) if xg_jsonl else {}

    # Load teams_meta (for conf/div)
    teams_meta_path = os.path.join(raw_dir, "teams_meta.json")
    teams_meta: Dict[str, Dict[str, Any]] = {}
    try:
        if os.path.exists(teams_meta_path):
            teams_meta = load_json(teams_meta_path) or {}
            if not isinstance(teams_meta, dict):
                teams_meta = {}
    except Exception:
        teams_meta = {}

    # ---------- REG-season ranks from standings_by_date_<season>.csv
    season_from_box = str(box.get("season") or "")
    standings_rank_pre: Dict[int, Optional[int]] = {}
    try:
        if season_from_box and len(season_from_box) == 8:
            standings_csv = _find_standings_by_date_csv(raw_dir, season_from_box)
            if standings_csv:
                game_date_iso = _extract_game_date_iso(pbp, box)
                use_final = (game_type_code == "PST")
                ranks = _get_snapshot_ranks(
                    standings_csv,
                    date_iso=game_date_iso,
                    use_final=use_final,
                    needed_team_ids=[home_id, away_id],
                )
                standings_rank_pre[home_id] = ranks.get(home_id)
                standings_rank_pre[away_id] = ranks.get(away_id)
            else:
                standings_rank_pre = {}
    except Exception:
        standings_rank_pre = {}

    pid_to_tid, pid_to_abbr, pid_to_pos = build_player_maps_from_box(box)
    onice, toi = build_onice_index(shift_rows)

    shots = extract_shots_apiweb(pbp)
    # Earliest penalty sortOrder per second for pre-penalty detection
    penalty_min_so_by_sec = penalty_min_sort_by_sec_apiweb(pbp)
    skaters_pen = manpower_timeline_from_penalties(pbp, home_id, away_id)

    # --- Build on-ice skater counts (exclude goalies) from shift charts
    max_sec = min(MAX_SECONDS, len(onice) - 1)
    skaters_from_onice: List[Tuple[int,int]] = [(5,5)] * (max_sec + 1)
    for s in range(max_sec + 1):
        h_skaters = a_skaters = 0
        for pid in onice[s]:
            pos = (pid_to_pos.get(pid) or "").upper()
            tid = pid_to_tid.get(pid)
            if pos != "G" and isinstance(tid, int):
                if tid == home_id:
                    h_skaters += 1
                elif tid == away_id:
                    a_skaters += 1
        skaters_from_onice[s] = (h_skaters, a_skaters)

    # Score by second
    score_ev = []
    h = a = 0
    for sec, _so, p in iter_apiweb_plays(pbp):
        if (p.get("typeDescKey") or "").lower() == "goal":
            d = p.get("details") or {}
            h = d.get("homeScore", h); a = d.get("awayScore", a)
            score_ev.append((sec, int(h), int(a)))
    score_ev.sort(key=lambda x: x[0])
    score_by_sec = [(0,0)] * (max_sec + 1)
    i = 0; hs = 0; as_ = 0
    for s in range(max_sec + 1):
        while i < len(score_ev) and score_ev[i][0] <= s:
            hs, as_ = score_ev[i][1], score_ev[i][2]
            i += 1
        score_by_sec[s] = (hs, as_)

    # TSF per second
    fo_secs = set()
    for sec, _so, p in iter_apiweb_plays(pbp):
        if (p.get("typeDescKey") or "").lower() == "faceoff":
            fo_secs.add(sec)
    tsf_at_sec = {}
    tsf = 9999
    for s in range(max_sec + 1):
        if (s-1) in fo_secs:
            tsf = 0
        else:
            tsf = (tsf + 1) if tsf != 9999 else 9999
        tsf_at_sec[s] = tsf

    faceoff_index = build_faceoff_index(pbp)

    # Per-period last GOAL (not last admin play)
    last_goal_by_period: Dict[int, Tuple[int,int]] = {}
    # Chronological unblocked attempts by team (GOAL/SOG/MISS) to backfill previous faced goalie
    attempts_by_team: Dict[int, List[Tuple[int,int]]] = defaultdict(list)
    for sec_ev, so_ev, p in iter_apiweb_plays(pbp):
        pd = (p.get("periodDescriptor") or {})
        per = int(pd.get("number", 1))
        t_raw = (p.get("typeDescKey") or "").lower()
        t = (
            "goal" if t_raw == "goal" else
            ("shot-on-goal" if ("shot" in t_raw and "goal" in t_raw) else
             ("missed-shot" if ("miss" in t_raw) else t_raw))
        )
        if t == "goal":
            cur = last_goal_by_period.get(per)
            if (cur is None) or (sec_ev > cur[0]) or (sec_ev == cur[0] and so_ev > cur[1]):
                last_goal_by_period[per] = (sec_ev, so_ev)
        if t in ("goal","shot-on-goal","missed-shot"):
            team_ev = p.get("teamId") or (p.get("details") or {}).get("eventOwnerTeamId")
            if isinstance(team_ev, int):
                attempts_by_team[team_ev].append((sec_ev, so_ev))
    for lst in attempts_by_team.values():
        lst.sort(key=lambda x: (x[0], x[1]))

    # Transition triggers
    triggers_by_team = defaultdict(list)
    for sec_ev, _so_ev, p in iter_apiweb_plays(pbp):
        kind = (p.get("typeDescKey") or "").lower()
        det  = p.get("details") or {}
        team = p.get("teamId") or det.get("eventOwnerTeamId")
        x = det.get("xCoord"); y = det.get("yCoord")
        if not isinstance(team, int):
            continue
        trig_for: Optional[int] = None
        label: Optional[str] = None
        if kind in ("takeaway",):
            trig_for = team; label = "turnover"
        elif kind in ("giveaway",):
            trig_for = home_id if team == away_id else away_id; label = "turnover"
        elif kind in ("blocked-shot", "shot-blocked", "shot blocked", "missed-shot", "shot-missed", "missed shot"):
            trig_for = home_id if team == away_id else away_id; label = "opp_attempt"
        elif kind == "faceoff":
            z = (det.get("zoneCode") or "").upper()
            win_id = det.get("winningTeamId") or det.get("eventOwnerTeamId")
            if z == "D" and isinstance(win_id, int):
                trig_for = win_id; label = "fo_d_zone"
        if trig_for is not None:
            triggers_by_team[trig_for].append({"sec": sec_ev, "x": x, "y": y, "label": label})

    prev_same_team  = build_prev_same_team_map(pbp)
    prev_global_any = build_prev_global_any_map(pbp)
    prev_global_xy  = build_prev_global_xy_map(pbp)

    panel: Dict[int, Dict[str, Any]] = {}
    pid_to_name: Dict[int, str] = {}
    def add_row(pid: int, name: str):
        panel[pid] = {"playerId": pid, "name": name, "team": pid_to_abbr.get(pid),
                      "toi_seconds": int(toi.get(pid,0)),
                      "sf":0,"sa":0,"gf":0,"ga":0,
                      "sf_5v5":0.0,"sa_5v5":0.0,"gf_5v5":0.0,"ga_5v5":0.0,
                      "sf_pp":0.0,"sa_pk":0.0,"gf_pp":0.0,"ga_pk":0.0,
                      "sf_pk":0.0,"sa_pp":0.0,"gf_pk":0.0,"ga_pk":0.0,
                      "lead_sec":0,"trail_sec":0,"tied_sec":0,
                      "tsf_sum":0,"tsf_bucket_0_5":0,"tsf_bucket_6_20":0,"tsf_bucket_21_60":0,"tsf_bucket_61p":0}
    pgs = (box.get("playerByGameStats") or {})
    for side in ("homeTeam","awayTeam"):
        stats = (pgs.get(side) or {})
        for group in ("forwards","defense","goalies"):
            for p in (stats.get(group) or []):
                pid = p.get("playerId")
                name_d = p.get("name", {})
                name = name_d.get("default") if isinstance(name_d, dict) else ""
                if isinstance(pid, int):
                    pid_to_name[pid] = name
                    add_row(pid, name)

    # Per-second accumulations
    for s in range(max_sec + 1):
        on = onice[s]
        hs, as_ = score_by_sec[s]
        for pid in on:
            row = panel.get(pid)
            if not row: continue
            tid = pid_to_tid.get(pid)
            diff = (hs - as_) if tid == home_id else (as_ - hs)
            if diff > 0: row["lead_sec"] += 1
            elif diff < 0: row["trail_sec"] += 1
            else: row["tied_sec"] += 1
            tsf = tsf_at_sec[s]
            row["tsf_sum"] += tsf
            if tsf <= 5: row["tsf_bucket_0_5"] += 1
            elif tsf <= 20: row["tsf_bucket_6_20"] += 1
            elif tsf <= 60: row["tsf_bucket_21_60"] += 1
            else: row["tsf_bucket_61p"] += 1

    # Goalies per second for EN + faced goalie
    goalies_by_team = [dict() for _ in range(max_sec + 1)]
    goalie_id_by_sec_team = [dict() for _ in range(max_sec + 1)]
    for s in range(max_sec + 1):
        counts = defaultdict(int); ids = defaultdict(list)
        for pid in onice[s]:
            if (pid_to_pos.get(pid) or "").upper() == "G":
                tid = pid_to_tid.get(pid)
                if isinstance(tid, int):
                    counts[tid] += 1; ids[tid].append(pid)
        goalies_by_team[s] = dict(counts)
        goalie_id_by_sec_team[s] = {t: lst[0] for t, lst in ids.items() if lst}

    # Attribute shots
    prev_shot_by_team: Dict[int, Dict[str, float]] = {}
    for sh in shots:
        s = int(sh.get("sec") or 0)
        so = int(sh.get("sortOrder", 0))
        if s > max_sec: continue
        shooter = sh.get("shooterId")
        stid = pid_to_tid.get(shooter)
        sh_team_abbr = pid_to_abbr.get(shooter)
        sh["shooterTeamId"] = stid
        sh["shooterTeamAbbr"] = sh_team_abbr
        sh["team"] = sh_team_abbr
        if isinstance(stid, int) and home_abbr and away_abbr:
            sh["opponent"] = away_abbr if stid == home_id else home_abbr
        else:
            faced_pid = sh.get("goalieId")
            opp_ab = pid_to_abbr.get(faced_pid)
            sh["opponent"] = opp_ab
        # Merge goalie SV% for team/opponent (prefer pregame over in-game)
        _rec_team = goalie_sv_by_game_team.get((game_pk, sh_team_abbr)) if sh_team_abbr else None
        _rec_opp  = goalie_sv_by_game_team.get((game_pk, sh.get("opponent"))) if sh.get("opponent") else None
        _pick_sv = (lambda rec: (rec.get("sv_pregame_all") if rec and rec.get("sv_pregame_all") is not None else (rec.get("sv_game") if rec else None)))
        sh["team_sv"] = _pick_sv(_rec_team)
        sh["team_sv_game"] = (_rec_team.get("sv_game") if _rec_team else None)
        sh["team_sv_pregame_all"] = (_rec_team.get("sv_pregame_all") if _rec_team else None)
        sh["opp_sv"] = _pick_sv(_rec_opp)
        sh["opp_sv_game"] = (_rec_opp.get("sv_game") if _rec_opp else None)
        sh["opp_sv_pregame_all"] = (_rec_opp.get("sv_pregame_all") if _rec_opp else None)
        
        is_goal = 1 if sh.get("isGoal") else 0
        # Skip attempts without coordinates (feeds can omit coords on some MISS)
        if sh.get("x") is None or sh.get("y") is None:
            continue
        xf, yf, dist, ang_abs, ang_signed = geom(sh.get("x"), sh.get("y"))
        sh["x_flipped"], sh["y_flipped"], sh["shot_distance"], sh["shot_angle_deg"], sh["shot_angle_signed"] = xf, yf, dist, ang_abs, ang_signed
        sh["shot_angle_signed_deg"] = ang_signed
        # Absolute angle feature (clamped to [0,90])
        try:
            abs_angle = abs(float(sh["shot_angle_signed"]))
        except Exception:
            abs_angle = None
        if abs_angle is not None:
            if abs_angle < 0.0: abs_angle = 0.0
            if abs_angle > 90.0: abs_angle = 90.0
        sh["abs_angle"] = abs_angle
        sh["lateral_abs_ft"] = abs(yf)
        sh["shot_distance_log"] = math.log(max(dist, 1e-6))
        sh["is_slot"] = 1 if (xf >= 60 and xf <= 89 and abs(yf) <= 22) else 0
        sh["is_inner_slot"] = 1 if (xf >= 70 and abs(yf) <= 15) else 0

        # --- Manpower (prefer shift-based counts, fall back to penalty timeline)
        sp = (s-1) if s > 0 else 0
        # Empty-net flag and faced goalie from *actual* on-ice goalies (prev-second aware)
        def_tid = home_id if stid == away_id else (away_id if stid is not None else None)
        opp_goalies_curr = goalies_by_team[s].get(def_tid, 0) if def_tid is not None else 0
        our_goalies_curr = goalies_by_team[s].get(stid, 0) if stid is not None else 0
        opp_goalies_prev = goalies_by_team[sp].get(def_tid, 0) if def_tid is not None else 0
        our_goalies_prev = goalies_by_team[sp].get(stid, 0) if stid is not None else 0
        # Net-state flags from previous second
        opp_net_empty_prev = 1 if opp_goalies_prev == 0 else 0   # defending net empty
        our_net_empty_prev = 1 if our_goalies_prev == 0 else 0   # we pulled our goalie
        # EN event only when exactly one side has no goalie at t-1
        is_en_event = 1 if ((opp_goalies_prev == 0) ^ (our_goalies_prev == 0)) else 0
        # Also force previous-second for any OT goal (period 4)
        period_num = (s // SECONDS_PER_PERIOD) + 1
        force_prev_for_ot_goal = 1 if (is_goal and period_num == 4) else 0
        force_prev_for_any_goal = 1 if is_goal else 0
        # Pre-penalty detection: if a penalty occurs later within the same second,
        # use pre-penalty manpower (previous second) for strength attribution.
        pre_penalty = False
        try:
            min_pen_so = penalty_min_so_by_sec.get(s)
            if (min_pen_so is not None) and (so < int(min_pen_so)):
                pre_penalty = True
        except Exception:
            pre_penalty = False

        chosen_sec = sp if (is_en_event or force_prev_for_ot_goal or force_prev_for_any_goal or pre_penalty) else s
        try:
            h_on, a_on = skaters_from_onice[chosen_sec]
            source = "onice_prev" if chosen_sec == sp else "onice"
        except Exception:
            h_on, a_on = skaters_pen[chosen_sec]
            # record whether the fallback came from pre-penalty adjustment
            if chosen_sec == sp and pre_penalty:
                source = "penalty_pre"
            else:
                source = "penalty_prev" if chosen_sec == sp else "penalty_fallback"
        if (h_on, a_on) == (0,0):  # extreme guardrail
            h_on, a_on = skaters_pen[chosen_sec]
            source = "penalty_prev" if chosen_sec == sp else "penalty_fallback"

        if stid == home_id:
            us, them = h_on, a_on
            score_diff_team = (score_by_sec[s][0] - score_by_sec[s][1])
        else:
            us, them = a_on, h_on
            score_diff_team = (score_by_sec[s][1] - score_by_sec[s][0])

        # Strength bucket: prioritize true EN from goalie-out side at t-1; otherwise pair mapping (including EA)
        if is_en_event:
            if opp_goalies_prev == 0 and our_goalies_prev > 0:
                bucket = "EN_for"      # we have goalie, they don't
            elif our_goalies_prev == 0 and opp_goalies_prev > 0:
                bucket = "EN_against"  # they have goalie, we don't
            else:
                pair = (int(us), int(them))
                if pair == (5,5):                      bucket = "5v5"
                elif pair in {(5,4),(5,3),(4,3)}:      bucket = "PP"
                elif pair in {(4,5),(3,5),(3,4)}:      bucket = "PK"
                else:                                  bucket = f"{pair[0]}v{pair[1]}"
        else:
            pair = (int(us), int(them))
            if pair == (5,5):                          bucket = "5v5"
            elif pair in {(5,4),(5,3),(4,3)}:          bucket = "PP"
            elif pair in {(4,5),(3,5),(3,4)}:          bucket = "PK"
            elif pair in {(6,5),(6,4)}:                bucket = "EA_for"
            elif pair in {(5,6),(4,6)}:                bucket = "EA_against"
            else:                                      bucket = f"{pair[0]}v{pair[1]}"
        sh["strength"] = bucket
        sh["us_skaters"], sh["them_skaters"] = us, them
        sh["manpower_diff"] = int(us) - int(them)
        sh["manpower_source"] = source

        # Empty-net flags (goal flag derived from defending goalie at t-1)
        sh["is_empty_net_event"] = is_en_event
        sh["is_extra_attacker_event"] = our_net_empty_prev
        try:
            sh["faced_goalie_id"] = goalie_id_by_sec_team[chosen_sec].get(def_tid)
        except Exception:
            sh["faced_goalie_id"] = None
        if is_goal:
            sh["is_empty_net_goal"] = 1 if (opp_goalies_prev == 0) else 0
        else:
            sh["is_empty_net_goal"] = 0

        # Override for decisive/end-of-period goals:
        # - OT goals (period 4)
        # - Any goal that is the last GOAL of the period
        is_last_goal_period = 1 if (last_goal_by_period.get(period_num) == (s, so)) else 0
        if is_goal and (period_num == 4 or is_last_goal_period == 1):
            try:
                prev_team_shot = prev_shot_by_team.get(stid)
                prev_face_gid = prev_team_shot.get("faced_goalie_id") if prev_team_shot else None
                if isinstance(prev_face_gid, int):
                    sh["faced_goalie_id"] = prev_face_gid
                else:
                    # Fallback 1: find the previous attempt from raw plays and use its faced goalie
                    prev_list = attempts_by_team.get(stid, [])
                    # binary search for last (sec,so) < (s,so)
                    idx = -1
                    lo, hi = 0, len(prev_list)-1
                    while lo <= hi:
                        mid = (lo+hi)//2
                        if prev_list[mid][0] < s or (prev_list[mid][0] == s and prev_list[mid][1] < so):
                            idx = mid; lo = mid+1
                        else:
                            hi = mid-1
                    fallback_gid = None
                    if idx >= 0:
                        ps, pso = prev_list[idx]
                        ps_prev = ps-1 if ps>0 else 0
                        try:
                            fallback_gid = goalie_id_by_sec_team[ps_prev].get(def_tid)
                        except Exception:
                            fallback_gid = None
                        if fallback_gid is None:
                            try:
                                fallback_gid = goalie_id_by_sec_team[ps].get(def_tid)
                            except Exception:
                                fallback_gid = None
                    # Fallback 2: defending goalie at previous second (or current)
                    if fallback_gid is None:
                        try:
                            fallback_gid = goalie_id_by_sec_team[sp].get(def_tid)
                        except Exception:
                            fallback_gid = None
                    if fallback_gid is None:
                        try:
                            fallback_gid = goalie_id_by_sec_team[s].get(def_tid)
                        except Exception:
                            fallback_gid = None
                    if isinstance(fallback_gid, int):
                        sh["faced_goalie_id"] = fallback_gid
            except Exception:
                pass

        # Per-shot opponent SV from faced goalie, if available (prefer pregame over in-game)
        faced_gid = sh.get("faced_goalie_id")
        # shots_train goalie_id should mirror the raw feed's goalieId exactly (not faced goalie)
        g_tmp = sh.get("goalieId")
        sh["goalie_id"] = int(g_tmp) if isinstance(g_tmp, int) else None
        if isinstance(faced_gid, int):
            rec_face = goalie_sv_by_game_goalie.get((game_pk, faced_gid))
            if rec_face:
                sv_pref = rec_face.get("sv_pregame_all") if rec_face.get("sv_pregame_all") is not None else rec_face.get("sv_game")
                sh["opp_sv"] = sv_pref
                sh["opp_sv_game"] = rec_face.get("sv_game")
                sh["opp_sv_pregame_all"] = rec_face.get("sv_pregame_all")

        sh["tsf"] = tsf_for_play(s, so, faceoff_index)
        sh["score_diff_team"] = score_diff_team
        sh["score_state"] = ("lead" if score_diff_team>0 else ("trail" if score_diff_team<0 else "tied"))

        # Rush / transition metrics
        sh["rush_trigger"] = ""
        sh["trans_dx"] = 0.0
        sh["trans_dt"] = 0
        sh["trans_vx"] = 0.0
        is_rush = 1 if sh["tsf"] <= TSF_RUSH else 0
        rush_trigger_label = ""
        if not is_rush and sh["strength"] == "5v5" and sh["tsf"] <= TSF_BURST and isinstance(stid, int):
            trig_list = triggers_by_team.get(stid, [])
            for k in range(len(trig_list)-1, -1, -1):
                te = trig_list[k]
                dt = s - int(te.get("sec", -10))
                if dt < 0:
                    continue
                if dt > TRANS_DT:
                    break
                dx = (sh.get("x") or 0) - (te.get("x") or 0)
                vx = (dx / dt) if dt > 0 else (dx if dx > 0 else 0.0)
                if dx >= MIN_DX and (dt == 0 or vx >= MIN_VX):
                    is_rush = 1
                    rush_trigger_label = str(te.get("label") or "")
                    sh["rush_trigger"] = rush_trigger_label
                    sh["trans_dx"], sh["trans_dt"], sh["trans_vx"] = float(dx), int(dt), float(vx)
                    break
        sh["is_rush"] = is_rush
        if rush_trigger_label:
            sh["rush_trigger"] = rush_trigger_label
 
        prev_sh = prev_shot_by_team.get(stid)
        # --- Attack-normalized rebound geometry (Option B)
        # Use flipped coords so net is always to +x; compare to previous UNBLOCKED attempt by same team.
        is_rebound = 0
        dt2 = 0.0
        dx2 = 0.0
        dy2 = 0.0

        if prev_sh:
            dt_raw = float(s - int(prev_sh.get("sec", 0)))
            dx_att = float(xf - float(prev_sh.get("xf", 0.0)))
            dy_att = float(yf - float(prev_sh.get("yf", 0.0)))

            # 0 < dt <= 4.0 is a very standard rebound window
            if (dt_raw > 0.0) and (dt_raw <= 4.0) and (abs(dx_att) <= 20.0) and (abs(dy_att) <= 20.0):
                is_rebound = 1
                dt2, dx2, dy2 = dt_raw, dx_att, dy_att

        sh["is_rebound"] = int(is_rebound)
        sh["rebound_dt"] = float(dt2) if is_rebound else 0.0
        sh["rebound_dx"] = float(dx2) if is_rebound else 0.0
        sh["rebound_dy"] = float(dy2) if is_rebound else 0.0
        # Also provide magnitudes (optional but useful)
        sh["rebound_dx_abs"] = abs(sh["rebound_dx"])
        sh["rebound_dy_abs"] = abs(sh["rebound_dy"])

        # Track previous attempt in the *flipped* frame for consistency
        prev_shot_by_team[stid] = {
            "sec": s,
            "xf": float(xf),
            "yf": float(yf),
            "faced_goalie_id": sh.get("faced_goalie_id"),
        }

        prev_same = prev_same_team.get((s, so))
        sh["prev_event_type_same"] = (prev_same.get("type") if prev_same else None)
        sh["delta_t_same_s"] = None
        sh["delta_x_same_ft"] = None
        sh["delta_y_same_ft"] = None
        if prev_same:
            dt = s - int(prev_same["sec"])
            if dt > 0:
                px = float(prev_same["x"]) ; py = float(prev_same["y"])
                dx_raw = (sh.get("x") or 0.0) - px
                dy_raw = (sh.get("y") or 0.0) - py
                sh["delta_t_same_s"] = dt
                sh["delta_x_same_ft"] = dx_raw
                sh["delta_y_same_ft"] = dy_raw
                px_f, _ = flip_xy_to_attacking(px, py)
                dx_att = sh["x_flipped"] - px_f

        pg_any = prev_global_any.get((s, so))
        sh["prev_event_type_global"] = (pg_any.get("type") if pg_any else None)
        sh["prev_team_global_abbr"]   = (pid_to_abbr.get(pg_any["team"]) if pg_any and pg_any.get("team") else None)
        if pg_any:
            sh["delta_t_global_s"] = s - int(pg_any["sec"])
            sh["time_since_prev_event_s"] = sh["delta_t_global_s"]
        else:
            sh["delta_t_global_s"] = None
            sh["time_since_prev_event_s"] = None

        pg_xy = prev_global_xy.get((s, so))
        if pg_xy:
            dtg = s - int(pg_xy["sec"])
            if dtg > 0:
                px, py = float(pg_xy["x"]), float(pg_xy["y"])
                dx_raw = (sh.get("x") or 0.0) - px
                dy_raw = (sh.get("y") or 0.0) - py
                sh["delta_x_from_prev_global_ft"] = dx_raw
                sh["delta_y_from_prev_global_ft"] = dy_raw
                px_f, _ = flip_xy_to_attacking(px, py)
                dx_att = sh["x_flipped"] - px_f

        # Period timing: REG OT (period 4) is 5 minutes; others use standard 20 minutes
        period_num_local = (s // SECONDS_PER_PERIOD) + 1
        per_start = (period_num_local - 1) * SECONDS_PER_PERIOD
        sec_in_period_local = s - per_start
        local_len = 300 if (period_num_local == 4 and game_type_code == "REG") else SECONDS_PER_PERIOD
        sh["sec_in_period"] = sec_in_period_local
        sh["sec_game"] = s
        sh["time_remaining_period_s"] = max(0, local_len - sec_in_period_local)

        # add calibrated xG if available
        if xg_idx:
            try:
                key = (int(game_pk), int(period_num_local), int(sh["sec_game"]), int(sh.get("sortOrder", 0)))
                if key in xg_idx:
                    sh["xg"] = xg_idx[key]
            except Exception:
                pass

        # --- EN-only xG: compute ONLY for EN_for attempts; leave everything else untouched
        try:
            strength_bucket = (sh.get("strength") or "").lower()
        except Exception:
            strength_bucket = ""
        if strength_bucket == "en_for":
            dist_ft = float(sh.get("shot_distance", 0.0) or 0.0)
            try:
                abs_deg = float(sh.get("abs_angle")) if sh.get("abs_angle") is not None else None
            except Exception:
                abs_deg = None
            sh["xg_en"] = en_xg(dist_ft, abs_deg)

        for k in ("us_skaters","them_skaters","manpower_diff","sec_in_period","time_remaining_period_s","abs_angle","trans_dt","trans_dx","trans_vx","is_rush","rush_trigger"):
            if sh.get(k) is None: sh[k] = ""

        # on-ice attribution of For/Against counts by strength bucket
        for pid in onice[s]:
            row = panel.get(pid)
            if not row: continue
            ptid = pid_to_tid.get(pid)
            if ptid is None or stid is None: continue
            is_for = (ptid == stid)
            row["sf" if is_for else "sa"] += 1
            if is_goal: row["gf" if is_for else "ga"] += 1
            if bucket == "5v5":
                if is_for: row["sf_5v5"] += 1; row["gf_5v5"] += is_goal
                else:      row["sa_5v5"] += 1; row["ga_5v5"] += is_goal
            elif bucket == "PP":
                if is_for: row["sf_pp"] += 1; row["gf_pp"] += is_goal
                else:      row["sa_pk"] += 1; row["ga_pk"] += is_goal
            elif bucket == "PK":
                if is_for: row["sf_pk"] += 1; row["gf_pk"] += is_goal
                else:      row["sa_pp"] += 1; row["ga_pk"] += is_goal
            elif bucket == "EA_for":
                if is_for: row["sf_5v5"] += 1; row["gf_5v5"] += is_goal
                else:      row["sa_5v5"] += 1; row["ga_5v5"] += is_goal
            elif bucket == "EA_against":
                if is_for: row["sf_5v5"] += 1; row["gf_5v5"] += is_goal
                else:      row["sa_5v5"] += 1; row["ga_5v5"] += is_goal
            elif bucket == "EN_for":
                if is_for: row["sf_5v5"] += 1; row["gf_5v5"] += is_goal
                else:      row["sa_5v5"] += 1; row["ga_5v5"] += is_goal
            elif bucket == "EN_against":
                if is_for: row["sf_5v5"] += 1; row["gf_5v5"] += is_goal
                else:      row["sa_5v5"] += 1; row["ga_5v5"] += is_goal

    def per60(n, sec): return 0.0 if sec <= 0 else 3600.0 * float(n) / float(sec)
    for r in panel.values():
        toi_all = max(1, int(r.get("toi_seconds", 0)))
        r["GF60_5v5"] = per60(r.get("gf_5v5",0.0), toi_all)
        r["GA60_5v5"] = per60(r.get("ga_5v5",0.0), toi_all)
        r["SF60_5v5"] = per60(r.get("sf_5v5",0.0), toi_all)
        r["SA60_5v5"] = per60(r.get("sa_5v5",0.0), toi_all)
        r["tsf_avg_s"]       = float(r.get("tsf_sum",0))/float(toi_all)
        r["tsf_share_0_5"]   = float(r.get("tsf_bucket_0_5",0))/float(toi_all)
        r["tsf_share_6_20"]  = float(r.get("tsf_bucket_6_20",0))/float(toi_all)
        r["tsf_share_21_60"] = float(r.get("tsf_bucket_21_60",0))/float(toi_all)
        r["tsf_share_61p"]   = float(r.get("tsf_bucket_61p",0))/float(toi_all)

    def _abbr_to_ids(abbr: Optional[str]) -> Optional[int]:
        if not abbr:
            return None
        if abbr == home_abbr:
            return home_id
        if abbr == away_abbr:
            return away_id
        return None
    home_meta = teams_meta.get(str(home_id), {}) if isinstance(teams_meta, dict) else {}
    away_meta = teams_meta.get(str(away_id), {}) if isinstance(teams_meta, dict) else {}
    for r in panel.values():
        team_ab = r.get("team")
        tid = _abbr_to_ids(team_ab)
        if tid == home_id:
            my_meta, opp_meta = home_meta, away_meta
        elif tid == away_id:
            my_meta, opp_meta = away_meta, home_meta
        else:
            my_meta, opp_meta = {}, {}
        same_conf = 1 if (my_meta.get("conference") and my_meta.get("conference") == opp_meta.get("conference")) else 0
        same_div  = 1 if (my_meta.get("division") and my_meta.get("division") == opp_meta.get("division")) else 0
        r["is_conference"] = same_conf
        r["is_divisional"] = same_div
        r["is_playoff"] = game_is_playoff
        my_rank_pre = standings_rank_pre.get(tid) if standings_rank_pre else None
        opp_rank_pre = standings_rank_pre.get(away_id if tid == home_id else home_id) if standings_rank_pre else None
        r["team_rank_pre"] = my_rank_pre
        r["opp_rank_pre"] = opp_rank_pre
        # Merge goalie SV% for player team/opponent (prefer pregame over in-game)
        opp_ab = away_abbr if team_ab == home_abbr else (home_abbr if team_ab == away_abbr else None)
        rec_team = goalie_sv_by_game_team.get((game_pk, team_ab)) if team_ab else None
        rec_opp  = goalie_sv_by_game_team.get((game_pk, opp_ab)) if opp_ab else None
        def _pick_sv_for_merge(rec):
            return rec.get("sv_pregame_all") if rec and rec.get("sv_pregame_all") is not None else (rec.get("sv_game") if rec else None)
        r["team_sv"] = _pick_sv_for_merge(rec_team)
        r["team_sv_game"] = rec_team.get("sv_game") if rec_team else None
        r["team_sv_pregame_all"] = rec_team.get("sv_pregame_all") if rec_team else None
        r["opp_sv"] = _pick_sv_for_merge(rec_opp)
        r["opp_sv_game"] = rec_opp.get("sv_game") if rec_opp else None
        r["opp_sv_pregame_all"] = rec_opp.get("sv_pregame_all") if rec_opp else None

    shots_out = shots
    panel_out_rows = list(panel.values())
    if only_team_abbr:
        panel_out_rows = [r for r in panel_out_rows if (r.get("team") == only_team_abbr)]
        shots_out = [s for s in shots if s.get("team") == only_team_abbr]

    # Training filter: non-EN, unblocked attempts (goal + SOG + MISS)
    def is_train_attempt(sh: Dict[str, Any]) -> bool:
        # Exclude shootouts (period 5) and EN_for; allow EN_against and EA
        try:
            if int(sh.get("period", 0)) == 5:
                return False
        except Exception:
            pass
        # Restrict to regulation periods 1-3
        try:
            p = int(sh.get("period", 0))
            if p < 1 or p > 3:
                return False
        except Exception:
            return False
        strength_bucket = (sh.get("strength") or "")
        # Keep only 5v5 and one-man powerplays (for regularization)
        if strength_bucket not in ("5v5", "PP", "PK") and strength_bucket not in ("5v5", "5v4", "4v5"):
            # If bucket format is normalized (5v4/4v5) accept; else PP/PK
            return False
        if int(sh.get("is_unblocked", 1)) != 1:
            return False
        et = (sh.get("eventType") or "").lower()
        return et in ("goal", "shot-on-goal", "missed-shot")

    shots_train = [s for s in shots_out if is_train_attempt(s)]
    shots_train_path = os.path.join(out_dir, f"shots_train_{game_pk}.csv")
    shots_train_fields = [
        "gamePk","period","sec_game","sortOrder",
        "team","opponent","shooterTeamAbbr","shooterTeamId","shooterId","strength","goalie_id",
        "is_unblocked","is_empty_net_event","isGoal","eventType",
        "x_flipped","y_flipped",
        "shot_distance_log","shot_angle_signed_deg","abs_angle","lateral_abs_ft","is_slot","is_inner_slot","shotType",
        "us_skaters","them_skaters","manpower_diff",
        "score_diff_team","sec_in_period","time_remaining_period_s","tsf",
        "is_rebound","rebound_dt","rebound_dx","rebound_dy","rebound_dx_abs","rebound_dy_abs","xg",
    ]
    write_csv_with_fields(shots_train, shots_train_path, shots_train_fields)
    print(f"Wrote {shots_train_path} ({len(shots_train)} rows)")
    if shots_train_only:
        return

    shots_path = os.path.join(out_dir, f"shots_{game_pk}.csv")
    panel_path = os.path.join(out_dir, f"panel_{game_pk}.csv")
    panel_train_path = os.path.join(out_dir, f"panel_train_{game_pk}.csv")
    write_csv(shots_out, shots_path)
    write_csv(panel_out_rows, panel_path)
    write_csv(panel_out_rows, panel_train_path)
    print(f"Wrote {shots_path} ({len(shots_out)} rows)")
    print(f"Wrote {panel_path} ({len(panel_out_rows)} rows)")

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Build shots/panel CSVs (api-web raw layout) with engineered features + team/opponent SV% + REG-season ranks (by-date snapshots).")
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--raw", type=str, default="artifacts/dumps/raw")
    ap.add_argument("--out", type=str, default="artifacts/training")
    ap.add_argument("--only-team-abbr", type=str, default=None,
                    help="If set (e.g., TOR), final CSVs are filtered to that team's rows/shots only.")
    ap.add_argument("--debug-goalies", action="store_true", help="Print debug info about goalie SV file scanning, candidates and chosen rows.")
    ap.add_argument("--debug-strength", action="store_true", help="Print detailed reasoning for strength bucket selection per shot.")
    ap.add_argument("--shots-train-only", action="store_true", help="Only write shots_train_<game>.csv and skip other outputs.")
    ap.add_argument("--xg-jsonl", type=str, default=None, help="Path to master JSONL (shots_xg.jsonl) to merge xG per shot.")
    args = ap.parse_args()
    build(args.game, args.raw, args.out, args.only_team_abbr, debug_goalies=args.debug_goalies, debug_strength=args.debug_strength, shots_train_only=args.shots_train_only, xg_jsonl=args.xg_jsonl)

if __name__ == "__main__":
    main()
