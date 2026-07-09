#!/usr/bin/env python3
"""
Team + Player-aware training windows (JSON-in -> windows.json, team_windows.json, player_windows.json/CSV)
[... docstring unchanged for brevity ...]
"""

from __future__ import annotations
import argparse, json, os, math, re, glob
from collections import defaultdict, Counter
from typing import Any, Dict, List, Tuple, Optional

# Globals set by CLI (main) so helper functions can access input/standings dirs
CLI_IN_PATH: Optional[str] = None
CLI_STANDINGS_DIR: Optional[str] = None

# -------------------- Constants --------------------
SECONDS_PER_PERIOD = 20 * 60
MAX_SECONDS        = SECONDS_PER_PERIOD * 6  # safety ceiling

NATURAL_BREAK_TYPES = {
    "faceoff","goal","penalty","offside","icing","stoppage",
    "puck-out-of-play","goalie-stopped","timeout","challenge"
}

# chemistry thresholds (adaptive)
SEC_FLOOR_BASE   = 5
RAW_SHARE_MIN    = 0.0225
SHORT_WIN_SEC    = 10
SHORT_SHARE_MIN  = 0.40
SHORT_SEC_FRAC   = 0.50
TOPK_FALLBACK    = 1

EA_PRUNE_MIN     = 6

# Edge cameo drop rule
EDGE_CAMEO_SEC          = 5
DROP_PLAYER_EDGE_ROWS   = True

# What counts as a "useful event" for protecting a ≤3s edge cameo?
# Define granular classes for attempts and on-target shots
SHOT_ON_GOAL_TYPES = {"shot", "shot-on-goal"}  # saved shots on goal (feed may use just "shot")
MISS_TYPES         = {"missed-shot"}
BLOCK_TYPES        = {"blocked-shot"}
GOAL_TYPES         = {"goal"}
SHOT_TYPES         = SHOT_ON_GOAL_TYPES | MISS_TYPES | BLOCK_TYPES  # all attempts
MICRO_PROTECT_TYPES = {"hit","takeaway","giveaway","faceoff"}

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def clock_str(abs_sec: int) -> str:
    p = abs_sec % SECONDS_PER_PERIOD
    return f"{p//60}:{p%60:02d}"

def period_of(abs_sec: int) -> int:
    return (abs_sec // SECONDS_PER_PERIOD) + 1

def is_long_change(start_sec: int) -> bool:
    """True in periods 2 and 4 (long change bench distance)."""
    p = period_of(start_sec)
    return p in (2, 4)

# Global strength label (uses CURRENT second goalies)
def strength_global(
    home_skaters: int, away_skaters: int,
    home_goalie_now: int, away_goalie_now: int
) -> str:
    if home_goalie_now == 0 and away_goalie_now == 1 and home_skaters >= away_skaters:
        return "EA_home"
    if away_goalie_now == 0 and home_goalie_now == 1 and away_skaters >= home_skaters:
        return "EA_away"
    pair = (home_skaters, away_skaters)
    if pair == (5, 5): return "5v5"
    if pair in {(5,4),(5,3),(4,3)}: return "PP_home"
    if pair in {(4,5),(3,5),(3,4)}: return "PP_away"
    if pair == (4, 4): return "4v4"
    return f"{home_skaters}v{away_skaters}"

# Per-team label from counts + current goalie presence
def strength_for_team(
    skaters_for: int, skaters_against: int,
    goalie_for: int, goalie_against: int
) -> str:
    if goalie_for == 0 and goalie_against == 1 and skaters_for >= skaters_against:
        return "EA"
    if goalie_against == 0 and goalie_for == 1 and skaters_against >= skaters_for:
        return "EN_for"
    if skaters_for == 5 and skaters_against == 5: return "5v5"
    if (skaters_for, skaters_against) in {(5,4),(5,3),(4,3)}: return "PP"
    if (skaters_for, skaters_against) in {(4,5),(3,5),(3,4)}: return "PK"
    if skaters_for == 4 and skaters_against == 4: return "4v4"
    return f"{skaters_for}v{skaters_against}"

# --- Role helpers for matchup-quality (C/L/R -> F; D stays D) ---
F_CODES = {"F","C","L","R","LW","RW"}
D_CODES = {"D","LD","RD"}

def role_of(pos_code: Optional[str]) -> str:
    c = (pos_code or "").upper()
    if c in D_CODES or c == "D":
        return "D"
    if c in F_CODES:
        return "F"
    return "F"

def is_credit_event(t: str) -> bool:
    t = (t or "").lower().replace("_","-")
    # attempts and goals are window-credit-relevant
    return t in SHOT_TYPES or t in GOAL_TYPES

# -------------------- Helpers --------------------
def _safe_abs_sec(e: Dict[str, Any]) -> int:
    try:
        s = int(e.get("sec_game", -1))
    except Exception:
        s = -1
    try:
        p = int(e.get("period", 0) or 0)
    except Exception:
        p = 0
    tip = str(e.get("timeInPeriod") or "").strip()
    mm, ss = 0, 0
    if tip and ":" in tip:
        try:
            mm, ss = (int(x) for x in tip.split(":"))
        except Exception:
            mm, ss = 0, 0
    expected = (p - 1) * SECONDS_PER_PERIOD + mm * 60 + ss if p >= 1 else max(0, s)
    if p >= 1:
        lo = (p - 1) * SECONDS_PER_PERIOD
        hi = p * SECONDS_PER_PERIOD
        if not (lo <= s < hi):
            return expected
    return s if s >= 0 else expected

def _check_consistency(evts: List[Dict[str, Any]]) -> None:
    bad = 0
    for e in evts:
        try:
            s = int(e.get("sec_game")) if e.get("sec_game") is not None else None
        except Exception:
            s = None
        s2 = _safe_abs_sec(e)
        if s is not None and isinstance(s, int) and s != s2:
            bad += 1
    if bad:
        print(f"[warn] corrected {bad} events with inconsistent sec_game vs period/timeInPeriod.")

# -------------------- Per-second on-ice reconstruction --------------------
def build_second_by_second_onice(evts: List[Dict[str,Any]]) -> Tuple[
    List[Tuple[Tuple[int,...],Tuple[int,...]]],
    List[Dict[str,int]]
]:
    if not evts:
        return [], []
    horizon = 0
    for e in evts:
        horizon = max(horizon, _safe_abs_sec(e))
    horizon = min(horizon + 1, MAX_SECONDS)

    first = evts[0]
    home = set(int(pid) for pid in (first.get("onice", {}).get("home") or []))
    away = set(int(pid) for pid in (first.get("onice", {}).get("away") or []))
    g_home = 1 if (first.get("onice", {}).get("goalies", {}).get("home")) else 0
    g_away = 1 if (first.get("onice", {}).get("goalies", {}).get("away")) else 0

    team_onice_by_sec: List[Tuple[Tuple[int,...],Tuple[int,...]]] = []
    goalies_by_team:   List[Dict[str,int]] = []

    idx_by_time: Dict[int, List[Dict[str,Any]]] = defaultdict(list)
    for e in evts:
        idx_by_time[_safe_abs_sec(e)].append(e)

    for s in range(horizon):
        if s in idx_by_time:
            for e in sorted(idx_by_time[s], key=lambda x: int(x.get("sortOrder", 0))):
                oi = e.get("onice") or {}
                sc = e.get("shift_change") or {}

                if sc:
                    for pid in sc.get("home_in") or []:  home.add(int(pid))
                    for pid in sc.get("home_out") or []: home.discard(int(pid))
                    for pid in sc.get("away_in") or []:  away.add(int(pid))
                    for pid in sc.get("away_out") or []: away.discard(int(pid))
                elif oi:
                    h = oi.get("home"); a = oi.get("away")
                    if isinstance(h, list) and isinstance(a, list):
                        home = set(int(p) for p in h)
                        away = set(int(p) for p in a)

                g = oi.get("goalies") or {}
                if "home" in g: g_home = 1 if g.get("home") else 0
                if "away" in g: g_away = 1 if g.get("away") else 0

        team_onice_by_sec.append((tuple(sorted(home)), tuple(sorted(away))))
        goalies_by_team.append({"home": g_home, "away": g_away})

    return team_onice_by_sec, goalies_by_team

# -------------------- Window builder --------------------
def build_windows(
    evts: List[Dict[str,Any]],
    hard_cap_sec: int = 0,
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    player_name_map: Optional[Dict[int, str]] = None,
    player_pos_map: Optional[Dict[int, str]] = None,
    debug_ga: bool = False,
    debug_standings: bool = False,
    game_pk: Optional[int] = None,
) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]], List[Dict[str,Any]]]:
    evts_sorted = sorted(evts, key=lambda e: (_safe_abs_sec(e), int(e.get("sortOrder",0))))
    if not evts_sorted:
        return [], [], []
    _check_consistency(evts_sorted)

    # derive game_pk from root if present (for standings matching)
    game_pk_local: Optional[int] = None
    # prefer explicit argument
    if isinstance(game_pk, int):
        game_pk_local = int(game_pk)
    else:
        try:
            root_game0 = evts_sorted[0].get("game", {})
            gpk = root_game0.get("gamePk") or root_game0.get("game_pk") or root_game0.get("id")
            game_pk_local = int(gpk) if gpk is not None else None
        except Exception:
            game_pk_local = None
    if debug_standings:
        print({"debug":"standings","init": True, "events": len(evts_sorted), "game_pk": game_pk_local})

    # pull optional team ids from root events block if present
    try:
        root = evts_sorted[0].get("game", {})
        home_team_id = home_team_id or root.get("home_team_id")
        away_team_id = away_team_id or root.get("away_team_id")
    except Exception:
        pass

    # per-second on-ice (skaters + goalie presence flags)
    team_onice_by_sec, goalies_by_team = build_second_by_second_onice(evts_sorted)
    horizon = len(team_onice_by_sec) - 1

    # Build per-second goalie IDs by side using on-ice snapshots in events (carry forward latest)
    snaps_goalie_ids: Dict[int, Dict[str,int]] = {}
    for e in evts_sorted:
        try:
            s_abs = _safe_abs_sec(e)
            oi = e.get("onice") or {}
            g = oi.get("goalies") or {}
            if "home" in g or "away" in g:
                entry = snaps_goalie_ids.setdefault(s_abs, {"home": 0, "away": 0})
                try:
                    if "home" in g: entry["home"] = int(g.get("home") or 0)
                except Exception:
                    entry["home"] = entry.get("home", 0)
                try:
                    if "away" in g: entry["away"] = int(g.get("away") or 0)
                except Exception:
                    entry["away"] = entry.get("away", 0)
        except Exception:
            pass
    goalie_ids_by_sec: List[Dict[str,int]] = [{"home":0,"away":0} for _ in range(max(0, horizon+1))]
    last_home_id, last_away_id = 0, 0
    for s in range(max(0, horizon+1)):
        snap = snaps_goalie_ids.get(s)
        if snap is not None:
            try:
                if snap.get("home") is not None:
                    last_home_id = int(snap.get("home") or 0)
                if snap.get("away") is not None:
                    last_away_id = int(snap.get("away") or 0)
            except Exception:
                pass
        goalie_ids_by_sec[s] = {"home": last_home_id, "away": last_away_id}

    # --- Pulled-goalie precomputation ---
    pulled_home = [0]*(horizon+1)
    pulled_away = [0]*(horizon+1)
    for s in range(horizon+1):
        g = goalies_by_team[s]
        pulled_home[s] = 1 if g["home"] == 0 else 0
        pulled_away[s] = 1 if g["away"] == 0 else 0

    since_pulled_home = [0]*(horizon+1)
    since_pulled_away = [0]*(horizon+1)
    for s in range(1, horizon+1):
        since_pulled_home[s] = (since_pulled_home[s-1] + 1) if pulled_home[s] else 0
        since_pulled_away[s] = (since_pulled_away[s-1] + 1) if pulled_away[s] else 0

    # --- 5v5 usage per game (seconds) and role-separated percentiles ---
    home_5v5_sec, away_5v5_sec = defaultdict(int), defaultdict(int)

    def is_5v5_strength(s: int) -> bool:
        ss = max(0, min(s, horizon))
        h_ids, a_ids = team_onice_by_sec[ss]
        g = goalies_by_team[ss]
        return strength_global(len(h_ids), len(a_ids), g["home"], g["away"]) == "5v5"

    for s in range(max(0, horizon)):
        if not is_5v5_strength(s):
            continue
        h_ids, a_ids = team_onice_by_sec[s]
        for pid in h_ids:
            home_5v5_sec[pid] += 1
        for pid in a_ids:
            away_5v5_sec[pid] += 1

    def percentiles_by_role(seconds_by_pid: Dict[int,int]) -> Dict[int,float]:
        buckets = defaultdict(list)  # role -> [(pid, sec)]
        for pid, sec in seconds_by_pid.items():
            pos = (player_pos_map.get(pid) if isinstance(player_pos_map, dict) else None)
            buckets[role_of(pos)].append((pid, sec))
        pct: Dict[int,float] = {}
        for role, rows in buckets.items():
            if not rows:
                continue
            rows.sort(key=lambda kv: kv[1])
            n = len(rows)
            if n == 1:
                pct[rows[0][0]] = 0.5
                continue
            i = 0
            while i < n:
                j = i
                sec_i = rows[i][1]
                while j+1 < n and rows[j+1][1] == sec_i:
                    j += 1
                mid = (i + j) / 2.0
                p = mid / (n - 1) if (n - 1) > 0 else 0.5
                for k in range(i, j+1):
                    pct[rows[k][0]] = float(p)
                i = j + 1
        return pct

    home_pct_role_5v5 = percentiles_by_role(home_5v5_sec)
    away_pct_role_5v5 = percentiles_by_role(away_5v5_sec)
    DEFAULT_PCT = 0.5

    def strength_at(s: int) -> str:
        s = max(0, min(s, horizon))
        home_ids, away_ids = team_onice_by_sec[s]
        gnow = goalies_by_team[s]
        return strength_global(len(home_ids), len(away_ids), gnow["home"], gnow["away"])

    # Index event types per second (for protective micro-events + faceoff awareness)
    types_by_sec: Dict[int, set] = defaultdict(set)
    # Event orders per second so we can reason about sort order within a second
    orders_by_sec: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    # Same-second intra-order per second (if provided by feed)
    sso_by_sec: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    # Full events by second for personal counts (e.g., giveaways/takeaways)
    events_by_sec: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    # Aggregate shift-change ins/outs per second for jitter correction
    shift_changes_by_sec: Dict[int, Dict[str, set]] = defaultdict(lambda: {
        "home_in": set(), "home_out": set(), "away_in": set(), "away_out": set()
    })
    for e in evts_sorted:
        t = (e.get("type") or "").lower().replace("_","-")
        s = _safe_abs_sec(e)
        so = int(e.get("sortOrder", 0))
        types_by_sec[s].add(t)
        orders_by_sec[s].append((so, t))
        try:
            sso = int(e.get("same_sec_order", 0))
        except Exception:
            sso = 0
        sso_by_sec[s].append((sso, t))
        events_by_sec[s].append({
            "type": t,
            "details": (e.get("details") or {}),
            "sortOrder": so,
        })
        sc = e.get("shift_change") or {}
        try:
            for k in ("home_in","home_out","away_in","away_out"):
                for pid in sc.get(k) or []:
                    shift_changes_by_sec[s][k].add(int(pid))
        except Exception:
            pass

    def has_credit_before_faceoff(sec: int) -> bool:
        # Prefer same_sec_order if present
        evs_sso = sorted(sso_by_sec.get(sec, []))
        face_sso = [sso for sso, t in evs_sso if t == "faceoff"]
        credit_sso = [sso for sso, t in evs_sso if (t in SHOT_TYPES or t in GOAL_TYPES)]
        if face_sso:
            if not credit_sso:
                return False
            return min(credit_sso) < min(face_sso)
        # Fallback to sortOrder
        evs = sorted(orders_by_sec.get(sec, []))
        if not evs:
            return False
        face_orders = [so for so, t in evs if t == "faceoff"]
        credit_orders = [so for so, t in evs if (t in SHOT_TYPES or t in GOAL_TYPES)]
        if not credit_orders:
            return False
        if not face_orders:
            return True
        return min(credit_orders) < min(face_orders)

    def has_sog_before_goalie_stop(sec: int) -> bool:
        evs = sorted(orders_by_sec.get(sec, []))
        if not evs:
            return False
        sog_orders = [so for so, t in evs if t in SHOT_ON_GOAL_TYPES]
        gs_orders  = [so for so, t in evs if t == "goalie-stopped"]
        if not sog_orders or not gs_orders:
            return False
        return min(sog_orders) < min(gs_orders)

    def has_defense_save_before_whistle(sec: int) -> bool:
        # treat blocks (BA for the attacker) or takeaways as protective if before boundary whistle/faceoff
        evs = sorted(orders_by_sec.get(sec, []))
        if not evs:
            return False
        block_orders = [so for so, t in evs if t in BLOCK_TYPES]
        takeaway_orders = [so for so, t in evs if t == "takeaway"]
        boundary_types = {"faceoff","stoppage","goal","goalie-stopped","penalty","timeout","challenge"}
        boundary_orders = [so for so, t in evs if t in boundary_types]
        if not boundary_orders:
            return bool(block_orders or takeaway_orders)
        bmin = min(boundary_orders)
        return (block_orders and min(block_orders) < bmin) or (takeaway_orders and min(takeaway_orders) < bmin)

    def goal_swap_cand_out_ids(sec: int, side: str) -> set:
        """Identify any single player who appears at 'sec' for 'side' but was not present at sec-1,
        while a credited goal participant (scorer/assist) is missing from 'side' at 'sec' but present at sec-1.
        If found, return a set with that one candidate id; else empty set.
        """
        try:
            end_events = [ev for ev in events_by_sec.get(sec, []) if ev.get("type") == "goal"]
        except Exception:
            end_events = []
        if not end_events:
            return set()
        # Respect boundary ordering: require the goal to occur before any boundary at this second
        evs_end = sorted(orders_by_sec.get(sec, []))
        boundary_types = {"faceoff","stoppage","goal","goalie-stopped","penalty","timeout","challenge"}
        end_boundary_orders = [so for so, t in evs_end if t in boundary_types]
        end_boundary_order = min(end_boundary_orders) if end_boundary_orders else None
        if end_boundary_order is None:
            return set()
        # choose the first goal with sortOrder < end_boundary
        goal_ev = None
        for ev in end_events:
            try:
                so = int(ev.get("sortOrder", 0))
            except Exception:
                so = 0
            if so < end_boundary_order:
                goal_ev = ev
                break
        if goal_ev is None:
            return set()
        det = goal_ev.get("details") or {}
        cb_end = credit_by_sec.get(sec)
        if not cb_end:
            return set()
        pre_home = set(cb_end.get("onice_home") or [])
        pre_away = set(cb_end.get("onice_away") or [])
        side_set = pre_home if side == "home" else pre_away
        # participants
        parts = []
        for k in ("scoringPlayerId","assist1PlayerId","assist2PlayerId"):
            v = det.get(k)
            try:
                if v is not None:
                    parts.append(int(v))
            except Exception:
                pass
        if not parts:
            return set()
        # Look back 1 second for prior on-ice
        if sec - 1 < 0 or sec - 1 >= len(team_onice_by_sec):
            return set()
        prev_home, prev_away = team_onice_by_sec[sec - 1]
        prev_side = set(prev_home) if side == "home" else set(prev_away)
        missing = None
        for pid in parts:
            if pid not in side_set and pid in prev_side:
                missing = pid
                break
        if missing is None:
            return set()
        # find a candidate to swap out: someone new now not present previously
        cand_out = None
        # prioritize from shift change signals
        now_in_key = ("home_in" if side == "home" else "away_in")
        prev_out_key = ("home_out" if side == "home" else "away_out")
        now_in = set((shift_changes_by_sec.get(sec) or {}).get(now_in_key, set()))
        prev_out = set((shift_changes_by_sec.get(sec - 1) or {}).get(prev_out_key, set()))
        for pid_now in side_set:
            if pid_now not in prev_side and (pid_now in now_in or pid_now in prev_out or True):
                cand_out = pid_now
                break
        return {cand_out} if cand_out is not None else set()

    # Faceoff meta
    fo_meta: Dict[int, Dict[str,Any]] = {}
    for e in evts_sorted:
        if (e.get("type") or "").lower().replace("_","-") == "faceoff":
            d = e.get("details") or {}
            s = _safe_abs_sec(e)
            fo_meta[s] = {
                "zone": (d.get("zoneCode") or "").upper()[:1] or "N",
                "winningPlayerId": d.get("winningPlayerId"),
                "losingPlayerId": d.get("losingPlayerId"),
                "eventOwnerTeamId": d.get("eventOwnerTeamId"),
            }

    # Natural breaks (keep highest-priority per second)
    break_secs = set()
    end_event_at: Dict[int, Dict[str,Any]] = {}

    def _prio(t: str) -> int:
        t = (t or "").lower()
        if t == "goal": return 4
        if t == "penalty": return 3
        if t in {"stoppage","puck-out-of-play","goalie-stopped","icing","offside","timeout","challenge"}: return 2
        if t == "faceoff": return 1
        return 0

    for e in evts_sorted:
        t = (e.get("type") or "").lower().replace("_","-")
        s = _safe_abs_sec(e)
        if t in NATURAL_BREAK_TYPES:
            break_secs.add(s)
            cur = end_event_at.get(s, {})
            if _prio(t) >= _prio(cur.get("type")): end_event_at[s] = {"type": t, "details": e.get("details") or {}}

    # Media timeout (TV) detection: EXPLICIT ONLY (no heuristics)
    tv_timeout_secs = set()
    # explicit via end_event_at summary
    def _tv_flag_from_details(det: Dict[str, Any]) -> bool:
        keys = (
            "reason",
            "secondaryReason",
            "stoppageReason",
            "secondary_reason",
            "stoppage_reason",
        )
        for k in keys:
            v = det.get(k)
            if v is None:
                continue
            txt = str(v).lower()
            if ("tv" in txt) or ("media" in txt) or ("commercial" in txt):
                return True
        return False

    for s, meta in end_event_at.items():
        t = (meta.get("type") or "").lower()
        det = (meta.get("details") or {})
        if t in {"stoppage","goalie-stopped"} and _tv_flag_from_details(det):
            tv_timeout_secs.add(s)
    # explicit via raw events at that second (covers cases where penalty is the end_event_at)
    for e in evts_sorted:
        s = _safe_abs_sec(e)
        t = (e.get("type") or "").lower().replace("_","-")
        if t not in ("stoppage","goalie-stopped"):
            continue
        det = (e.get("details") or {})
        if _tv_flag_from_details(det):
            tv_timeout_secs.add(s)
    # remove previous heuristic stamping completely

    # --- Delayed penalty active seconds: starts at 'delayed-penalty', ends at next 'penalty' whistle ---
    delayed_active_by_sec: List[bool] = [False] * (horizon + 1)
    dp_active = False
    for s in range(0, horizon + 1):
        tset = types_by_sec.get(s, set())
        if "penalty" in tset:
            dp_active = False
        if "delayed-penalty" in tset:
            dp_active = True
        delayed_active_by_sec[s] = dp_active

    # --- Pre-index credited events per second (DIRECTIONAL), store OWNER + PRE-CHANGE on-ice ---
    credit_by_sec: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "home": Counter(), "away": Counter(),
        "owner_team_id": None,
        "onice_home": set(), "onice_away": set()
    })

    # >>> NEW: dedupe multiple goal packets for the same (second, owner team, scorer)
    seen_goal_keys = set()  # (sec_game, owner_team_id, scorerId)

    def _owner_side_for_event(d: Dict[str,Any], home_pre: set, away_pre: set) -> Optional[str]:
        """Return 'home' or 'away' if we can determine the owner; else None."""
        owner_team_id = d.get("eventOwnerTeamId")
        side = None
        if owner_team_id is not None and home_team_id is not None and away_team_id is not None:
            try:
                if int(owner_team_id) == int(home_team_id): side = "home"
                elif int(owner_team_id) == int(away_team_id): side = "away"
            except Exception:
                side = None
        if side is None:
            shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId")
            try:
                shooter = int(shooter) if shooter is not None else None
            except Exception:
                shooter = None
            if shooter is not None:
                if shooter in home_pre: side = "home"
                elif shooter in away_pre: side = "away"
        return side

    events_by_sec: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for e in evts_sorted:
        t = (e.get("type") or "").lower().replace("_","-")
        s = _safe_abs_sec(e)
        d = e.get("details") or {}
        oi = e.get("onice") or {}
        home_pre = set(int(x) for x in (oi.get("home") or []))
        away_pre = set(int(x) for x in (oi.get("away") or []))
        # Always retain personal events for later per-player credit
        if t in ("hit","giveaway","takeaway"):
            events_by_sec[s].append({"type": t, "details": d})
        # Only proceed with team attempt/goal credit for shots/goals
        if t not in SHOT_TYPES and t not in GOAL_TYPES:
            continue

        credit_by_sec[s]["onice_home"] = home_pre
        credit_by_sec[s]["onice_away"] = away_pre
        credit_by_sec[s]["owner_team_id"] = d.get("eventOwnerTeamId")

        # keep a minimal record of the event (shots/blocks/goals) for per-player crediting later
        events_by_sec[s].append({"type": t, "details": d})

        side = _owner_side_for_event(d, home_pre, away_pre)  # 'home' / 'away' / None
        opp  = "away" if side == "home" else ("home" if side == "away" else None)

        if t in SHOT_TYPES:
            # All attempts (including on-goal/miss/block) count toward AF/AA
            # On-target (saved shots + goals) additionally count SF/SA and xG
            xg = float(d.get("xg", 0.0)) if d.get("xg") is not None else 0.0
            if side in ("home","away") and opp is not None:
                # attempts
                credit_by_sec[s][side]["AF"] += 1
                credit_by_sec[s][opp ]["AA"] += 1

                # on-target: saved shot or goal
                if (t in SHOT_ON_GOAL_TYPES) or (t in GOAL_TYPES):
                    credit_by_sec[s][side]["SF"] += 1
                credit_by_sec[s][opp]["SA"] += 1
                credit_by_sec[s][side]["xGF"] += xg
                credit_by_sec[s][opp]["xGA"] += xg

                # block attribution: use blocker if available; else fallback to defender side
                if t in BLOCK_TYPES:
                    # try to detect blocker player id from various key names
                    det = d.get("details") or {}
                    blk_pid = (
                        det.get("blockingPlayerId")
                        or det.get("blockedByPlayerId")
                        or det.get("blockerPlayerId")
                        or det.get("blockerId")
                    )
                    blocker_side: Optional[str] = None
                    try:
                        blk_pid = int(blk_pid) if blk_pid is not None else None
                    except Exception:
                        blk_pid = None
                    if blk_pid is not None:
                        if blk_pid in home_pre:
                            blocker_side = "home"
                        elif blk_pid in away_pre:
                            blocker_side = "away"
                    # Fallback: blocked shots are by the defending team (opposite of shooter/owner)
                    if blocker_side is None:
                        blocker_side = opp  # defender side opposes owner/shooter side
                    defender_side = blocker_side
                    attacker_side = "home" if defender_side == "away" else "away"
                    credit_by_sec[s][defender_side]["BF"] += 1
                    credit_by_sec[s][attacker_side]["BA"] += 1
            else:
                # No side resolved: skip attempt credit to avoid over-attribution
                pass

        elif t in GOAL_TYPES:
            # >>> NEW: only count one goal per (second, owner team, scorer)
            owner_tid = d.get("eventOwnerTeamId")
            scorer = d.get("scoringPlayerId") or d.get("shootingPlayerId")
            goal_key = (s, owner_tid, scorer)
            if goal_key in seen_goal_keys:
                continue
            seen_goal_keys.add(goal_key)

            if side in ("home","away") and opp is not None:
                credit_by_sec[s][side]["GF"] += 1
                credit_by_sec[s][opp]["GA"]  += 1
            else:
                # No side resolved: skip goal credit
                pass
    # --- end directional pre-index ---

    # --- cumulative score up to (and including) each second ---
    horizon = len(team_onice_by_sec) - 1
    cum_home_goals = [0] * (horizon + 2)
    cum_away_goals = [0] * (horizon + 2)
    for s in range(horizon + 1):
        cum_home_goals[s+1] = cum_home_goals[s]
        cum_away_goals[s+1] = cum_away_goals[s]
        cb = credit_by_sec.get(s)
        if cb:
            if int(cb["home"].get("GF", 0)) > 0:
                cum_home_goals[s+1] += int(cb["home"]["GF"])
            if int(cb["away"].get("GF", 0)) > 0:
                cum_away_goals[s+1] += int(cb["away"]["GF"])

    def score_at_start(start_sec: int) -> Tuple[int,int]:
        # goals BEFORE this window start
        return cum_home_goals[start_sec], cum_away_goals[start_sec]

    def score_diff_for_side(side: str, start_sec: int) -> int:
        h, a = score_at_start(start_sec)
        return (h - a) if side == "home" else (a - h)

    def clock_s_at(start_sec: int) -> int:
        return start_sec % SECONDS_PER_PERIOD

    windows: List[Dict[str,Any]] = []
    open_s = 0
    last_strength = strength_at(0)
    last_home, last_away = team_onice_by_sec[0]
    last_fo_zone: Optional[str] = None
    last_fo_winner_team_id: Optional[int] = None
    last_fo_winner_player_id: Optional[int] = None
    last_fo_loser_player_id: Optional[int] = None
    # Break metadata to carry into the NEXT window's start
    last_break_type: Optional[str] = None
    last_break_team_id: Optional[int] = None
    last_break_subtype: Optional[str] = None
    last_media_timeout: bool = False
    media_timeout_next_flag: bool = False
    media_timeout_tag_start_sec: Optional[int] = None

    if 0 in fo_meta:
        last_fo_zone = fo_meta[0]["zone"]
        last_fo_winner_team_id = fo_meta[0].get("eventOwnerTeamId")
        last_fo_winner_player_id = fo_meta[0].get("winningPlayerId")
        last_fo_loser_player_id = fo_meta[0].get("losingPlayerId")

    def flush(end_s: int, reason: str):
        nonlocal open_s, last_strength, last_home, last_away, last_fo_zone, last_fo_winner_team_id, last_fo_winner_player_id, last_fo_loser_player_id, last_break_type, last_break_team_id, last_break_subtype, last_media_timeout, media_timeout_next_flag, media_timeout_tag_start_sec
        if end_s <= open_s:
            return
        start_goalies = goalies_by_team[open_s]
        start_goalie_ids = goalie_ids_by_sec[open_s] if open_s < len(goalie_ids_by_sec) else {"home":0,"away":0}
        end_goalies   = goalies_by_team[max(0, min(end_s-1, horizon))]
        home_end, away_end = team_onice_by_sec[max(0, min(end_s-1, horizon))]
        w = {
            "window_id": f"W{len(windows)+1:04d}",
            "period": period_of(open_s),
            "start_sec": open_s,
            "end_sec": end_s,
            "duration": end_s - open_s,
            "clock_start": clock_str(open_s),
            "strength_global": last_strength,
            "fo_zone": last_fo_zone or "flow",
            "fo_won_team_id": last_fo_winner_team_id,
            "fo_won_player_id": last_fo_winner_player_id,
            "fo_lost_player_id": last_fo_loser_player_id,
            # start-of-window previous break metadata
            "start_prev_break_type": (last_break_type or "period_start"),
            "start_prev_break_team_id": last_break_team_id,
            "start_prev_break_subtype": last_break_subtype,
            # Stamp only if the armed tag matches this window's start second
            "media_timeout_start": int(1 if (media_timeout_tag_start_sec is not None and media_timeout_tag_start_sec == open_s) else 0),
            "end_event_type": reason,
            "home_ids_start": list(last_home),
            "away_ids_start": list(last_away),
            "home_ids_end": list(home_end),
            "away_ids_end": list(away_end),
            "goalies_start": {"home": start_goalies["home"], "away": start_goalies["away"]},
            "goalie_ids_start": {"home": int(start_goalie_ids.get("home",0)), "away": int(start_goalie_ids.get("away",0))},
            "goalies_end":   {"home": end_goalies["home"],   "away": end_goalies["away"]},
        }
        # delayed-penalty flag: must be under delayed-penalty, end with penalty, and have a pulled goalie (either side) during the window
        dp_active_in_window = any(delayed_active_by_sec[s_] for s_ in range(open_s, end_s))
        pulled_any_in_window = any((pulled_home[s_] or pulled_away[s_]) for s_ in range(open_s, end_s))
        w["delayed_penalty"] = int(bool(dp_active_in_window and (reason == "penalty") and pulled_any_in_window))
        if reason == "faceoff" and end_s in fo_meta:
            w["fo_zone_next"] = fo_meta[end_s]["zone"]
        windows.append(w)

        open_s = end_s
        last_strength = strength_at(end_s)
        last_home, last_away = team_onice_by_sec[end_s]
        last_fo_zone = "flow"
        last_fo_winner_team_id = None
        last_fo_winner_player_id = None
        last_fo_loser_player_id = None
        # consume the carry tag if we just stamped it
        if media_timeout_tag_start_sec is not None and media_timeout_tag_start_sec == open_s:
            media_timeout_next_flag = False
            media_timeout_tag_start_sec = None
        # do not clear last_break_* here; it should carry to the next window start until overwritten

    s = 1
    while s <= horizon:
        cur_strength = strength_at(s)
        elapsed = s - open_s

        # 1) natural break first (goal/penalty beats faceoff at same second)
        if s in break_secs:
            base_type = (end_event_at.get(s, {}).get("type") or "stoppage")
            # capture team causing the whistle if available and sub-reason for stoppages (e.g., icing)
            dmeta = (end_event_at.get(s, {}) or {}).get("details") or {}
            sub_reason = str(dmeta.get("reason") or dmeta.get("stoppageReason") or "").lower()
            # Effective reason: if it's a stoppage with reason "icing", treat as icing
            reason = ("icing" if (base_type in {"stoppage","goalie-stopped"} and sub_reason == "icing") else base_type)
            by_tid = dmeta.get("byTeamId") or dmeta.get("teamId") or dmeta.get("committingTeamId") or dmeta.get("eventOwnerTeamId")
            try:
                by_tid = int(by_tid) if by_tid is not None else None
            except Exception:
                by_tid = None
            flush(s, reason)
            # record break for NEXT window start
            last_break_type = reason
            last_break_team_id = by_tid
            # subtype and media timeout
            last_break_subtype = sub_reason if base_type in {"stoppage","goalie-stopped"} else None
            last_media_timeout = (s in tv_timeout_secs) or ("tv" in sub_reason or "media" in sub_reason or "commercial" in sub_reason)
            # Strict rule: stamp ONLY if explicit sec is in tv_timeout_secs
            # The start_sec for the next window must equal s
            if s in tv_timeout_secs:
                media_timeout_next_flag = True
                media_timeout_tag_start_sec = s
            else:
                media_timeout_next_flag = False
                media_timeout_tag_start_sec = None
            if s in fo_meta:
                last_fo_zone = fo_meta[s]["zone"]
                last_fo_winner_team_id = fo_meta[s].get("eventOwnerTeamId")
                last_fo_winner_player_id = fo_meta[s].get("winningPlayerId")
                last_fo_loser_player_id = fo_meta[s].get("losingPlayerId")
            s += 1
            continue

        # 2) pure faceoff-only break
        if s in fo_meta:
            flush(s, "faceoff")
            # mark the faceoff as the break that precedes NEXT window
            last_break_type = "faceoff"
            last_break_team_id = None
            last_break_subtype = None
            last_media_timeout = False
            last_fo_zone = fo_meta[s]["zone"]
            last_fo_winner_team_id = fo_meta[s].get("eventOwnerTeamId")
            last_fo_winner_player_id = fo_meta[s].get("winningPlayerId")
            last_fo_loser_player_id = fo_meta[s].get("losingPlayerId")
            s += 1
            continue

        if cur_strength != last_strength:
            flush(s, "strength")
            s += 1
            continue

        if hard_cap_sec > 0 and elapsed >= hard_cap_sec:
            flush(s, "hard_cap")
            s += 1
            continue

        s += 1

    flush(horizon, "period_end")

    # (Removed) delayed-penalty tagging per request
    delayed_penalty_windows = set()

    # helpers for cameo protection
    def second_has_credit_event(sec: int) -> bool:
        cb = credit_by_sec.get(sec)
        return bool(cb and (cb["home"] or cb["away"]))

    def protective_has_event(sec: int, window_start_sec: int, window_end_sec: int) -> bool:
        if second_has_credit_event(sec):
            return True
        tset = types_by_sec.get(sec, set())
        if tset & MICRO_PROTECT_TYPES:
            return True
        if sec == window_start_sec and "faceoff" in tset:
            return True
        # end-edge protection: protect last included sec only if end_sec has a credited event BEFORE any faceoff
        if sec == window_end_sec - 1 and has_credit_before_faceoff(window_end_sec):
            return True
        return False

    # Track forced cameo drops when a goal participant swap triggers
    forced_drop_cameo_by_sec_side: Dict[Tuple[int, str], set] = defaultdict(set)

    def _maybe_correct_onice_for_goal(
        sec: int,
        ev: Dict[str, Any],
        pre_home_set: set,
        pre_away_set: set,
    ) -> Tuple[set, set]:
        """Apply a tiny jitter correction for goals.
        - Respect sortOrder inside the second: if a shift-change has lower sortOrder than the goal
          and that creates a conflict, treat the shift as after the goal.
        - If a credited skater (scorer/assisters) is missing from the on-ice snapshot, look back 1s.
          If present there, and someone else replaced him now, perform a single 1-for-1 swap for crediting.
        """
        try:
            det = ev.get("details") or {}
            goal_so = int(ev.get("sortOrder", 0))
        except Exception:
            det = ev.get("details") or {}
            goal_so = 0

        # Determine team side from owner team id if available
        owner_tid = det.get("eventOwnerTeamId")
        side_local: Optional[str] = None
        try:
            if owner_tid is not None and home_team_id is not None and int(owner_tid) == int(home_team_id):
                side_local = "home"
            elif owner_tid is not None and away_team_id is not None and int(owner_tid) == int(away_team_id):
                side_local = "away"
        except Exception:
            side_local = None

        # Base sets we will possibly adjust
        base_home = set(pre_home_set)
        base_away = set(pre_away_set)

        # If a same-second shift-change occurs before the goal OR there was any shift at sec-1,
        # use the most recent prior snapshot before that change, scanning back for a stable 5v5 if possible.
        use_prior_snapshot = False
        start_prev = sec - 1
        try:
            evs_this = orders_by_sec.get(sec, [])
            same_sec_shift_before = any((t == "shift-change" and so < goal_so) for so, t in evs_this)
        except Exception:
            same_sec_shift_before = False
        try:
            tminus1_has_shift = bool(shift_changes_by_sec.get(sec - 1)) if sec - 1 >= 0 else False
        except Exception:
            tminus1_has_shift = False
        if same_sec_shift_before or tminus1_has_shift:
            use_prior_snapshot = True
            start_prev = sec - 2 if tminus1_has_shift else sec - 1
        if use_prior_snapshot:
            lookback = 0
            prev_s = start_prev
            while prev_s >= 0 and lookback < 30:
                if prev_s < len(team_onice_by_sec):
                    ph, pa = team_onice_by_sec[prev_s]
                    # Prefer frames with 5 skaters per side; accept first available otherwise
                    if (len(ph) >= 5 and len(pa) >= 5) or lookback == 0:
                        base_home = set(ph)
                        base_away = set(pa)
                        if len(ph) >= 5 and len(pa) >= 5:
                            break
                prev_s -= 1
                lookback += 1

        # Also handle the jitter: if an assist/scorer was subbed off shortly before and replaced now, swap back
        try:
            sc_prev = shift_changes_by_sec.get(sec - 1, {}) if sec - 1 >= 0 else {}
            sc_now = shift_changes_by_sec.get(sec, {})
        except Exception:
            sc_prev, sc_now = {}, {}

        # Participant override: ensure scorer/assist(1/2) are on for the goal team
        parts: List[int] = []
        for k in ("scoringPlayerId", "assist1PlayerId", "assist2PlayerId"):
            v = det.get(k)
            try:
                if v is not None:
                    parts.append(int(v))
            except Exception:
                continue

        if parts:
            if side_local is None:
                # Fall back to owner_side detection using current bases
                side_local = _owner_side_for_event(det, list(base_home), list(base_away))
            # Work on the goal-side set only; allow a single swap
            goal_set = base_home if side_local == "home" else base_away if side_local == "away" else None
            opp_set  = base_away if side_local == "home" else base_home if side_local == "away" else None
            if isinstance(goal_set, set):
                # find first missing participant
                missing = None
                for pid in parts:
                    if pid not in goal_set:
                        missing = pid
                        break
                # Look back up to 3 seconds for the missing credited skater
                if missing is not None:
                    prev_side_set = None
                    for back in range(1, 4):
                        s_prev = sec - back
                        if s_prev < 0 or s_prev >= len(team_onice_by_sec):
                            continue
                        prev_home, prev_away = team_onice_by_sec[s_prev]
                        prev_side_tmp = set(prev_home) if side_local == "home" else set(prev_away)
                        if missing in prev_side_tmp:
                            prev_side_set = prev_side_tmp
                            break
                    if prev_side_set is not None:
                        # choose a cameo present now but not in the prev snapshot
                        cand_out = None
                        now_in_key = ("home_in" if side_local == "home" else "away_in")
                        prev_out_key = ("home_out" if side_local == "home" else "away_out")
                        now_in = set((sc_now or {}).get(now_in_key, set()))
                        prev_out = set((sc_prev or {}).get(prev_out_key, set()))
                        for pid_now in list(goal_set):
                            if pid_now not in prev_side_set and (pid_now in now_in or pid_now in prev_out or True):
                                cand_out = pid_now
                                break
                        if cand_out is not None:
                            try:
                                goal_set.discard(cand_out)
                                goal_set.add(missing)
                                forced_drop_cameo_by_sec_side[(sec, side_local)].add(int(cand_out))
                            except Exception:
                                pass
                # write back
                if side_local == "home":
                    base_home = goal_set
                elif side_local == "away":
                    base_away = goal_set

        return base_home, base_away

    # Map rink faceoff zone to team-relative zone_start label using FO owner team id
    def zone_start_for(team_side: str,
                       fo_zone: Optional[str],
                       owner_team_id: Optional[int],
                       home_tid: Optional[int],
                       away_tid: Optional[int]) -> str:
        z = (fo_zone or "flow").upper()
        if z == "FLOW":
            return "flow"
        if z == "N":
            return "NZ"
        # determine which side (home/away) owns the FO per eventOwnerTeamId
        owner_side: Optional[str] = None
        try:
            if owner_team_id is not None and home_tid is not None and away_tid is not None:
                if int(owner_team_id) == int(home_tid):
                    owner_side = "home"
                elif int(owner_team_id) == int(away_tid):
                    owner_side = "away"
        except Exception:
            owner_side = None
        if z == "O":
            # Offensive zone for the owner; defender is DZ
            if owner_side is None:
                return "flow"
            return "OZ" if team_side == owner_side else "DZ"
        if z == "D":
            # Defensive zone for the owner; opponent is OZ
            if owner_side is None:
                return "flow"
            return "DZ" if team_side == owner_side else "OZ"
        return "flow"

    # Team-relative after-icing instrument
    def after_icing_for_side(start_break_type: Optional[str],
                             start_break_team_id: Optional[int],
                             side: str,
                             home_team_id: Optional[int],
                             away_team_id: Optional[int],
                             fo_zone: Optional[str]) -> bool:
        # Simplest rule as requested: if previous break was icing, flag True
        return (str(start_break_type or "").lower() == "icing")

    # --- Standings/schedule helpers (best-effort) ---
    def _load_csv_rows(path: str) -> List[Dict[str, Any]]:
        try:
            import csv
            with open(path, "r", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def _find_standings_dir() -> Optional[str]:
        # 1) explicit CLI
        if CLI_STANDINGS_DIR and os.path.isdir(CLI_STANDINGS_DIR):
            if debug_standings:
                print({"debug":"standings","found_dir_explicit": CLI_STANDINGS_DIR})
            return CLI_STANDINGS_DIR
        # 2) near input raw (../standings)
        try:
            in_dir = os.path.abspath(os.path.dirname(CLI_IN_PATH)) if CLI_IN_PATH else None
            raw_dir = os.path.dirname(in_dir)
            rel_dir = os.path.join(raw_dir, "standings")
            if os.path.isdir(rel_dir):
                if debug_standings:
                    print({"debug":"standings","found_dir_relative": rel_dir})
                return rel_dir
        except Exception:
            pass
        # 3) fallbacks under artifacts
        for rel in (
            os.path.join("artifacts","dumps","raw","standings"),
            os.path.join("artifacts","dumps2","raw","standings"),
        ):
            if os.path.isdir(rel):
                if debug_standings:
                    print({"debug":"standings","found_dir": rel})
                return rel
        return None

    def _lower_keys(row: Dict[str, Any]) -> Dict[str, Any]:
        return {str(k).lower(): v for k, v in row.items()}

    def _parse_date_any(val: Any) -> Optional[str]:
        s = str(val or "").strip()
        if not s:
            return None
        # accept YYYY-MM-DD or full ISO
        if len(s) >= 10:
            return s[:10]
        return None

    # discover game date (best-effort from root metadata)
    game_date_str: Optional[str] = None
    try:
        root_game = evts_sorted[0].get("game", {})
        for key in ("date","gameDate","startTimeUTC","game_date"):
            if root_game.get(key):
                game_date_str = _parse_date_any(root_game.get(key))
                if debug_standings:
                    print({"debug":"standings","source":"pbp_root","key": key, "game_date": game_date_str})
                    break
    except Exception:
        game_date_str = None

    _standings_cache: Dict[Tuple[int,str], Tuple[Optional[int], Optional[bool], Optional[int]]] = {}

    def compute_standings_and_b2b(team_id: Optional[int]) -> Tuple[Optional[int], Optional[bool], Optional[int]]:
        if not isinstance(team_id, int):
            if debug_standings:
                print({"debug":"standings","error":"team_id_missing", "team_id": team_id})
            return None, None, None
        standings_dir = _find_standings_dir()
        if not standings_dir:
            if debug_standings:
                print({"debug":"standings","error":"standings_dir_not_found"})
            return None, None, None
        # Determine game date from game_results by gamePk if not already set
        nonlocal game_date_str
        gdate = game_date_str
        # Prefer the explicit filenames provided by user, then fallback to scanning
        explicit_game_results = os.path.join(standings_dir, "game_results_20242025.csv")
        explicit_by_date = os.path.join(standings_dir, "standings_by_date_20242025.csv")
        explicit_after_each = os.path.join(standings_dir, "standings_after_each_game_20242025.csv")
        # 1) Resolve game date
        try:
            used_game_results_files: List[str] = []
            if os.path.exists(explicit_game_results):
                used_game_results_files.append(explicit_game_results)
            else:
                # fallback: any game_results_*.csv in dir
                for name in os.listdir(standings_dir):
                    if "game_results" in name and name.endswith(".csv"):
                        used_game_results_files.append(os.path.join(standings_dir, name))
            for path in used_game_results_files:
                rows = _load_csv_rows(path)
                if debug_standings:
                    print({"debug":"standings","scan":"game_results","file": os.path.basename(path), "rows": len(rows)})
                for r in rows:
                    lr = _lower_keys(r)
                    gp = lr.get("gamepk") or lr.get("game_pk") or lr.get("gameid")
                    try:
                        gp = int(gp) if gp is not None else None
                    except Exception:
                        gp = None
                    if gp is not None and game_pk_local is not None and gp == int(game_pk_local):
                        gd = _parse_date_any(lr.get("date") or lr.get("game_date") or lr.get("gamedate"))
                        if gd:
                            gdate = gd
                            game_date_str = gd
                            if debug_standings:
                                print({"debug":"standings","resolved_game_date_from":"game_results","file": os.path.basename(path), "gamePk": int(game_pk_local), "game_date": gd})
                break
                if gdate:
                    break
        except Exception:
            pass

        cache_key = (team_id, gdate or "")
        if cache_key in _standings_cache:
            if debug_standings:
                print({"debug":"standings","cache_hit": True, "team_id": team_id, "game_date": gdate})
            return _standings_cache[cache_key]

        # Build team game dates for rest/B2B
        dates_for_team: List[str] = []
        try:
            used_game_results_files: List[str] = []
            if os.path.exists(explicit_game_results):
                used_game_results_files.append(explicit_game_results)
            else:
                for name in os.listdir(standings_dir):
                    if "game_results" in name and name.endswith(".csv"):
                        used_game_results_files.append(os.path.join(standings_dir, name))
            for path in used_game_results_files:
                rows = _load_csv_rows(path)
                if debug_standings:
                    print({"debug":"standings","team_dates_scan":"game_results","file": os.path.basename(path), "rows": len(rows)})
                for r in rows:
                    lr = _lower_keys(r)
                    d = _parse_date_any(lr.get("date") or lr.get("game_date") or lr.get("gamedate"))
                    if not d:
                        continue
                    matched = False
                    # flexible team id matching
                    for key in ("team_id","teamid","team"):
                        try:
                            vv = lr.get(key)
                            if vv is not None and int(vv) == team_id:
                                matched = True
                                break
                        except Exception:
                            pass
                    if not matched:
                        for key in ("home_team_id","away_team_id","hometeamid","awayteamid","home_id","away_id"):
                            try:
                                vv = lr.get(key)
                                if vv is not None and int(vv) == team_id:
                                    matched = True
                                    break
                            except Exception:
                                pass
                    if matched:
                        dates_for_team.append(d)
        except Exception:
            pass
        dates_for_team = sorted(set(dates_for_team))
        rest_days: Optional[int] = None
        b2b: Optional[bool] = None
        if gdate and dates_for_team:
            from datetime import datetime
            try:
                gd = datetime.strptime(gdate, "%Y-%m-%d").date()
                prev = None
                for d in dates_for_team:
                    if d < gdate:
                        prev = d
                if prev is not None:
                    pd = datetime.strptime(prev, "%Y-%m-%d").date()
                    diff = (gd - pd).days
                    rest_days = diff
                    b2b = (diff == 1)
                    if debug_standings:
                        print({"debug":"standings","team_id": team_id, "prev_date": str(pd), "game_date": str(gd), "rest_days": rest_days, "b2b": b2b})
                else:
                    # First game of season for this team: treat as long rest and not B2B
                    rest_days = 999
                    b2b = False
                    if debug_standings:
                        print({"debug":"standings","team_id": team_id, "first_game": True, "game_date": str(gd), "rest_days": rest_days, "b2b": b2b})
            except Exception:
                pass

        # standing prior: from standings_by_date (preferred)
        standing_prior: Optional[int] = None
        try:
            used_by_date_files: List[str] = []
            if os.path.exists(explicit_by_date):
                used_by_date_files.append(explicit_by_date)
            else:
                for name in os.listdir(standings_dir):
                    if "standings_by_date" in name and name.endswith(".csv"):
                        used_by_date_files.append(os.path.join(standings_dir, name))
            for path in used_by_date_files:
                rows = _load_csv_rows(path)
                if not rows:
                    continue
                # choose exact game date; if not found, choose max date <= game date
                best_date = None
                for r in rows:
                    lr = _lower_keys(r)
                    d = _parse_date_any(lr.get("date") or lr.get("game_date"))
                    if not d:
                        continue
                    tid = lr.get("team_id") or lr.get("teamid") or lr.get("team")
                    try:
                        tid = int(tid) if tid is not None else None
                    except Exception:
                        tid = None
                    if tid != team_id:
                        continue
                    if gdate and d == gdate:
                        best_date = d
                        break
                    if gdate and d < gdate:
                        best_date = d
                if best_date:
                    if debug_standings:
                        print({"debug":"standings","file": os.path.basename(path), "best_date": best_date, "team_id": team_id})
                    # get rank on best_date
                    for r in rows:
                        lr = _lower_keys(r)
                        d = _parse_date_any(lr.get("date") or lr.get("game_date"))
                        if d != best_date:
                            continue
                        tid = lr.get("team_id") or lr.get("teamid") or lr.get("team")
                        try:
                            tid = int(tid) if tid is not None else None
                        except Exception:
                            tid = None
                        if tid != team_id:
                            continue
                        rank_val = (
                            lr.get("rank") or lr.get("standing") or lr.get("standing_rank")
                            or lr.get("rank_overall") or lr.get("league_rank") or lr.get("league_rank_unique")
                        )
                        try:
                            standing_prior = int(rank_val)
                        except Exception:
                            standing_prior = None
                        break
                if standing_prior is not None:
                    break
        except Exception:
            pass

        # Fallback: standings_after_each_game (use last <= game date or <= gamePk)
        if standing_prior is None:
            try:
                used_after_files: List[str] = []
                if os.path.exists(explicit_after_each):
                    used_after_files.append(explicit_after_each)
                else:
                    for name in os.listdir(standings_dir):
                        if "standings_after_each_game" in name and name.endswith(".csv"):
                            used_after_files.append(os.path.join(standings_dir, name))
                fallback_candidate: Optional[Tuple[str, int, Optional[str]]] = None  # (rank, after_gamePk, date)
                for path in used_after_files:
                    rows = _load_csv_rows(path)
                    if debug_standings:
                        print({"debug":"standings","scan":"after_each","file": os.path.basename(path), "rows": len(rows)})
                    for r in rows:
                        lr = _lower_keys(r)
                        tid = lr.get("team_id")
                        try:
                            tid = int(tid) if tid is not None else None
                        except Exception:
                            tid = None
                        if tid != team_id:
                            continue
                        date_row = _parse_date_any(lr.get("date") or lr.get("game_date"))
                        after_pk = lr.get("after_gamepk") or lr.get("after_game_pk")
                        try:
                            after_pk = int(after_pk) if after_pk is not None else None
                        except Exception:
                            after_pk = None
                        rank_val = (
                            lr.get("league_rank") or lr.get("league_rank_unique") or lr.get("rank")
                            or lr.get("standing") or lr.get("standing_rank") or lr.get("rank_overall")
                        )
                        # choose: latest <= game date if we know date, else latest <= gamePk
                        ok = False
                        if gdate and date_row and date_row <= gdate:
                            ok = True
                        elif (not gdate) and (game_pk_local is not None) and (after_pk is not None) and (after_pk <= int(game_pk_local)):
                            ok = True
                        if ok and rank_val is not None:
                            try:
                                rank_int = int(rank_val)
                            except Exception:
                                rank_int = None
                            if rank_int is None:
                                continue
                            # keep the best (latest) candidate by date or by after_pk
                            if fallback_candidate is None:
                                fallback_candidate = (str(rank_int), after_pk or 0, date_row)
                            else:
                                # prefer newer date; if tie/unknown, prefer larger after_pk
                                old_rank, old_pk, old_date = fallback_candidate
                                newer = False
                                if gdate and date_row and old_date and date_row > old_date:
                                    newer = True
                                elif (not gdate) and (after_pk or 0) > (old_pk or 0):
                                    newer = True
                                if newer:
                                    fallback_candidate = (str(rank_int), after_pk or 0, date_row)
                if fallback_candidate is not None:
                    try:
                        standing_prior = int(fallback_candidate[0])
                    except Exception:
                        standing_prior = None
                    if debug_standings:
                        print({"debug":"standings","fallback":"after_each","standing_prior": standing_prior, "team_id": team_id})
            except Exception:
                pass

        _standings_cache[cache_key] = (standing_prior, b2b, rest_days)
        if debug_standings:
            print({"debug":"standings","computed": True, "team_id": team_id, "game_date": gdate, "standing_prior": standing_prior, "b2b": b2b, "rest_days": rest_days})
        return _standings_cache[cache_key]

    # ---------- Per-team rows ----------
    team_rows: List[Dict[str,Any]] = []
    for w in windows:
        for side in ("home", "away"):
            opp = "away" if side == "home" else "home"
            ids_start = w[f"{side}_ids_start"]
            opp_start = w[f"{opp}_ids_start"]
            ids_end   = w[f"{side}_ids_end"]
            opp_end   = w[f"{opp}_ids_end"]
            g_for  = int(w["goalies_start"][side])
            g_opp  = int(w["goalies_start"][opp])

            sk_for = len(ids_start)
            sk_opp = len(opp_start)

            # team-relative zone for this side
            _zone_rel = zone_start_for(side, w["fo_zone"], w.get("fo_won_team_id"), home_team_id, away_team_id)
            _ai_flag = int(after_icing_for_side(
                w.get("start_prev_break_type"), w.get("start_prev_break_team_id"), side, home_team_id, away_team_id, w.get("fo_zone")
            ))
            _ai_by_team = int(_ai_flag == 1 and _zone_rel == "DZ")
            _ai_by_opp  = int(_ai_flag == 1 and _zone_rel == "OZ")

            row = {
                "window_id": w["window_id"],
                "period": w["period"],
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "duration": w["duration"],
                "clock_start": w["clock_start"],
                "end_event_type": w["end_event_type"],
                "strength_global": w["strength_global"],
                "team_side": side,
                "skaters_for": sk_for,
                "skaters_against": sk_opp,
                "goalie_for": g_for,
                "goalie_against": g_opp,
                "strength_team": strength_for_team(sk_for, sk_opp, g_for, g_opp),
                "strength_team_label": (
                    "powerplay" if strength_for_team(sk_for, sk_opp, g_for, g_opp) == "PP" else (
                    "penalty_kill" if strength_for_team(sk_for, sk_opp, g_for, g_opp) == "PK" else (
                    "even_strength" if strength_for_team(sk_for, sk_opp, g_for, g_opp) == "5v5" else (
                    "four_on_four" if strength_for_team(sk_for, sk_opp, g_for, g_opp) == "4v4" else (
                    "empty_net_for" if strength_for_team(sk_for, sk_opp, g_for, g_opp) == "EN_for" else strength_for_team(sk_for, sk_opp, g_for, g_opp)
                ))))),
                "fo_zone": _zone_rel,
                "zone_start": _zone_rel,
                "score_diff_start": score_diff_for_side(side, w["start_sec"]),
                "clock_s": clock_s_at(w["start_sec"]),
                "start_prev_break_type": w.get("start_prev_break_type"),
                "start_prev_break_subtype": w.get("start_prev_break_subtype"),
                "media_timeout_start": int(bool(w.get("media_timeout_start"))),
                "after_icing": _ai_flag,
                # Team-relative icing source flags via zone when after-icing
                "after_icing_by_team": _ai_by_team,
                "after_icing_by_opponent": _ai_by_opp,
                "fo_won_team_id": w["fo_won_team_id"],
                "home_away": (side == "home"),
                "long_change": int(is_long_change(w["start_sec"])),
                "team_ids_start": ids_start,
                "opp_ids_start":  opp_start,
                "team_ids_end":   ids_end,
                "opp_ids_end":    opp_end,
                "team_rows_marker": 1 if True else 1,
            }
            # fill standings/b2b best-effort once per side
            tid = home_team_id if side=="home" else away_team_id
            rank_prior, b2b_calc, rest_days_calc = compute_standings_and_b2b(tid)
            if rank_prior is not None:
                row["standing_prior"] = int(rank_prior)
            if b2b_calc is not None:
                row["b2b_team"] = int(bool(b2b_calc))
            if rest_days_calc is not None:
                row["rest_days_team"] = int(rest_days_calc)

            # Faceoff flags at window start (team-relative)
            won_tid = w.get("fo_won_team_id")
            team_tid = home_team_id if side == "home" else away_team_id
            # Gate 'seen' by being on-ice at the faceoff start (team had skaters on at start)
            ids_start_set = set(row.get("team_ids_start", []) or [])
            fo_seen_start = (won_tid is not None and len(ids_start_set) > 0)
            fo_won_start  = (fo_seen_start and team_tid is not None and int(won_tid) == int(team_tid))
            row.update({
                "fo_seen_start": int(bool(fo_seen_start)),
                "fo_won_start":  int(bool(fo_won_start)),
                "fo_lost_start": int(bool(fo_seen_start) and not bool(fo_won_start)),
            })

            # Home last-change opportunity at window start (rule-based)
            stoppage_types = {
                'goal','penalty','icing','offside','stoppage',
                'puck-out-of-play','goalie-stopped','timeout','challenge','period_start'
            }
            spbt = (w.get("start_prev_break_type") or "").lower()
            is_home_side = (side == "home")
            fo_expected = bool(fo_seen_start) or (spbt in stoppage_types)
            excluded = spbt in {"strength","hard_cap","flow",""}
            _hlc = bool(is_home_side and fo_expected and (not excluded))
            if str(row.get("zone_start")).lower() == "flow":
                _hlc = False
            # If the home team committed the icing, they cannot change → force 0
            try:
                prev_tid = w.get("start_prev_break_team_id")
                if spbt == "icing" and is_home_side and prev_tid is not None and home_team_id is not None and int(prev_tid) == int(home_team_id):
                    _hlc = False
            except Exception:
                pass
            row["home_last_change_opportunity"] = int(_hlc)

            # pulled-goalie exposure at window start
            gp_since = since_pulled_home[w["start_sec"]] if side == "home" else since_pulled_away[w["start_sec"]]
            row.update({
                "pulled_goalie_start": int(bool(g_for == 0)),
                "goalie_pulled_since": int(gp_since),
            })
            team_rows.append(row)

    # ---------- Per-player rows ----------
    by_sec_home = {s: set(team_onice_by_sec[s][0]) for s in range(len(team_onice_by_sec))}
    by_sec_away = {s: set(team_onice_by_sec[s][1]) for s in range(len(team_onice_by_sec))}

    player_rows: List[Dict[str,Any]] = []

    def _debug_log_ga(context: str, player_id: int, player_side: str, s_evt: int, win_id: str, start_sec: int, end_sec: int):
        if not debug_ga:
            return
        cb = credit_by_sec.get(s_evt, {})
        owner_tid = cb.get("owner_team_id")
        owner_side = None
        if owner_tid is not None and home_team_id is not None and away_team_id is not None:
            try:
                if int(owner_tid) == int(home_team_id): owner_side = "home"
                elif int(owner_tid) == int(away_team_id): owner_side = "away"
            except Exception:
                owner_side = None
        ev_types = sorted(list(types_by_sec.get(s_evt, set())))
        pre_home = sorted(list(cb.get("onice_home", set()) or []))
        pre_away = sorted(list(cb.get("onice_away", set()) or []))
        name = (player_name_map.get(player_id) if isinstance(player_name_map, dict) else None)
        print(
            {
                "debug": "GA_credit",
                "context": context,
                "game_second": s_evt,
                "clock": clock_str(s_evt),
                "window_id": win_id,
                "window": [start_sec, end_sec],
                "playerId": player_id,
                "playerName": name,
                "player_side": player_side,
                "event_types_at_s": ev_types,
                "owner_team_id": owner_tid,
                "owner_side": owner_side,
                "pre_onice_home": pre_home,
                "pre_onice_away": pre_away,
                "cb_home": dict(credit_by_sec.get(s_evt, {}).get("home", {})),
                "cb_away": dict(credit_by_sec.get(s_evt, {}).get("away", {})),
            }
        )

    # helper: TSF bucket shares by per-second time-since-last-faceoff
    def tsf_bucket_shares(start_sec: int, end_sec: int) -> Dict[str,float]:
        dur = max(0, end_sec - start_sec)
        if dur == 0:
            return {"tsf_0_5":0,"tsf_6_20":0,"tsf_21_60":0,"tsf_61p":0}
        shares = {"tsf_0_5":0,"tsf_6_20":0,"tsf_21_60":0,"tsf_61p":0}
        fo_seconds = sorted(fo_meta.keys())
        k = 0
        for s in range(start_sec, end_sec):
            while k + 1 < len(fo_seconds) and fo_seconds[k+1] <= s:
                k += 1
            last_fo = fo_seconds[k] if fo_seconds and fo_seconds[0] <= s else None
            tsf = (s - last_fo) if last_fo is not None else 10_000
            if tsf <= 5: shares["tsf_0_5"] += 1
            elif tsf <= 20: shares["tsf_6_20"] += 1
            elif tsf <= 60: shares["tsf_21_60"] += 1
            else: shares["tsf_61p"] += 1
        for k in shares: shares[k] /= float(dur)
        return shares

    # --- NEW: shift metrics helpers ---
    def onice_elapsed_before_second(p: int, sec: int, onice_map: Dict[int, set]) -> int:
        """How long (in seconds) player p had been on before 'sec' (exclusive)."""
        t = sec - 1
        streak = 0
        while t >= 0 and p in onice_map.get(t, ()):  # look back until off-ice
            streak += 1
            t -= 1
        return streak

    def last_shift_and_rest_before(p: int, sec: int, onice_map: Dict[int, set]) -> Tuple[int, int, int]:
        """Return (last_shift_len_s, rest_gap_raw_s, prev_end_sec) before 'sec'.
        If player is on at 'sec', we locate the previous completed shift and OFF gap before the
        current shift. If no previous shift exists, last=0, rest_gap_raw=curr_start, prev_end=-1.
        If player is off at 'sec', returns (0,0,-1).

        Important: Treat period boundaries as hard breaks. The "current shift" cannot extend
        backward into a prior period. This ensures intermission rest is handled as a gap
        (e.g., 900s between regulation periods; 120s before OT)."""
        if sec <= 0 or p not in onice_map.get(sec, ()):  # only for players on at this instant
            return 0, 0, -1
        curr_start = sec
        sec_period = period_of(sec)
        # Do not cross period boundary when walking back to find current shift start
        while (
            curr_start - 1 >= 0
            and period_of(curr_start - 1) == sec_period
            and p in onice_map.get(curr_start - 1, ())
        ):
            curr_start -= 1
        t = curr_start - 1
        while t >= 0 and p not in onice_map.get(t, ()):  # off gap
            t -= 1
        if t < 0:
            return 0, curr_start, -1
        prev_end = t
        while t - 1 >= 0 and p in onice_map.get(t - 1, ()):  # previous on segment
            t -= 1
        prev_start = t
        last_len = max(0, prev_end - prev_start + 1)
        rest_len = max(0, curr_start - (prev_end + 1))
        return int(last_len), int(rest_len), int(prev_end)

    def shift_count_in_window(p: int, start_sec: int, end_sec: int, onice_map: Dict[int, set]) -> int:
        """
        Count contiguous on-ice segments that INTERSECT [start_sec, end_sec).
        - If player is already on at start_sec, that counts as 1.
        - Each off→on transition within the window adds 1.
        """
        if end_sec <= start_sec:
            return 0

        count = 0

        # Does a segment already intersect at the window start?
        on_at_start = (p in onice_map.get(start_sec, ()))
        if on_at_start:
            count += 1

        # Count new entries after the start second
        for s in range(start_sec + 1, end_sec):
            now_on  = (p in onice_map.get(s, ()))
            prev_on = (p in onice_map.get(s - 1, ()))
            if now_on and not prev_on:
                count += 1

        return count

    for w in windows:
        start_s, end_s = w["start_sec"], w["end_sec"]
        dur = max(0, end_s - start_s)
        if dur == 0:
            continue

        for side in ("home","away"):
            opp = "away" if side == "home" else "home"

            ids_start = set(w[f"{side}_ids_start"])
            opp_start = set(w[f"{opp}_ids_start"])

            onice_map = by_sec_home if side=="home" else by_sec_away
            opp_map   = by_sec_away if side=="home" else by_sec_home

            seen: Counter[int] = Counter()
            for s in range(start_s, end_s):
                for p in onice_map.get(s, ()): seen[p] += 1
            if not seen:
                continue

            # outcomes: credit_by_sec is already directional
            def event_for_against_for_second(s: int, player_side: str) -> Counter:
                cb = credit_by_sec.get(s)
                if not cb:
                    return Counter()
                return Counter(cb[player_side])

            for p, sec_i in seen.items():
                seconds_i = int(sec_i)
                if seconds_i <= 0:
                    continue

                # ----- Edge-only tiny cameo drop with protective events -----
                if DROP_PLAYER_EDGE_ROWS and seconds_i <= EDGE_CAMEO_SEC:
                    first_p, last_p, protected = None, None, False
                    for s in range(start_s, end_s):
                        if p in onice_map.get(s, ()):
                            if first_p is None: first_p = s
                            last_p = s
                            if protective_has_event(s, start_s, end_s):
                                protected = True
                    # NEW: Protect if player is in the credited goal freeze set at end_sec
                    try:
                        if str(w.get("end_event_type")) == "goal":
                            # Build the freeze set used for goal credit (side-aware)
                            # Reuse the same logic as in _maybe_correct_onice_for_goal but only to detect membership
                            use_sec = end_s
                            # If t-1 shift exists, scan back before it
                            start_prev = end_s - 2 if (end_s - 1 >= 0 and shift_changes_by_sec.get(end_s - 1)) else end_s - 1
                            freeze_home, freeze_away = set(), set()
                            # Prefer most recent prior with stable 5v5; else use immediate prior
                            chosen = None
                            for back in range(0, 30):
                                ss = start_prev - back if start_prev >= 0 else end_s - 1
                                if ss < 0: break
                                if ss < len(team_onice_by_sec):
                                    hh, aa = team_onice_by_sec[ss]
                                    freeze_home, freeze_away = set(hh), set(aa)
                                    chosen = ss
                                    if len(hh) >= 5 and len(aa) >= 5:
                                        break
                            freeze_set = freeze_home if side == "home" else freeze_away
                            if p in freeze_set:
                                protected = True
                    except Exception:
                        pass
                    if first_p is not None:
                        at_edge = (first_p == start_s) or (last_p == end_s - 1)
                        # Force drop if a goal swap flagged this cameo at end edge
                        try:
                            if at_edge and last_p == end_s - 1 and p in forced_drop_cameo_by_sec_side.get((end_s, side), set()):
                                continue
                        except Exception:
                            pass
                        # New hard rule: if there was a shift-change at end_s-1 and window ends on a goal,
                        # ignore that change entirely for this window — drop any player who arrived at end_s-1
                        try:
                            if at_edge and last_p == end_s - 1 and str(w.get("end_event_type")) == "goal":
                                cameo_in = set((shift_changes_by_sec.get(end_s - 1) or {}).get(f"{side}_in", set()))
                                if p in cameo_in:
                                    continue
                        except Exception:
                            pass
                        # New stricter rule: allow <=4s edge cameos ONLY if
                        # (a) the cameo spans the entire window (small windows), or
                        # (b) the window ends on a goal (reason == "goal"), or
                        # (c) the end-second is a goalie-stopped after a SOG (SOG precedes goalie-stopped in sort order), or
                        # (d) the final second has a defensive save before whistle/FO (block or takeaway before the boundary)
                        small_window = (end_s - start_s) <= 10
                        window_reason = str(w.get("end_event_type"))
                        goalie_stop_protected = has_sog_before_goalie_stop(end_s)
                        defense_save_protected = has_defense_save_before_whistle(end_s)
                        end_is_goal = (window_reason == "goal")
                        if not small_window and at_edge and not (end_is_goal or goalie_stop_protected or defense_save_protected):
                            # if not protected by the stricter cases, drop the cameo row
                            continue
                        if at_edge and not protected:
                            continue
                # -----------------------------------------------------------

                # exposures
                pp_sec_i = 0
                pk_sec_i = 0
                team_strength = strength_for_team(
                    skaters_for=len(w[f"{side}_ids_start"]),
                    skaters_against=len(w[f"{opp}_ids_start"]),
                    goalie_for=int(w["goalies_start"][side]),
                    goalie_against=int(w["goalies_start"][opp]),
                )
                if team_strength == "PP": pp_sec_i = seconds_i
                if team_strength == "PK": pk_sec_i = seconds_i

                # outcomes: event-driven credit only while p is on pre-change on-ice for that event/second
                Y = Counter(); xgF = 0.0; xgA = 0.0
                for s in range(start_s, end_s):
                    # boundary guard — include start-second events only if they occur AFTER the start boundary
                    if s == start_s and second_has_credit_event(s):
                        # Determine start boundary; prefer same_sec_order of faceoff when available, else sortOrder
                        start_boundary_order = None
                        start_face_sso = None
                        evs_so = sorted(orders_by_sec.get(start_s, []))
                        boundary_types = {"faceoff","stoppage","goal","goalie-stopped","penalty","timeout","challenge"}
                        # faceoff same-sec order at start
                        try:
                            sso_list = [sso for sso, t in sorted(sso_by_sec.get(start_s, [])) if t == "faceoff"]
                            if sso_list:
                                start_face_sso = min(sso_list)
                        except Exception:
                            start_face_sso = None
                        # boundary sortOrder fallback
                        face_orders = [so for so, t in evs_so if t == "faceoff"]
                        if face_orders:
                            start_boundary_order = min(face_orders)
                        else:
                            boundary_orders = [so for so, t in evs_so if t in boundary_types]
                            if boundary_orders:
                                start_boundary_order = min(boundary_orders)
                        # filter the events list in-place to keep only those with sortOrder > boundary
                        filtered = []
                        for ev in events_by_sec.get(s, []):
                            et = ev.get("type")
                            if not et: continue
                            if (et not in SHOT_TYPES) and (et not in GOAL_TYPES):
                                continue
                            so = int(ev.get("sortOrder", 0))
                            # Primary: same_sec_order vs faceoff at start; include only if sso > face_sso
                            if start_face_sso is not None:
                                try:
                                    sso_ev = int(ev.get("same_sec_order", 0))
                                except Exception:
                                    sso_ev = None
                                if sso_ev is None or sso_ev <= start_face_sso:
                                    continue
                            elif start_boundary_order is not None and so <= start_boundary_order:
                                # Fallback to sortOrder if no same_sec_order boundary
                                continue
                            filtered.append(ev)
                        # If nothing remains after filtering, skip this second
                        if not filtered:
                            continue
                        # Use the filtered events for start-second boundary
                        evs = filtered
                    # If no start-second boundary filtering applied, fetch raw events
                    if not (s == start_s and second_has_credit_event(s)):
                        evs = events_by_sec.get(s, [])
                    if not evs:
                        continue

                    cb = credit_by_sec.get(s) or {}
                    pre_home = list(cb.get("onice_home", ()))
                    pre_away = list(cb.get("onice_away", ()))

                    for ev in evs:
                        et = ev.get("type")
                        if not et:
                            continue
                        if (et not in SHOT_TYPES) and (et not in GOAL_TYPES):
                            continue
                        det = ev.get("details") or {}
                        ev_side = _owner_side_for_event(det, pre_home, pre_away)
                        if ev_side not in ("home","away"):
                            continue
                        ev_opp = "away" if ev_side == "home" else "home"
                        # If there was any shift-change at the prior second, freeze goal credit on the prior snapshot
                        if et in GOAL_TYPES and (shift_changes_by_sec.get(s-1) or {}):
                            # use on-ice of s-1 for crediting (freeze pre-change six)
                            if s - 1 >= 0 and (s - 1) < len(team_onice_by_sec):
                                prev_home_ids, prev_away_ids = team_onice_by_sec[s-1]
                                pre_home2 = list(prev_home_ids)
                                pre_away2 = list(prev_away_ids)
                            else:
                                pre_home2, pre_away2 = pre_home, pre_away
                            corr_home, corr_away = _maybe_correct_onice_for_goal(s, ev, set(pre_home2), set(pre_away2))
                            on_for = list(corr_home) if ev_side == "home" else list(corr_away)
                            on_opp = list(corr_away) if ev_side == "home" else list(corr_home)
                        elif et in GOAL_TYPES:
                            corr_home, corr_away = _maybe_correct_onice_for_goal(s, ev, set(pre_home), set(pre_away))
                            on_for = list(corr_home) if ev_side == "home" else list(corr_away)
                            on_opp = list(corr_away) if ev_side == "home" else list(corr_home)
                        else:
                            on_for = pre_home if ev_side == "home" else pre_away
                            on_opp = pre_away if ev_side == "home" else pre_home
                        xg = float(det.get("xg", 0.0)) if det.get("xg") is not None else 0.0

                        # Always AF/AA from owner/shooter perspective
                        if p in on_for:
                            Y["AF"] += 1
                            if (et in SHOT_ON_GOAL_TYPES) or (et in GOAL_TYPES):
                                Y["SF"] += 1
                                xgF += xg
                        elif p in on_opp:
                            Y["AA"] += 1
                            if (et in SHOT_ON_GOAL_TYPES) or (et in GOAL_TYPES):
                                Y["SA"] += 1
                                xgA += xg
                        # For blocked shots, attribute BF/BA by blocker/defender side
                        if et in BLOCK_TYPES:
                            blk_det = det
                            blk_pid = (
                                blk_det.get("blockingPlayerId")
                                or blk_det.get("blockedByPlayerId")
                                or blk_det.get("blockerPlayerId")
                                or blk_det.get("blockerId")
                            )
                            defender_side = None
                            try:
                                blk_pid = int(blk_pid) if blk_pid is not None else None
                            except Exception:
                                blk_pid = None
                            if blk_pid is not None:
                                if blk_pid in pre_home:
                                    defender_side = "home"
                                elif blk_pid in pre_away:
                                    defender_side = "away"
                            if defender_side is None:
                                defender_side = ev_opp  # fallback: defender is opposite of shooter
                            attacker_side = "home" if defender_side == "away" else "away"
                            if defender_side == "home" and p in pre_home:
                                Y["BF"] += 1
                            if defender_side == "away" and p in pre_away:
                                Y["BF"] += 1
                            if attacker_side == "home" and p in pre_home:
                                Y["BA"] += 1
                            if attacker_side == "away" and p in pre_away:
                                Y["BA"] += 1
                        if et in GOAL_TYPES:
                            # GF/GA using the same corrected sets already used for AF/SF/xg
                            if p in on_for:
                                Y["GF"] += 1
                            elif p in on_opp:
                                Y["GA"] += 1

                # Track corrected on-ice for a goal at end second (for teammate co-pres later)
                end_goal_on_for_ids: set = set()

                # include end-second credited events only if they occur BEFORE the end boundary (by sortOrder)
                if second_has_credit_event(end_s):
                    # Determine end boundary; prefer same_sec_order of faceoff when available, else sortOrder
                    evs_end = sorted(orders_by_sec.get(end_s, []))
                    boundary_types = {"faceoff","stoppage","goal","goalie-stopped","penalty","timeout","challenge"}
                    end_face_sso = None
                    try:
                        face_sso_list = [sso for sso, t in sorted(sso_by_sec.get(end_s, [])) if t == "faceoff"]
                        if face_sso_list:
                            end_face_sso = min(face_sso_list)
                    except Exception:
                        end_face_sso = None
                    face_orders_end = [so for so, t in evs_end if t == "faceoff"]
                    if face_orders_end:
                        end_boundary_order = min(face_orders_end)
                    else:
                        end_boundary_orders = [so for so, t in evs_end if t in boundary_types]
                        end_boundary_order = min(end_boundary_orders) if end_boundary_orders else None
                    # Use pre-change snapshot: if there was a shift at end_s-1, prefer end_s-2
                    use_sec = end_s - 1
                    if shift_changes_by_sec.get(end_s - 1):
                        use_sec = max(0, end_s - 2)
                    if 0 <= use_sec < len(team_onice_by_sec):
                        home_ids, away_ids = team_onice_by_sec[use_sec]
                        pre_home = list(home_ids)
                        pre_away = list(away_ids)
                    else:
                        cb_end = credit_by_sec[end_s]
                        pre_home = list(cb_end.get("onice_home", ()))
                        pre_away = list(cb_end.get("onice_away", ()))
                    for ev in events_by_sec.get(end_s, []):
                        et = ev.get("type")
                        if (et not in SHOT_TYPES) and (et not in GOAL_TYPES):
                            continue
                        # Include only events before the faceoff boundary at end: primary same_sec_order, fallback sortOrder
                        try:
                            so = int(ev.get("sortOrder", 0))
                        except Exception:
                            so = 0
                        if end_face_sso is not None:
                            try:
                                sso_ev = int(ev.get("same_sec_order", 0))
                            except Exception:
                                sso_ev = None
                            if sso_ev is None or sso_ev >= end_face_sso:
                                continue
                        elif end_boundary_order is not None and so >= end_boundary_order:
                            continue
                        det = ev.get("details") or {}
                        ev_side = _owner_side_for_event(det, pre_home, pre_away)
                        if ev_side not in ("home","away"):
                            continue
                        ev_opp = "away" if ev_side == "home" else "home"
                        # Use corrected on-ice for goal events at end second; pre-change otherwise
                        if et in GOAL_TYPES:
                            corr_home, corr_away = _maybe_correct_onice_for_goal(end_s, ev, set(pre_home), set(pre_away))
                            on_for = list(corr_home) if ev_side == "home" else list(corr_away)
                            on_opp = list(corr_away) if ev_side == "home" else list(corr_home)
                        else:
                            on_for = pre_home if ev_side == "home" else pre_away
                            on_opp = pre_away if ev_side == "home" else pre_home
                        xg = float(det.get("xg", 0.0)) if det.get("xg") is not None else 0.0

                        # AF/AA from shooter; BF/BA by blocker/defender side
                        if p in on_for:
                            Y["AF"] += 1
                            if (et in SHOT_ON_GOAL_TYPES) or (et in GOAL_TYPES):
                                Y["SF"] += 1
                                xgF += xg
                        elif p in on_opp:
                            Y["AA"] += 1
                            if (et in SHOT_ON_GOAL_TYPES) or (et in GOAL_TYPES):
                                Y["SA"] += 1
                                xgA += xg
                        if et in BLOCK_TYPES:
                            blk_det = det
                            blk_pid = (
                                blk_det.get("blockingPlayerId")
                                or blk_det.get("blockedByPlayerId")
                                or blk_det.get("blockerPlayerId")
                                or blk_det.get("blockerId")
                            )
                            defender_side = None
                            try:
                                blk_pid = int(blk_pid) if blk_pid is not None else None
                            except Exception:
                                blk_pid = None
                            if blk_pid is not None:
                                if blk_pid in pre_home:
                                    defender_side = "home"
                                elif blk_pid in pre_away:
                                    defender_side = "away"
                            if defender_side is None:
                                defender_side = ev_opp
                            attacker_side = "home" if defender_side == "away" else "away"
                            if defender_side == "home" and p in pre_home:
                                Y["BF"] += 1
                            if defender_side == "away" and p in pre_away:
                                Y["BF"] += 1
                            if attacker_side == "home" and p in pre_home:
                                Y["BA"] += 1
                            if attacker_side == "away" and p in pre_away:
                                Y["BA"] += 1
                        if et in GOAL_TYPES:
                            # GF/GA consistent with the same corrected sets
                            # also capture corrected team on-ice for teammate co-presence update later
                            try:
                                end_goal_on_for_ids = set(on_for)
                            except Exception:
                                end_goal_on_for_ids = set()
                            if p in on_for:
                                Y["GF"] += 1
                            elif p in on_opp:
                                Y["GA"] += 1

                # GF/GA are now credited strictly event-driven above (per on-ice membership)

                # chemistry copresence (adaptive, no event override)
                overlap_with: Counter[int] = Counter()
                overlap_vs:  Counter[int] = Counter()
                event_copres_with: Dict[int, Counter] = defaultdict(Counter)

                for s in range(start_s, end_s):
                    on = onice_map.get(s, ())
                    if p not in on: continue
                    for q in on:
                        if q == p: continue
                        overlap_with[q] += 1
                        if credit_by_sec.get(s):
                            event_copres_with[q]["any"] += 1
                            for k in ("SF","SA","GF","GA"):
                                if credit_by_sec[s][side].get(k,0):
                                    event_copres_with[q][k] += int(credit_by_sec[s][side][k])
                    for k in opp_map.get(s, ()):
                        overlap_vs[k] += 1

                def build_weighted_list(
                    overlap: Counter[int],
                    seconds_i: int,
                    event_copres_map: Dict[int, Counter],
                ) -> Tuple[List[int], List[float], List[int], Dict[int,Counter], List[float]]:
                    ids, ws, secs = [], [], []
                    if seconds_i <= 0:
                        return ids, ws, secs, {}, []
                    if seconds_i < SHORT_WIN_SEC:
                        share_min = SHORT_SHARE_MIN
                        sec_floor_eff = max(2, int(SHORT_SEC_FRAC * seconds_i))
                    else:
                        share_min = RAW_SHARE_MIN
                        sec_floor_eff = SEC_FLOOR_BASE
                    kept: List[Tuple[int, int, float]] = []
                    for q, ov_sec in overlap.items():
                        raw_share = ov_sec / float(seconds_i)
                        if (ov_sec >= sec_floor_eff) and (raw_share >= share_min):
                            kept.append((q, ov_sec, raw_share))
                    if not kept and seconds_i < SHORT_WIN_SEC and seconds_i >= 2 and TOPK_FALLBACK > 0 and overlap:
                        qbest, ovbest = max(overlap.items(), key=lambda kv: (kv[1], kv[0]))
                        if ovbest >= 2:
                            kept = [(qbest, ovbest, ovbest / float(seconds_i))]
                    if not kept:
                        return ids, ws, secs, {}, []
                    raw_shares = []
                    w_tmp = []
                    for q, ov_sec, raw_share in kept:
                        ids.append(q)
                        secs.append(int(ov_sec))
                        raw_shares.append(float(raw_share))
                        w_tmp.append(raw_share)
                    ssum = sum(w_tmp)
                    ws = [w/ssum for w in w_tmp] if ssum > 0 else w_tmp
                    return ids, ws, secs, {q: event_copres_map.get(q, Counter()) for q, *_ in kept}, raw_shares

                ids_with, w_with, sec_with, ev_with, share_with = build_weighted_list(
                    overlap_with, seconds_i, event_copres_with
                )
                # If a corrected end-second goal occurred for this side, bump with_event_GF for those corrected ids
                if end_goal_on_for_ids:
                    try:
                        for q in ids_with:
                            if q in end_goal_on_for_ids:
                                ev_with.setdefault(q, Counter())
                                ev_with[q]["GF"] = max(1, int(ev_with[q].get("GF", 0)))
                    except Exception:
                        pass
                ids_vs,   w_vs,   sec_vs,   _,       share_vs   = build_weighted_list(
                    overlap_vs, seconds_i, {}
                )

                # Weighted mean of opponent 5v5 usage percentile within their role
                opp_pct_map = away_pct_role_5v5 if side == "home" else home_pct_role_5v5
                if ids_vs and w_vs:
                    mq, wsum = 0.0, 0.0
                    for q, wq in zip(ids_vs, w_vs):
                        mq += opp_pct_map.get(q, DEFAULT_PCT) * float(wq)
                        wsum += float(wq)
                    matchup_quality_pct = (mq / wsum) if wsum > 0 else None
                else:
                    matchup_quality_pct = None

                # elapsed-in-window shares (over the player's own seconds)
                elap = {"elapsed_share_0_5":0,"elapsed_share_6_20":0,"elapsed_share_21_60":0,"elapsed_share_61p":0}
                if seconds_i > 0:
                    for s in range(start_s, end_s):
                        if p not in onice_map.get(s, ()): continue
                        t = s - start_s
                        if   t <= 5:  elap["elapsed_share_0_5"]  += 1
                        elif t <= 20: elap["elapsed_share_6_20"] += 1
                        elif t <= 60: elap["elapsed_share_21_60"]+= 1
                        else:         elap["elapsed_share_61p"]  += 1
                    for k in elap: elap[k] = elap[k] / seconds_i

                tsf = tsf_bucket_shares(start_s, end_s)

                # --- NEW: shift metrics for this player ---
                elapsed_at_start = onice_elapsed_before_second(p, start_s, onice_map)
                shifts_in_win    = shift_count_in_window(p, start_s, end_s, onice_map)

                # Anchor at player's first ON second in the window (or start_sec if already on)
                anchor_s = start_s if p in onice_map.get(start_s, ()) else None
                if anchor_s is None:
                    for s_ in range(start_s, end_s):
                        if p in onice_map.get(s_, ()):  # first on in window
                            anchor_s = s_
                            break
                # Determine last ON second within the window
                last_on_s = end_s - 1
                while last_on_s >= start_s and p not in onice_map.get(last_on_s, ()):  # walk back
                    last_on_s -= 1
                # Entry/exit context relative to window bounds
                entry_offset_s = int(anchor_s - start_s) if anchor_s is not None else 0
                entered_after_start = 1 if (anchor_s is not None and entry_offset_s > 0) else 0
                exit_offset_s = int((end_s - 1) - last_on_s) if last_on_s >= start_s else 0
                exited_before_end = 1 if (last_on_s >= start_s and exit_offset_s > 0) else 0
                # Compute previous shift history relative to anchor_s
                last_len_s = float('nan')
                time_since_last_shift_s = float('nan')
                last_shift_missing = 1
                if anchor_s is not None:
                    last_len_raw, rest_gap_raw_s, prev_end_sec = last_shift_and_rest_before(p, anchor_s, onice_map)
                    if prev_end_sec < 0:
                        # First appearance this game: leave NaNs and masks = 1
                        last_shift_missing = 1
                        last_len_s = float('nan')
                        time_since_last_shift_s = float('nan')
                    else:
                        last_shift_missing = 0
                        last_len_s = float(int(last_len_raw))
                        prev_period = period_of(prev_end_sec)
                        curr_period = period_of(anchor_s)
                        intermission_added_s = 0
                        if curr_period > prev_period:
                            intermission_added_s = 120 if curr_period == 4 else 900
                        # TV-timeout bonus applies only within the same period; do not add across intermission
                        tv_timeout_added_s = 0
                        if curr_period == prev_period:
                            try:
                                for ss in range(prev_end_sec + 1, anchor_s + 1):
                                    if ss in tv_timeout_secs:
                                        tv_timeout_added_s = 90
                                        break
                            except Exception:
                                tv_timeout_added_s = 0
                        time_since_last_shift_s = float(int(rest_gap_raw_s + intermission_added_s + tv_timeout_added_s))

                # Stint runs inside the window: contiguous ON segments lengths
                stint_lengths: List[int] = []
                _run = 0
                for s_ in range(start_s, end_s):
                    if p in onice_map.get(s_, ()):  # on
                        _run += 1
                    else:
                        if _run > 0:
                            stint_lengths.append(_run)
                            _run = 0
                if _run > 0:
                    stint_lengths.append(_run)
                if stint_lengths:
                    stint_duration_max = int(max(stint_lengths))
                    if len(stint_lengths) == 1:
                        stint_duration_std = 0.0
                    else:
                        _n = len(stint_lengths)
                        _mean = sum(stint_lengths) / _n
                        _var = sum((x - _mean) * (x - _mean) for x in stint_lengths) / _n
                        stint_duration_std = float(math.sqrt(_var))
                else:
                    stint_duration_max = 0
                    stint_duration_std = 0.0

                # compute after-icing flag for player rows (team-relative)
                ai_flag_player = after_icing_for_side(
                    w.get("start_prev_break_type"),
                    w.get("start_prev_break_team_id"),
                    side, home_team_id, away_team_id, w.get("fo_zone")
                )

                # faceoff winner flag at window start for this player
                fo_seen_start_flag = (w.get("fo_won_player_id") is not None)
                try:
                    is_faceoff_winner_start = bool(fo_seen_start_flag and w.get("fo_won_player_id") is not None and int(w.get("fo_won_player_id")) == int(p))
                except Exception:
                    is_faceoff_winner_start = False
                try:
                    is_faceoff_loser_start = bool(fo_seen_start_flag and w.get("fo_lost_player_id") is not None and int(w.get("fo_lost_player_id")) == int(p))
                except Exception:
                    is_faceoff_loser_start = False

                # Home last-change opportunity (team-side property, stamped on player rows for convenience)
                stoppage_types = {
                    'goal','penalty','icing','offside','stoppage',
                    'puck-out-of-play','goalie-stopped','timeout','challenge','period_start'
                }
                spbt_p = (w.get("start_prev_break_type") or "").lower()
                fo_seen_start_team = (w.get("fo_won_team_id") is not None)
                is_home_side_player = (side == "home")
                fo_expected_player = bool(fo_seen_start_team) or (spbt_p in stoppage_types)
                excluded_player = spbt_p in {"strength","hard_cap","flow",""}
                home_last_change_opportunity_flag = bool(is_home_side_player and fo_expected_player and (not excluded_player))
                if str(zone_start_for(side, w["fo_zone"], w.get("fo_won_team_id"), home_team_id, away_team_id)).lower() == "flow":
                    home_last_change_opportunity_flag = False
                # Home iced → no change allowed
                try:
                    prev_tid_p = w.get("start_prev_break_team_id")
                    if spbt_p == "icing" and is_home_side_player and prev_tid_p is not None and home_team_id is not None and int(prev_tid_p) == int(home_team_id):
                        home_last_change_opportunity_flag = False
                except Exception:
                    pass

                # player-side zone and after-icing flags
                _zone_rel_p = zone_start_for(side, w["fo_zone"], w.get("fo_won_team_id"), home_team_id, away_team_id)
                _ai_flag_player_int = int(ai_flag_player)
                _ai_by_team_p = int(_ai_flag_player_int == 1 and _zone_rel_p == "DZ")
                _ai_by_opp_p  = int(_ai_flag_player_int == 1 and _zone_rel_p == "OZ")

                # Safe position mapping (boxscore may be missing in some games)
                _pos_code = (player_pos_map.get(p) if isinstance(player_pos_map, dict) else None)
                _pos_upper = str(_pos_code).upper() if _pos_code is not None else ""
                _pos_simple = ("F" if _pos_upper in {"C","L","R"} else ("D" if _pos_upper == "D" else None))

                row = {
                    "window_id": w["window_id"],
                    "team_side": side,
                    "period": w["period"],
                    "start_sec": start_s,
                    "end_sec": end_s,
                    "duration": dur,
                    "clock_start": w["clock_start"],
                    "end_event_type": w["end_event_type"],
                    "strength_global": ("PP" if w["strength_global"] == "PP_home" and side == "home" else (
                                           "PP" if w["strength_global"] == "PP_away" and side == "away" else (
                                           "PK" if w["strength_global"] == "PP_home" and side == "away" else (
                                           "PK" if w["strength_global"] == "PP_away" and side == "home" else w["strength_global"])))),
                    "fo_zone": _zone_rel_p,
                    "zone_start": _zone_rel_p,
                    "start_prev_break_type": w.get("start_prev_break_type"),
                    "start_prev_break_subtype": w.get("start_prev_break_subtype"),
                    "media_timeout_start": int(bool(w.get("media_timeout_start"))),
                    "delayed_penalty": int(bool(w.get("delayed_penalty"))),
                    "after_icing": _ai_flag_player_int,
                    # Team-relative icing source flags via zone when after-icing
                    "after_icing_by_team": _ai_by_team_p,
                    "after_icing_by_opponent": _ai_by_opp_p,
                    "is_faceoff_winner_start": is_faceoff_winner_start,
                    "is_faceoff_loser_start": is_faceoff_loser_start,
                    "home_away": (side == "home"),
                    "long_change": int(is_long_change(start_s)),
                    "home_last_change_opportunity": int(home_last_change_opportunity_flag),
                    # Faceoff taker markers at window start (player-centric)
                    "fo_took_start": int(1 if (fo_seen_start and p == (w.get("fo_center_start_id") or -1)) else 0),
                    "fo_won_taken_start": (1 if (fo_seen_start and p == (w.get("fo_center_start_id") or -1) and w.get("fo_won_start")) else (0 if (fo_seen_start and p == (w.get("fo_center_start_id") or -1) and w.get("fo_lost_start")) else None)),
                    "fo_lost_taken_start": (1 if (fo_seen_start and p == (w.get("fo_center_start_id") or -1) and w.get("fo_lost_start")) else (0 if (fo_seen_start and p == (w.get("fo_center_start_id") or -1) and w.get("fo_won_start")) else None)),
                    "playerId": p,
                    "positionCode": _pos_code,
                    "position": _pos_simple,
                    "playerName": (player_name_map.get(p) if isinstance(player_name_map, dict) else None),
                    # exposures
                    "seconds": seconds_i,
                    "pp_seconds": pp_sec_i,
                    "pk_seconds": pk_sec_i,
                    # new: roster context at window start
                    "us_skaters_start": int(len(w[f"{side}_ids_start"])),
                    "them_skaters_start": int(len(w[f"{opp}_ids_start"])),
                    "opponent_goalie_id_start": int((w.get("goalie_ids_start", {}) or {}).get(opp, 0) or 0),
                    # optional shares for treatments/controls
                    "pp_share": (pp_sec_i / seconds_i) if seconds_i > 0 else 0.0,
                    "pk_share": (pk_sec_i / seconds_i) if seconds_i > 0 else 0.0,
                    "offset_log_toi": math.log(max(1, seconds_i)),
                    "offset_log_pp":  math.log(max(1, pp_sec_i)),
                    "offset_log_pk":  math.log(max(1, pk_sec_i)),
                    # NEW: shift metrics
                    "onice_elapsed_at_window_start": elapsed_at_start,
                    "shift_count_in_window": shifts_in_win,
                    "last_shift_len_s": (int(last_len_s) if not math.isnan(last_len_s) else float('nan')),
                    "time_since_last_shift_s": (int(time_since_last_shift_s) if not math.isnan(time_since_last_shift_s) else float('nan')),
                    "last_shift_len_missing": int(1 if math.isnan(last_len_s) else 0),
                    "time_since_last_shift_missing": int(1 if math.isnan(time_since_last_shift_s) else 0),
                    "entered_after_start": int(entered_after_start),
                    "entry_offset_s": int(entry_offset_s),
                    "exited_before_end": int(exited_before_end),
                    "exit_offset_s": int(exit_offset_s),
                    "stint_duration_max": int(stint_duration_max),
                    "stint_duration_std": float(stint_duration_std),
                    "stint_duration_st": float(stint_duration_std),
                    # TRAIN–X controls
                    "score_diff_start": score_diff_for_side(side, start_s),
                    "score_state_start": _score_state_bucket(score_diff_for_side(side, start_s)),
                    "clock_s": clock_s_at(start_s),
                    "matchup_quality_pct": (float(matchup_quality_pct) if matchup_quality_pct is not None else None),
                    # outcomes
                    "xGF": float(xgF), "xGA": float(xgA),
                    "GF": int(Y["GF"]), "GA": int(Y["GA"]),
                    "SF": int(Y["SF"]), "SA": int(Y["SA"]),
                    "AF": int(Y["AF"]), "AA": int(Y["AA"]),
                    "BF": int(Y["BF"]), "BA": int(Y["BA"]),
                    # TSF + elapsed
                    **tsf,
                    **elap,
                    # chemistry rep
                    "teammates_onice_ids_start": sorted(list(ids_start)),
                    "opponents_onice_ids_start": sorted(list(opp_start)),

                    "teammates_onice_ids_w": ids_with,
                    "teammates_onice_w": [float(x) for x in w_with],
                    "teammates_onice_sec_w": [int(x) for x in sec_with],
                    "teammates_onice_share_raw": [float(x) for x in share_with],

                    "opponents_onice_ids_w": ids_vs,
                    "opponents_onice_w": [float(x) for x in w_vs],
                    "opponents_onice_sec_w": [int(x) for x in sec_vs],
                    "opponents_onice_share_raw": [float(x) for x in share_vs],

                    # event co-presence (aligned to teammates list)
                    "with_event_GF": [int(ev_with.get(q,{}).get("GF",0)) for q in ids_with],
                    "with_event_GA": [int(ev_with.get(q,{}).get("GA",0)) for q in ids_with],
                    "with_event_SF": [int(ev_with.get(q,{}).get("SF",0)) for q in ids_with],
                    "with_event_SA": [int(ev_with.get(q,{}).get("SA",0)) for q in ids_with],
                    # flags
                    "goalie_pulled_since": 0,
                }
                # Personal puck-management: giveaways committed and takeaways forced within the window
                gv_p, tk_p = 0, 0
                hp_p, bp_p = 0, 0
                for s in range(start_s, end_s):
                    for ev in events_by_sec.get(s, []):
                        et = str(ev.get("type", "")).lower()
                        det = ev.get("details") or {}

                        # giveaways / takeaways credited to the actor only
                        if et in ("giveaway", "takeaway"):
                            pid_ev = (
                                det.get("playerId")
                                or det.get("player_id")
                                or det.get("actorPlayerId")
                            )
                            try:
                                pid_ev = int(pid_ev) if pid_ev is not None else None
                            except Exception:
                                pid_ev = None
                            if pid_ev is not None and pid_ev == p:
                                if et == "giveaway":
                                    gv_p += 1
                                else:
                                    tk_p += 1

                        # hits credited to the hitter only
                        elif et == "hit":
                            hit_pid = (
                                det.get("hittingPlayerId")
                                or det.get("hitterPlayerId")
                                or det.get("hitterId")
                                or det.get("hitter")
                                or det.get("playerId")
                            )
                            try:
                                hit_pid = int(hit_pid) if hit_pid is not None else None
                            except Exception:
                                hit_pid = None
                            if hit_pid is not None and hit_pid == p:
                                hp_p += 1

                        # blocks credited to the blocker only
                        elif et == "blocked-shot":
                            blk_pid = (
                                det.get("blockingPlayerId")
                                or det.get("blockedByPlayerId")
                                or det.get("blockerPlayerId")
                                or det.get("blockerId")
                                or det.get("playerId")
                            )
                            try:
                                blk_pid = int(blk_pid) if blk_pid is not None else None
                            except Exception:
                                blk_pid = None
                            if blk_pid is not None and blk_pid == p:
                                bp_p += 1
                row["giveaways_committed"] = int(gv_p)
                row["takeaways_forced"] = int(tk_p)
                row["hits_personal"] = int(hp_p)
                row["blocks_personal"] = int(bp_p)
                # standings/b2b on player rows (cached per team)
                tid_player = home_team_id if side=="home" else away_team_id
                rank_prior_p, b2b_calc_p, rest_days_calc_p = compute_standings_and_b2b(tid_player)
                if rank_prior_p is not None:
                    row["standing_prior"] = int(rank_prior_p)
                if b2b_calc_p is not None:
                    row["b2b_team"] = int(bool(b2b_calc_p))
                if rest_days_calc_p is not None:
                    row["rest_days_team"] = int(rest_days_calc_p)
                # Faceoff flags at window start (player + team + taker)
                won_tid = w.get("fo_won_team_id")
                # Treat presence of fo_won_team_id as authoritative indicator that this window starts on a faceoff
                is_fo_start = (won_tid is not None)
                team_tid = (home_team_id if side == "home" else away_team_id)
                ids_start_set = set(w.get(f"{side}_ids_start", []) or [])
                fo_seen_start_val = int(1 if (is_fo_start and (p in ids_start_set)) else 0)
                team_won = bool(is_fo_start and (team_tid is not None) and int(won_tid) == int(team_tid))
                team_lost = bool(is_fo_start and (team_tid is not None) and int(won_tid) != int(team_tid))
                # Taker IDs from window metadata
                taker_win_pid = None
                taker_lose_pid = None
                try:
                    taker_win_pid = int(w.get("fo_won_player_id")) if w.get("fo_won_player_id") is not None else None
                except Exception:
                    taker_win_pid = None
                try:
                    taker_lose_pid = int(w.get("fo_lost_player_id")) if w.get("fo_lost_player_id") is not None else None
                except Exception:
                    taker_lose_pid = None
                took = bool(is_fo_start and (p == taker_win_pid or p == taker_lose_pid))
                took_won = bool(is_fo_start and (p == taker_win_pid))
                took_lost = bool(is_fo_start and (p == taker_lose_pid))
                row.update({
                    # Player present for FO at window start
                    "fo_seen_start": int(fo_seen_start_val),
                    # Team-level explicit flags (00/01)
                    "fo_team_won_start": int(1 if (fo_seen_start_val and team_won) else 0),
                    "fo_team_lost_start": int(1 if (fo_seen_start_val and team_lost) else 0),
                    # Back-compat names (if present elsewhere)
                    "fo_won_start": int(1 if (fo_seen_start_val and team_won) else 0),
                    "fo_lost_start": int(1 if (fo_seen_start_val and team_lost) else 0),
                    # Taker flags
                    "fo_took_start": int(1 if took else 0),
                    "fo_took_won_start": int(1 if took_won else 0),
                    "fo_took_lost_start": int(1 if took_lost else 0),
                })
                # Early mass dose: only weight early seconds if player actually took/was on for the FO
                try:
                    row["early_mass"] = float(int(fo_seen_start_int)) * float(elap.get("elapsed_share_0_5", 0.0))
                except Exception:
                    row["early_mass"] = 0.0
                # Pure pre-change instruments at window start (no realized overlap applied)
                try:
                    row["ai_OZ_start"] = int(row.get("after_icing_by_opponent", 0)) * int(fo_seen_start_int)
                except Exception:
                    row["ai_OZ_start"] = 0
                try:
                    row["ai_DZ_start"] = int(row.get("after_icing_by_team", 0)) * int(fo_seen_start_int)
                except Exception:
                    row["ai_DZ_start"] = 0
                try:
                    row["fo_O_start"] = int(_zone_rel_p == "OZ") * int(fo_seen_start_int)
                except Exception:
                    row["fo_O_start"] = 0
                try:
                    row["fo_D_start"] = int(_zone_rel_p == "DZ") * int(fo_seen_start_int)
                except Exception:
                    row["fo_D_start"] = 0
                try:
                    row["last_change_start"] = int(home_last_change_opportunity_flag) * int(fo_seen_start_int)
                except Exception:
                    row["last_change_start"] = 0
                # Force: home iced into DZ → no last change, regardless of FO participation
                try:
                    if (side == "home" and int(row.get("after_icing_by_team", 0)) == 1 and str(row.get("zone_start")) == "DZ"):
                        row["last_change_start"] = 0
                        row["home_last_change_opportunity"] = 0
                except Exception:
                    pass
                try:
                    row["long_change_start"] = int(is_long_change(start_s)) * int(fo_seen_start_int)
                except Exception:
                    row["long_change_start"] = 0
                # pulled-goalie exposure for player rows
                row.update({
                    "pulled_goalie_start": int(bool(w["goalies_start"][side] == 0)),
                    "goalie_pulled_since": int(since_pulled_home[start_s] if side=="home" else since_pulled_away[start_s]),
                })
                player_rows.append(row)

    # -------- EA/EN prune (drop shards <6s with zero credited events) --------
    def window_zero_events(start_s: int, end_s: int) -> bool:
        for s in range(start_s, end_s):
            cb = credit_by_sec.get(s)
            if not cb: continue
            if cb["home"] or cb["away"]:
                return False
        return True

    kept_team_rows = []
    for r in team_rows:
        is_ea = (r["strength_team"] in ("EA","EN_for"))
        if is_ea and (r["duration"] < EA_PRUNE_MIN) and window_zero_events(r["start_sec"], r["end_sec"]):
            continue
        kept_team_rows.append(r)
    team_rows = kept_team_rows

    kept_player_rows = []
    keep_key = {(r["window_id"], r["team_side"]): True for r in team_rows}
    for r in player_rows:
        key = (r["window_id"], r["team_side"])
        if key not in keep_key:
            continue
        kept_player_rows.append(r)
    player_rows = kept_player_rows

    return windows, team_rows, player_rows

# -------------------- Writers --------------------
def write_json_pretty(objs: List[Dict[str,Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(objs, f, ensure_ascii=False, indent=2)

def write_csv(path: str, rows: List[Dict[str,Any]], keys: List[str]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in keys}
            for k, v in list(row.items()):
                if isinstance(v, list):
                    row[k] = "|".join(str(x) for x in v)
            w.writerow(row)

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Build windows + team/player rows from pbp_onice JSON.")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to pbp_onice_<gamePk>.json")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--csv", action="store_true", help="Also write CSVs")
    ap.add_argument("--hard-cap-sec", type=int, default=0, help="Optional hard cap in seconds (0=off)")
    ap.add_argument("--home_team_id", type=int, default=None, help="Optional: home team id for directional credit")
    ap.add_argument("--away_team_id", type=int, default=None, help="Optional: away team id for directional credit")
    ap.add_argument("--players-json", type=str, default=None, help="Optional player map JSON (id->name or list of players)")
    ap.add_argument("--boxscore-json", type=str, default=None, help="Optional boxscore JSON (will extract playerId->name)")
    ap.add_argument("--standings_dir", type=str, default=None, help="Directory with standings CSVs (game_results_*.csv, standings_by_date_*.csv)")
    ap.add_argument("--debug-ga", action="store_true", help="Print detailed reasoning whenever a player is assigned GA")
    ap.add_argument("--debug-standings", action="store_true", help="Debug prints for standings/b2b derivation")
    ap.add_argument("--csv_only_player_train", action="store_true", help="When --csv is set, only write player_windows_train_<gamePk>.csv")
    args = ap.parse_args()
    # seed globals for helper discovery
    global CLI_IN_PATH, CLI_STANDINGS_DIR
    CLI_IN_PATH = args.in_path
    CLI_STANDINGS_DIR = args.standings_dir
    # helper: discover boxscore files across any dumpsN layouts
    def _boxscore_candidates(game_id: str) -> List[str]:
        pats = [
            os.path.join("artifacts", "dumps*", "raw", "boxscore", f"{game_id}.json"),
            os.path.join("artifacts", "dumps_rawonly", "raw", f"boxscore_{game_id}.json"),
        ]
        paths: List[str] = []
        for pat in pats:
            try:
                for p in glob.glob(pat):
                    if os.path.exists(p):
                        paths.append(p)
            except Exception:
                pass
        # add explicit known locations for precedence/back-compat
        for p in (
            os.path.join("artifacts", "dumps",  "raw", "boxscore", f"{game_id}.json"),
            os.path.join("artifacts", "dumps2", "raw", "boxscore", f"{game_id}.json"),
        ):
            if os.path.exists(p):
                paths.append(p)
        # also try next to the input pbp file, e.g. API/Final/<year>/raw/boxscore/<gamePk>.json
        try:
            in_dir = os.path.abspath(os.path.dirname(args.in_path))
            raw_dir = os.path.dirname(in_dir)
            local_box = os.path.join(raw_dir, "boxscore", f"{game_id}.json")
            if os.path.exists(local_box):
                paths.append(local_box)
            local_alt = os.path.join(raw_dir, f"boxscore_{game_id}.json")
            if os.path.exists(local_alt):
                paths.append(local_alt)
        except Exception:
            pass
        # de-dup in order
        seen = set(); uniq: List[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq

    ensure_dir(args.out_dir)
    data = load_json(args.in_path)
    gamePk = data.get("gamePk") or "unknown"
    evts   = data.get("events") or []
    if not evts:
        raise SystemExit("No events in input.")

    root_home = data.get("home_team_id")
    root_away = data.get("away_team_id")
    home_id = args.home_team_id or root_home
    away_id = args.away_team_id or root_away

    # Fallback: resolve team IDs from boxscore when missing
    if (home_id is None or away_id is None) and gamePk:
        for rel in _boxscore_candidates(str(gamePk)):
            if os.path.exists(rel):
                try:
                    box_try = load_json(rel)
                    h_obj = (box_try.get("homeTeam") or box_try.get("home") or {})
                    a_obj = (box_try.get("awayTeam") or box_try.get("away") or {})
                    hid = h_obj.get("id") or (h_obj.get("team") or {}).get("id")
                    aid = a_obj.get("id") or (a_obj.get("team") or {}).get("id")
                    if hid is not None and home_id is None:
                        home_id = int(hid)
                    if aid is not None and away_id is None:
                        away_id = int(aid)
                    break
                except Exception:
                    pass
    if args.debug_standings:
        print({"debug":"standings","cli_team_ids": {"home_team_id": home_id, "away_team_id": away_id}})

    name_map: Optional[Dict[int, str]] = None
    pos_map: Optional[Dict[int, str]] = None
    def _extract_name_map(obj: Any) -> Optional[Dict[int, str]]:
        try:
            if isinstance(obj, dict) and "playerByGameStats" in obj:
                out: Dict[int, str] = {}
                pgs = obj.get("playerByGameStats") or {}
                for side in ("homeTeam", "awayTeam"):
                    side_obj = pgs.get(side) or {}
                    for group in ("forwards", "defense", "goalies"):
                        for p in side_obj.get(group) or []:
                            try:
                                pid = int(p.get("playerId")) if p.get("playerId") is not None else None
                            except Exception:
                                pid = None
                            if pid is None:
                                continue
                            nm = p.get("name")
                            name = None
                            if isinstance(nm, dict):
                                name = nm.get("default") or nm.get("fullName") or nm.get("lastFirstName")
                            elif isinstance(nm, str):
                                name = nm
                            if not name:
                                name = p.get("fullName") or p.get("playerName")
                            if name:
                                out[pid] = str(name)
                return out or None
            if isinstance(obj, dict):
                if "players" in obj and isinstance(obj["players"], list):
                    return _extract_name_map(obj["players"]) or {}
                if all(isinstance(k, (str,int)) for k in obj.keys()):
                    out = {}
                    for k, v in obj.items():
                        try:
                            pid = int(k); name = str(v)
                            out[pid] = name
                        except Exception:
                            continue
                    return out
            if isinstance(obj, list):
                out = {}
                for p in obj:
                    if not isinstance(p, dict): continue
                    pid = p.get("playerId") or p.get("id") or p.get("player_id")
                    if pid is None: continue
                    try: pid = int(pid)
                    except Exception: continue
                    name = (
                        p.get("fullName")
                        or p.get("name")
                        or (p.get("firstName") and p.get("lastName") and f"{p.get('firstName')} {p.get('lastName')}")
                        or None
                    )
                    if name is None:
                        fn = p.get("firstName") or (isinstance(p.get("firstName"), dict) and p.get("firstName").get("default"))
                        ln = p.get("lastName") or (isinstance(p.get("lastName"), dict) and p.get("lastName").get("default"))
                        if fn or ln:
                            name = f"{fn or ''} {ln or ''}".strip()
                    if name: out[pid] = str(name)
                return out
        except Exception:
            return None
        return None

    def _extract_pos_map(obj: Any) -> Optional[Dict[int, str]]:
        try:
            # raw PBP style: { players: [ { playerId, positionCode, teamId, ... } ] }
            if isinstance(obj, dict) and isinstance(obj.get("players"), list):
                out: Dict[int, str] = {}
                for p in obj["players"]:
                    if not isinstance(p, dict):
                        continue
                    pid = p.get("playerId") or p.get("id") or p.get("player_id")
                    pos = p.get("positionCode") or p.get("pos") or p.get("position")
                    if pid is None or pos is None:
                        continue
                    try:
                        pid = int(pid)
                    except Exception:
                        continue
                    out[pid] = str(pos if not isinstance(pos, dict) else pos.get("code"))
                return out or None
            # list of player dicts
            if isinstance(obj, list):
                out: Dict[int, str] = {}
                for p in obj:
                    if not isinstance(p, dict):
                        continue
                    pid = p.get("playerId") or p.get("id") or p.get("player_id")
                    pos = p.get("positionCode") or p.get("pos") or p.get("position")
                    if pid is None or pos is None:
                        continue
                    try:
                        pid = int(pid)
                    except Exception:
                        continue
                    out[pid] = str(pos if not isinstance(pos, dict) else pos.get("code"))
                return out or None
            # boxscore style: playerByGameStats
            if isinstance(obj, dict) and "playerByGameStats" in obj:
                out: Dict[int, str] = {}
                pgs = obj.get("playerByGameStats") or {}
                for side in ("homeTeam","awayTeam"):
                    for group in ("forwards","defense","goalies"):
                        for p in (pgs.get(side) or {}).get(group) or []:
                            try:
                                pid = int(p.get("playerId")) if p.get("playerId") is not None else None
                            except Exception:
                                pid = None
                            if pid is None:
                                continue
                            # handle position as code string or nested code
                            pos_raw = p.get("positionCode") or p.get("pos") or p.get("position")
                            if isinstance(pos_raw, dict):
                                pos = pos_raw.get("code")
                            else:
                                pos = pos_raw
                            if pos:
                                out[pid] = str(pos)
                return out or None
        except Exception:
            return None
        return None

    if args.players_json and os.path.exists(args.players_json):
        try:
            name_src = load_json(args.players_json)
            name_map = _extract_name_map(name_src)
        except Exception:
            name_map = None
    # If not provided or failed, try explicit boxscore
    if not name_map and args.boxscore_json and os.path.exists(args.boxscore_json):
        try:
            name_src = load_json(args.boxscore_json)
            name_map = _extract_name_map(name_src)
        except Exception:
            name_map = None
    if not name_map:
        # try embedded (best-effort)
        name_map = _extract_name_map(data.get("players") or {}) or None
    if not name_map and gamePk:
        # heuristic defaults across any dumps*
        for rel in _boxscore_candidates(str(gamePk)):
            if os.path.exists(rel):
                try:
                    name_src = load_json(rel)
                    name_map = _extract_name_map(name_src)
                except Exception:
                    name_map = None
                if name_map:
                    break

    # Position map: use boxscore only (explicit arg or auto from artifacts/dumps/raw/boxscore/<gamePk>.json)
    if args.boxscore_json and os.path.exists(args.boxscore_json):
        try:
            pos_src = load_json(args.boxscore_json)
            pos_map = _extract_pos_map(pos_src)
        except Exception:
            pos_map = None
    if not pos_map and gamePk:
        for rel in _boxscore_candidates(str(gamePk)):
            if os.path.exists(rel):
                try:
                    pos_src = load_json(rel)
                    pos_map = _extract_pos_map(pos_src)
                except Exception:
                    pos_map = None
                if pos_map:
                    break

    # Final fallback: extract positions from the pbp_onice payload if available
    if not pos_map:
        try:
            pos_map = _extract_pos_map(data.get("players") or {}) or None
        except Exception:
            pos_map = None

    # try to derive numeric game_pk for standings matching
    game_pk_int: Optional[int] = None
    try:
        if isinstance(data.get("gamePk"), int):
            game_pk_int = int(data.get("gamePk"))
        else:
            m = re.search(r"(\d{10})", os.path.basename(args.in_path))
            if m:
                game_pk_int = int(m.group(1))
    except Exception:
        game_pk_int = None

    windows, team_rows, player_rows = build_windows(
        evts,
        hard_cap_sec=args.hard_cap_sec,
        home_team_id=home_id,
        away_team_id=away_id,
        player_name_map=name_map,
        player_pos_map=pos_map,
        debug_ga=args.debug_ga,
        debug_standings=args.debug_standings,
        game_pk=game_pk_int,
    )

    # Resolve game date from boxscore
    boxscore_date = None
    boxscore_rinkid = None
    if gamePk:
        for rel in _boxscore_candidates(str(gamePk)):
            if os.path.exists(rel):
                try:
                    j = load_json(rel)
                    cand = j.get("gameDate") or j.get("date") or j.get("game_date")
                    if not cand:
                        gd = j.get("gameData") or {}
                        dt = gd.get("datetime") or {}
                        cand = dt.get("dateTime")
                    if isinstance(cand, str) and len(cand) >= 10:
                        boxscore_date = cand[:10]
                    # rink id/name
                    venue_obj = j.get("venue") if isinstance(j.get("venue"), dict) else None
                    if venue_obj is not None:
                        rink_cand = venue_obj.get("id") or venue_obj.get("default") or venue_obj.get("name")
                    else:
                        rink_cand = None
                    if not rink_cand:
                        gd = j.get("gameData") or {}
                        v2 = gd.get("venue") if isinstance(gd.get("venue"), dict) else {}
                        rink_cand = v2.get("id") or v2.get("name")
                    if not rink_cand:
                        gi = j.get("gameInfo") or {}
                        v3 = gi.get("venue") if isinstance(gi.get("venue"), dict) else {}
                        rink_cand = v3.get("id") or v3.get("name")
                    if rink_cand is not None:
                        try:
                            boxscore_rinkid = int(rink_cand)
                        except Exception:
                            boxscore_rinkid = str(rink_cand)
                except Exception:
                    pass
                # We found a boxscore file; stop after first found
                break

    # Stamp gameid/teamid/date/rinkid into JSON outputs (without mutating originals)
    windows_json = [dict(r, gameid=gamePk, teamid=None, date=boxscore_date, rinkid=boxscore_rinkid) for r in windows]
    team_rows_json = []
    for r in team_rows:
        side = r.get("team_side")
        tid = (home_id if side == "home" else (away_id if side == "away" else None))
        team_rows_json.append(dict(r, gameid=gamePk, teamid=tid, date=boxscore_date, rinkid=boxscore_rinkid))
    player_rows_json = []
    for r in player_rows:
        side = r.get("team_side")
        tid = (home_id if side == "home" else (away_id if side == "away" else None))
        player_rows_json.append(dict(r, gameid=gamePk, teamid=tid, date=boxscore_date, rinkid=boxscore_rinkid))

    # JSON outputs (skip if user only wants player_train CSVs)
    if not getattr(args, "csv_only_player_train", False):
        jw = os.path.join(args.out_dir, f"windows_{gamePk}.json")
        jt = os.path.join(args.out_dir, f"team_windows_{gamePk}.json")
        jp = os.path.join(args.out_dir, f"player_windows_{gamePk}.json")
        write_json_pretty(windows_json, jw)
        write_json_pretty(team_rows_json, jt)
        write_json_pretty(player_rows_json, jp)
        print(f"Wrote {jw} ({len(windows)} windows)")
        print(f"Wrote {jt} ({len(team_rows)} team-rows; {len(team_rows)//2} windows × 2 teams minus EA prunes)")
        print(f"Wrote {jp} ({len(player_rows)} player-rows)")

    if args.csv:
        # Stamp gameid/teamid/date/rinkid for CSV outputs
        windows_csv_rows = [dict(r, gameid=gamePk, teamid=None, date=boxscore_date, rinkid=boxscore_rinkid) for r in windows]
        team_rows_csv_rows = []
        for r in team_rows:
            side = r.get("team_side")
            tid = (home_id if side == "home" else (away_id if side == "away" else None))
            team_rows_csv_rows.append(dict(r, gameid=gamePk, teamid=tid, date=boxscore_date, rinkid=boxscore_rinkid))
        player_rows_csv_rows = []
        for r in player_rows:
            side = r.get("team_side")
            tid = (home_id if side == "home" else (away_id if side == "away" else None))
            player_rows_csv_rows.append(dict(r, gameid=gamePk, teamid=tid, date=boxscore_date, rinkid=boxscore_rinkid))
        if not args.csv_only_player_train:
            write_csv(
            os.path.join(args.out_dir, f"windows_{gamePk}.csv"),
            windows_csv_rows,
            ["window_id","period","start_sec","end_sec","duration","clock_start",
             "strength_global","fo_zone","fo_won_team_id","end_event_type","delayed_penalty",
             "home_ids_start","away_ids_start","home_ids_end","away_ids_end","gameid","teamid","date","rinkid"]
        )
        write_csv(
            os.path.join(args.out_dir, f"team_windows_{gamePk}.csv"),
            team_rows_csv_rows,
            ["window_id","team_side","period","start_sec","end_sec","duration","clock_start",
                        "end_event_type","strength_global","strength_team","strength_team_label",
             "skaters_for","skaters_against","goalie_for","goalie_against",
                        "fo_zone","zone_start","fo_won_team_id","start_prev_break_type","start_prev_break_subtype","media_timeout_start","after_icing","after_icing_by_team","after_icing_by_opponent","home_away","long_change",
             "score_diff_start","clock_s",
             "pulled_goalie_start","goalie_pulled_since",
             "fo_seen_start","fo_won_start","fo_lost_start","home_last_change_opportunity",
             "rest_days_team","b2b_team",
             "standing_prior",
             "team_ids_start","opp_ids_start","team_ids_end","opp_ids_end","gameid","teamid","date","rinkid"]
        )
        write_csv(
            os.path.join(args.out_dir, f"player_windows_{gamePk}.csv"),
            player_rows_csv_rows,
                    [
                        "window_id","team_side","period","start_sec","end_sec","duration","clock_start",
                        "end_event_type","strength_global","fo_zone","zone_start","start_prev_break_type","start_prev_break_subtype","media_timeout_start","delayed_penalty","after_icing","after_icing_by_team","after_icing_by_opponent","home_away","long_change","playerId","positionCode","position","playerName",
             "seconds","pp_seconds","pk_seconds","pp_share","pk_share","offset_log_toi","offset_log_pp","offset_log_pk",
                        "us_skaters_start","them_skaters_start","opponent_goalie_id_start",
             "onice_elapsed_at_window_start","shift_count_in_window",
                        "last_shift_len_s","time_since_last_shift_s","last_shift_len_missing","time_since_last_shift_missing",
                        "entered_after_start","entry_offset_s","exited_before_end","exit_offset_s",
                        "stint_duration_max","stint_duration_st",
             "score_diff_start","score_state_start","clock_s",
             "matchup_quality_pct",
                        "fo_seen_start","fo_team_won_start","fo_team_lost_start","fo_won_start","fo_lost_start","fo_took_start","fo_took_won_start","fo_took_lost_start","home_last_change_opportunity",
             "rest_days_team","b2b_team",
             "standing_prior",
             "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
             "tsf_0_5","tsf_6_20","tsf_21_60","tsf_61p",
             "elapsed_share_0_5","elapsed_share_6_20","elapsed_share_21_60","elapsed_share_61p",
                        "early_mass",
                        "ai_OZ_start","ai_DZ_start","fo_O_start","fo_D_start","last_change_start","long_change_start",
             "teammates_onice_ids_start","opponents_onice_ids_start",
             "teammates_onice_ids_w","teammates_onice_w","teammates_onice_sec_w",
             "opponents_onice_ids_w","opponents_onice_w","opponents_onice_sec_w",
             "with_event_GF","with_event_GA","with_event_SF","with_event_SA",
                        "giveaways_committed","takeaways_forced","hits_personal","blocks_personal",
                        "gameid","teamid","date","rinkid"]
                    )
        # Also write a train-filtered player windows CSV with only PP/PK/5v5
        player_rows_train_csv_rows = [
            r for r in player_rows_csv_rows
            if str(r.get("strength_global")) in {"5v5","PP","PK"}
            and (int(r.get("period", 0)) in (1, 2, 3))
        ]
        # Canonicalize keys in output header: use gamePk/teamId only
        for _r in player_rows_train_csv_rows:
            if "gamePk" not in _r and "gameid" in _r:
                _r["gamePk"] = _r.get("gameid")
            if "teamId" not in _r and "teamid" in _r:
                _r["teamId"] = _r.get("teamid")
            _r.pop("gameid", None)
            _r.pop("teamid", None)
        write_csv(
            os.path.join(args.out_dir, f"player_windows_train_{gamePk}.csv"),
            player_rows_train_csv_rows,
            ["window_id","team_side","period","start_sec","end_sec","duration","clock_start",
            "end_event_type","strength_global","fo_zone","zone_start","start_prev_break_type","start_prev_break_subtype","media_timeout_start","after_icing","after_icing_by_team","after_icing_by_opponent","home_away","long_change","playerId","positionCode","position","playerName",
            "seconds","pp_seconds","pk_seconds","pp_share","pk_share","offset_log_toi","offset_log_pp","offset_log_pk",
            "us_skaters_start","them_skaters_start","opponent_goalie_id_start",
            "onice_elapsed_at_window_start","shift_count_in_window","last_shift_len_s","time_since_last_shift_s","last_shift_len_missing","time_since_last_shift_missing",
            "entered_after_start","entry_offset_s","exited_before_end","exit_offset_s",
            "stint_duration_max","stint_duration_st",
             "score_diff_start","score_state_start","clock_s",
             "matchup_quality_pct",
             "fo_seen_start","fo_team_won_start","fo_team_lost_start","fo_won_start","fo_lost_start","fo_took_start","fo_took_won_start","fo_took_lost_start","home_last_change_opportunity",
             "rest_days_team","b2b_team",
             "standing_prior",
             "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
             "tsf_0_5","tsf_6_20","tsf_21_60","tsf_61p",
             "elapsed_share_0_5","elapsed_share_6_20","elapsed_share_21_60","elapsed_share_61p",
             "early_mass",
             "ai_OZ_start","ai_DZ_start","fo_O_start","fo_D_start","last_change_start","long_change_start",
             "teammates_onice_ids_start","opponents_onice_ids_start",
             "teammates_onice_ids_w","teammates_onice_w","teammates_onice_sec_w",
             "opponents_onice_ids_w","opponents_onice_w","opponents_onice_sec_w",
             "with_event_GF","with_event_GA","with_event_SF","with_event_SA",
             "giveaways_committed","takeaways_forced","hits_personal","blocks_personal",
             "gamePk","teamId","date","rinkid"]
        )

        # Player rollup: re-read the finalized train CSV from disk, and project exact columns
        train_csv_path = os.path.join(args.out_dir, f"player_windows_train_{gamePk}.csv")
        import pandas as _pd
        df_train = _pd.read_csv(train_csv_path)
        # Derive season from date column
        def _season_from_date(df):
            try:
                dt = _pd.to_datetime(df["date"].iloc[0], errors="coerce")
                if _pd.notna(dt):
                    y = int(dt.year if dt.month >= 7 else dt.year - 1)
                    return f"{y}{y+1}"
            except Exception:
                pass
            try:
                y = int(str(gamePk)[:4])
                return f"{y}{y+1}"
            except Exception:
                return ""
        season_str = _season_from_date(df_train)
        df_train["season"] = season_str
        # Canonicalize keys (gamePk/teamId); then select requested columns
        if "gamePk" not in df_train.columns and "gameid" in df_train.columns:
            df_train["gamePk"] = df_train["gameid"]
        if "teamId" not in df_train.columns and "teamid" in df_train.columns:
            df_train["teamId"] = df_train["teamid"]
        wanted_cols = [
            # A) Identity / keys
            "season","date","gamePk","teamId","playerId","positionCode","position","team_side","window_id",
            # B) Window timing
            "start_sec","end_sec","seconds","period",
            # C) Strength & skater counts
            "pp_seconds","pk_seconds","us_skaters_start","them_skaters_start","strength_global",
            # D) Events/numerators
            "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
            # E) EV deployment markers
            "zone_start", "tsf_0_5","tsf_6_20","tsf_21_60","tsf_61p","last_change_start","long_change_start","early_mass",
            # F) Co-presence / chemistry payload
            "teammates_onice_ids_w","teammates_onice_sec_w","teammates_onice_w",
            "opponents_onice_ids_w","opponents_onice_sec_w","opponents_onice_w",
            # G) Faceoff flags at window start
            "fo_seen_start","fo_team_won_start","fo_team_lost_start","fo_took_start","fo_took_won_start","fo_took_lost_start",
            # H) Bench / shift history
            "last_shift_len_s","time_since_last_shift_s","entry_offset_s","exited_before_end","exit_offset_s",
            # I) Micro events
            "giveaways_committed","takeaways_forced","hits_personal","blocks_personal",
        ]
        # Ensure all wanted columns exist; backfill missing with zeros (numeric) or empty lists as appropriate
        for c in wanted_cols:
            if c not in df_train.columns:
                # Faceoff flags and counts default to 0
                df_train[c] = 0
        cols_present = [c for c in wanted_cols if c in df_train.columns]
        player_rollup_rows = df_train[cols_present].to_dict(orient="records")

        write_csv(
            os.path.join(args.out_dir, f"player_rollup_{gamePk}.csv"),
            player_rollup_rows,
            [
                # A) Identity / keys
                "season","date","gameid","teamid","playerId","positionCode","position","team_side","window_id",
                # B) Window timing
                "start_sec","end_sec","seconds","period",
                # C) Strength & skater counts
                "pp_seconds","pk_seconds","us_skaters_start","them_skaters_start","strength_global",
                # D) Events/numerators
                "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
                # E) EV deployment markers
                "zone_start",
                # F) Co-presence / chemistry payload
                "teammates_onice_ids_w","teammates_onice_sec_w","teammates_onice_w",
                "opponents_onice_ids_w","opponents_onice_sec_w","opponents_onice_w",
                # G) Faceoff flags at window start
                "fo_seen_start","fo_team_won_start","fo_team_lost_start","fo_took_start","fo_took_won_start","fo_took_lost_start",
                # H) Bench / shift history
                "last_shift_len_s","time_since_last_shift_s","entry_offset_s","exited_before_end","exit_offset_s",
                # I) Micro events
                "giveaways_committed","takeaways_forced","hits_personal","blocks_personal",
            ]
        )

        # --- Team rollup (projection of train rows, one row per player/window with only team-context cols) ---
        try:
            df_team = df_train.copy()
            try:
                df_team = df_team.loc[:, ~df_team.columns.duplicated()].copy()
            except Exception:
                pass
            # Map aliases
            if "gamePk" not in df_team.columns and "gameid" in df_team.columns:
                df_team["gamePk"] = df_team["gameid"]
            if "teamId" not in df_team.columns and "teamid" in df_team.columns:
                df_team["teamId"] = df_team["teamid"]
            # Ensure opponent_goalie_id_start is present by deriving from goalies_start if missing/blank
            try:
                if "opponent_goalie_id_start" not in df_team.columns or df_team["opponent_goalie_id_start"].isna().all():
                    # Need team_side and goalies_start to infer
                    if "team_side" in df_team.columns and "goalies_start" in df_team.columns:
                        def _derive_opp_goalie(row):
                            try:
                                gs = row.get("goalies_start") if isinstance(row, dict) else row["goalies_start"]
                            except Exception:
                                gs = None
                            try:
                                side = row.get("team_side") if isinstance(row, dict) else row["team_side"]
                            except Exception:
                                side = None
                            try:
                                if isinstance(gs, dict):
                                    if str(side) == "home":
                                        return int(gs.get("away", 0) or 0)
                                    if str(side) == "away":
                                        return int(gs.get("home", 0) or 0)
                            except Exception:
                                return 0
                            return 0
                        df_team["opponent_goalie_id_start"] = df_team.apply(_derive_opp_goalie, axis=1)
            except Exception:
                # If anything fails, default to 0 to avoid blanks
                df_team["opponent_goalie_id_start"] = df_team.get("opponent_goalie_id_start", 0)
            team_required = [
                "season","date","gamePk","teamId","playerId","positionCode","position","team_side","window_id","strength_global","seconds","rinkid",
                "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
                "elapsed_share_0_5","elapsed_share_6_20","elapsed_share_21_60","elapsed_share_61p",
                "score_state_start","score_diff_start","home_away","period","clock_s",
                "long_change","start_prev_break_type","start_prev_break_subtype","media_timeout_start",
                "after_icing","after_icing_by_team","after_icing_by_opponent","home_last_change_opportunity",
                "us_skaters_start", "them_skaters_start","tsf_0_5","tsf_6_20","tsf_21_60","tsf_61p","last_change_start","long_change_start","early_mass","fo_took_start","fo_took_won_start","fo_took_lost_start",
                "zone_start","fo_zone","fo_seen_start","fo_team_won_start","fo_team_lost_start","fo_won_start","fo_lost_start",
                "fo_O_start","fo_D_start","rest_days_team","b2b_team",
                "opponent_goalie_id_start",
            ]
            for c in team_required:
                if c not in df_team.columns:
                    df_team[c] = 0
            team_cols_present = [c for c in team_required if c in df_team.columns]
            team_rollup_rows = df_team[team_cols_present].to_dict(orient="records")
            write_csv(
                os.path.join(args.out_dir, f"team_rollup_{gamePk}.csv"),
                team_rollup_rows,
                team_required,
            )
        except Exception as _e:
            try:
                print({"warn":"team_rollup_build_failed","gamePk": gamePk, "error": str(_e)})
            except Exception:
                pass
        if args.csv_only_player_train:
            print("CSV file written: player_windows_train only.")
        else:
            print("CSV files written.")

def _score_state_bucket(diff: int) -> str:
    if diff <= -2:
        return "trail2+"
    if diff == -1:
        return "trail1"
    if diff == 0:
        return "tie"
    if diff == 1:
        return "lead1"
    return "lead2+"

if __name__ == "__main__":
    main()
