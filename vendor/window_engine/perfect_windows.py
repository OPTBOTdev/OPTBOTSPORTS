import argparse
import csv
import json
import math
import os
import re
import datetime
import urllib.request
import urllib.error
import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

"""
perfect_windows.py (TOKENS ONLY, CSV ONLY)

This script builds:
- Window *containers* (same segmentation concept as before): contiguous time spans in seconds, bounded by
  stoppages / faceoffs / goals / penalties / challenges / timeouts / strength changes / hard caps / period end.
  These containers are NOT the modeling unit; they are only the parent structure.

- Player *tokens* inside each container: a token is a decision-time snapshot for a specific player, predicting
  what happens NEXT over a fixed horizon. Tokens are allowed to overlap; they are NOT additive and should not
  be used to reconstruct the container.

Outputs (CSV-only):
- windows_{gamePk}.csv
- player_tokens_{gamePk}.csv

Token row key (primary):
  (season, date, gamePk, teamId, strength_global, window_id, playerId, token_idx)

Train vs Sim feature namespaces:
- sim_* columns: mechanistic / known-in-sim variables at t_token (time, strength, score, roster IDs, goalie IDs, etc.)
- train_* columns: may include richer observed-world history up to t_token (currently a small superset)
  NOTE: both namespaces are flattened into CSV columns.
"""


# -------------------- Constants --------------------

SECONDS_PER_PERIOD = 1200

SHOT_ON_GOAL_TYPES = {"shot-on-goal"}
GOAL_TYPES = {"goal"}
MISS_TYPES = {"missed-shot"}
BLOCK_TYPES = {"blocked-shot"}
SHOT_TYPES = SHOT_ON_GOAL_TYPES | MISS_TYPES | BLOCK_TYPES

MICRO_TYPES = {"giveaway", "takeaway", "hit", "blocked-shot"}
ZONE_BUCKETS = ("OZ", "NZ", "DZ")

TOKEN_PRIORITIES = {"WINDOW_START": 0, "ENTRY": 1, "STATE": 2, "EXIT": 3}

# Selection/termination supervision tokens:
ANCHOR_TOKEN_TYPES = {"ENTRY", "EXIT", "WINDOW_START"}
# Outcome supervision can be attached to any token row that coincides with a state-run start.
# Practically:
# - Mid-shift state starts use token_type=STATE
# - If a state starts at ENTRY or WINDOW_START, we merge them into a single row (token_type remains ENTRY/WINDOW_START)
#   and set is_outcome_token=1.
OUTCOME_TOKEN_TYPES = {"STATE"}  # kept for backwards compatibility / readability

# Chemistry / co-presence filtering (kept; used in token horizon aggregation)
RAW_SHARE_MIN = 0.12
SEC_FLOOR_BASE = 6
SHORT_WIN_SEC = 25
SHORT_SHARE_MIN = 0.22
SHORT_SEC_FRAC = 0.35
TOPK_FALLBACK = 1


# -------------------- IO helpers --------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cap_skaters_to_six(cur: List[int], prev: List[int]) -> List[int]:
    """
    Data guard: some raw feeds occasionally provide >6 skaters for a side at a second (physically impossible).
    We deterministically drop extras to get back to 6, preferring to keep continuity with the previous second.
    """
    cur2 = [int(x) for x in (cur or [])]
    prev_set = set(int(x) for x in (prev or []))
    while len(cur2) > 6:
        # Prefer dropping a player who is not present in the previous snapshot (minimizes jitter)
        drop_candidates = [x for x in cur2 if x not in prev_set]
        if drop_candidates:
            drop = sorted(drop_candidates)[-1]  # deterministic
        else:
            drop = sorted(cur2)[-1]
        try:
            cur2.remove(drop)
        except ValueError:
            cur2 = cur2[:-1]
    return cur2


def load_player_meta_csv(path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Load player metadata mapping from a CSV that contains at least:
      - playerId (or PlayerID)
      - playerName (or PlayerName)
      - positionCode (or Position)
    Returns (player_name_map, player_pos_map). Missing/parse errors are skipped.
    """
    if not path:
        return {}, {}
    if not os.path.exists(path):
        return {}, {}
    name_map: Dict[int, str] = {}
    pos_map: Dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rr = csv.DictReader(f)
            for row in rr:
                pid = row.get("playerId") or row.get("PlayerID") or row.get("player_id") or row.get("id")
                try:
                    pid_i = int(float(str(pid).strip()))
    except Exception:
                    continue
                if pid_i <= 0:
                    continue
                nm = (row.get("playerName") or row.get("PlayerName") or row.get("name") or "").strip()
                pos = (row.get("positionCode") or row.get("Position") or row.get("position") or "").strip()
                if nm and pid_i not in name_map:
                    name_map[pid_i] = nm
                if pos and pid_i not in pos_map:
                    pos_map[pid_i] = pos
    except Exception:
        # best effort; return whatever we loaded
        pass
    return name_map, pos_map


def load_player_handedness_csv(path: str) -> Dict[int, str]:
    """
    Load handedness (shoots/catches) from a CSV if available.
    Accepts columns: handedness / shootsCatches / shoots_catches.
    Returns playerId -> handedness (e.g. "L","R").
    """
    if not path or not os.path.exists(path):
        return {}
    out: Dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            rr = csv.DictReader(f)
            for row in rr:
                pid = row.get("playerId") or row.get("PlayerID") or row.get("player_id") or row.get("id")
                try:
                    pid_i = int(float(str(pid).strip()))
    except Exception:
                    continue
                if pid_i <= 0:
                    continue
                hand = (row.get("handedness") or row.get("shootsCatches") or row.get("shoots_catches") or "").strip()
                if hand and pid_i not in out:
                    out[pid_i] = str(hand)
    except Exception:
        pass
    return out


def _parse_yyyy_mm_dd(s: str) -> Optional[datetime.date]:
    try:
        s = str(s or "").strip()
        if not s:
            return None
        return datetime.date.fromisoformat(s[:10])
        except Exception:
        return None


def compute_age_years(birth_date: Optional[datetime.date], on_date: Optional[datetime.date]) -> Optional[int]:
    """Compute integer age in years at on_date."""
    if birth_date is None or on_date is None:
        return None
    try:
        years = on_date.year - birth_date.year
        if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
            years -= 1
        if years < 0 or years > 80:
            return None
        return int(years)
        except Exception:
        return None


def last_non_shift_event_before(
    *,
    t_context: int,
    lookback_sec: int,
    events_by_sec: Dict[int, List[Dict[str, Any]]],
) -> Tuple[str, int, int, str, int]:
    """
    Return the last "meaningful" event at or before t_context, within lookback_sec seconds.
    Excludes shift-change events and period/game boundary admin events.

    Returns: (event_type, event_sec, time_since_s, zone_code, owner_team_id)
      - event_type="none" if not found within lookback
      - event_sec=-1 if none
      - time_since_s = t_context - event_sec (or lookback_sec+1 if none)
      - zone_code in {"O","N","D","flow","na"} if available
    """
    t0 = int(t_context)
    lb = max(0, int(lookback_sec))
    ignore = {"shift-change", "shift_change", "shift change", "period-start", "period-end", "game-end"}
    for s in range(t0, max(-1, t0 - lb) - 1, -1):
        best_so = None
        best_t = None
        best_zone = "na"
        best_owner = 0
        for ev in (events_by_sec.get(int(s), []) or []):
            et = str(ev.get("type") or "").lower()
            if not et or et in ignore:
                continue
            so = _safe_int(ev.get("sortOrder", 0))
            if best_so is None or so > int(best_so):
                best_so = int(so)
                best_t = et
                det = ev.get("details") or {}
                zc = det.get("zoneCode") or det.get("zone_code")
                z = "na"
                if zc:
                    zc = str(zc).upper().strip()
                    if zc in {"O", "N", "D"}:
                        z = zc
                else:
                    # If no zoneCode, but this is a faceoff-like record, treat as flow/unknown.
                    z = "na"
                best_zone = z
                best_owner = _safe_int(det.get("eventOwnerTeamId"), 0)
        if best_t is not None:
            return str(best_t), int(s), int(t0 - int(s)), str(best_zone), int(best_owner)
    return "none", -1, int(lb + 1), "na", 0


def last_non_shift_event_before_adaptive(
    *,
    t_context: int,
    events_by_sec: Dict[int, List[Dict[str, Any]]],
    lookback_primary_sec: int = 6,
    lookback_extended_sec: int = 10,
) -> Tuple[str, int, int, int, str, int]:
    """
    Adaptive version:
    - First search within lookback_primary_sec
    - If not found, search within lookback_extended_sec

    Returns (event_type, event_sec, time_since_s, lookback_used_sec, zone_code, owner_team_id)
    where lookback_used_sec is:
      - lookback_primary_sec if found in primary window
      - lookback_extended_sec if only found after extending
      - 0 if not found at all
    """
    et, es, dt, z, owner = last_non_shift_event_before(t_context=t_context, lookback_sec=int(lookback_primary_sec), events_by_sec=events_by_sec)
    if et != "none":
        return et, es, dt, int(lookback_primary_sec), z, owner
    et2, es2, dt2, z2, owner2 = last_non_shift_event_before(t_context=t_context, lookback_sec=int(lookback_extended_sec), events_by_sec=events_by_sec)
    if et2 != "none":
        return et2, es2, dt2, int(lookback_extended_sec), z2, owner2
    return "none", -1, int(max(0, int(lookback_extended_sec)) + 1), 0, "na", 0


def build_shift_length_stats(onice_map: Dict[int, set], horizon: int) -> Dict[int, Dict[str, Any]]:
    """
    Build per-player completed shift length stats from a per-second on-ice map.
    Returns dict:
      stats[pid] = {"ends": [...], "cum_sum": [...], "cum_sum_sq": [...]}
    where ends[i] is the last on-ice second of a completed shift, and cum_* are prefix sums
    over the corresponding shift lengths in seconds (inclusive length).
    """
    # Track on/off transitions per player by scanning seconds.
    # Only skaters appear in onice_map here; goalies are handled elsewhere.
    current_start: Dict[int, int] = {}
    ends_by_pid: Dict[int, List[int]] = defaultdict(list)
    lens_by_pid: Dict[int, List[int]] = defaultdict(list)

    for s in range(0, int(horizon) + 1):
        on_now = onice_map.get(int(s), set()) or set()
        on_prev = onice_map.get(int(s) - 1, set()) if s > 0 else set()

        entered = on_now - on_prev
        exited = on_prev - on_now

        for pid in entered:
            if pid not in current_start:
                current_start[int(pid)] = int(s)
        for pid in exited:
            pid_i = int(pid)
            st = current_start.pop(pid_i, None)
            if st is None:
                continue
            # completed shift ended at s-1
            end_s = int(s - 1)
            ln = int(end_s - int(st) + 1)
            if ln > 0 and ln <= 600:
                ends_by_pid[pid_i].append(end_s)
                lens_by_pid[pid_i].append(ln)

    # Close any shift still on at horizon (treat as completed at horizon)
    for pid_i, st in list(current_start.items()):
        end_s = int(horizon)
        ln = int(end_s - int(st) + 1)
        if ln > 0 and ln <= 600:
            ends_by_pid[pid_i].append(end_s)
            lens_by_pid[pid_i].append(ln)

    out: Dict[int, Dict[str, Any]] = {}
    for pid_i, ends in ends_by_pid.items():
        # Ensure sorted by end time
        pairs = sorted(zip(ends, lens_by_pid.get(pid_i, [])), key=lambda x: x[0])
        ends_sorted = [int(e) for e, _ in pairs]
        lens_sorted = [int(l) for _, l in pairs]
        cum_sum = []
        cum_sum_sq = []
        s1 = 0
        s2 = 0
        for l in lens_sorted:
            s1 += int(l)
            s2 += int(l) * int(l)
            cum_sum.append(int(s1))
            cum_sum_sq.append(int(s2))
        out[int(pid_i)] = {"ends": ends_sorted, "cum_sum": cum_sum, "cum_sum_sq": cum_sum_sq}
    return out


def shift_mean_sd_before(stats: Dict[int, Dict[str, Any]], pid: int, t_context: int) -> Tuple[Optional[float], Optional[float], int]:
    """
    Return (mean, sd, n) of completed shift lengths for pid with shift_end < t_context.
    """
    st = stats.get(int(pid)) or {}
    ends = st.get("ends") or []
    if not ends:
        return None, None, 0
    idx = bisect.bisect_left(ends, int(t_context))
    if idx <= 0:
        return None, None, 0
    cum_sum = st.get("cum_sum") or []
    cum_sum_sq = st.get("cum_sum_sq") or []
    s1 = float(cum_sum[idx - 1])
    s2 = float(cum_sum_sq[idx - 1])
    n = int(idx)
    mean = float(s1 / float(n))
    var = max(0.0, float(s2 / float(n) - mean * mean))
    sd = float(math.sqrt(var))
    return mean, sd, n


def reason_proxy_bundle(
    *,
    token_type: str,
    t_token: int,
    t_context: int,
    strength_team: str,
    score_diff: int,
    last_event_type: str,
    last_event_owner_team_id: int,
    our_team_id: int,
    last_event_zone: str,
    current_shift_elapsed_s: int,
    mean_shift_len_s: Optional[float],
    sd_shift_len_s: Optional[float],
    position_code: str,
    is_period_boundary: int,
    own_goalie_pulled: int,
    opp_goalie_pulled: int,
    own_goalie_pull_transition: int,
    opp_goalie_pull_transition: int,
    boundary_prev_break_type: str,
) -> Dict[str, Any]:
    """
    High-confidence reasoning proxy for shift/line changes.
    Strategy: only emit "high confidence" labels when evidence is strong; otherwise label "unknown".
    Always return a confidence score and evidence flags so you can filter or model uncertainty.
    """
    tt = str(token_type or "").upper()
    if tt not in ("ENTRY", "EXIT", "STATE", "WINDOW_START"):
        return {"label": "na", "confidence": 0.0}

    let = str(last_event_type or "").lower()
    z = str(last_event_zone or "").upper()
    st = str(strength_team or "").upper()
    owner_tid = int(last_event_owner_team_id or 0)
    our_tid = int(our_team_id or 0)

    flags = {
        "is_special_teams": int(st in ("PP", "PK")),
        "is_after_faceoff": int(let == "faceoff"),
        "is_after_stoppage": int(let in ("stoppage", "goalie-stopped", "timeout", "challenge", "offside")),
        "is_after_icing": int(let == "icing"),
        "is_after_goal": int(let == "goal"),
        "is_after_penalty": int(let == "penalty"),
        "is_period_boundary": int(is_period_boundary),
        "own_goalie_pulled": int(own_goalie_pulled),
        "opp_goalie_pulled": int(opp_goalie_pulled),
        "own_goalie_pull_transition": int(own_goalie_pull_transition),
        "opp_goalie_pull_transition": int(opp_goalie_pull_transition),
        "boundary_prev_break_type": str(boundary_prev_break_type or ""),
        "zone_O": int(z == "O"),
        "zone_D": int(z == "D"),
        "zone_N": int(z == "N"),
        "score_big": int(abs(int(score_diff)) >= 2),
    }

    # 1) Strongest: special teams rotations
    if flags["is_special_teams"]:
        return {"label": "special_teams", "confidence": 0.92, **flags}

    # 1b) Period boundary is a very reliable shift-termination driver (esp for EXIT tokens)
    if int(is_period_boundary) != 0:
        return {"label": "period_boundary", "confidence": 0.98, **flags}

    # 1c) Goalie pull/return transitions: extremely high confidence for goalie EXIT/ENTRY moments
    pos = str(position_code or "").upper().strip()
    if pos == "G" and int(own_goalie_pull_transition) != 0:
        # goalie leaving because pulled or returning
        return {"label": "goalie_pull_transition", "confidence": 0.99, **flags}

    # 1d) Boundary strength-change context (window boundary tells us this cleanly)
    if str(boundary_prev_break_type or "").lower() == "strength":
        return {"label": "after_strength_change", "confidence": 0.97, **flags}

    # 2) Strong: immediate context events (stoppages/faceoffs/goals/penalties/icing)
    if let in ("faceoff", "stoppage", "goalie-stopped", "timeout", "challenge", "offside", "icing", "goal", "penalty"):
        # classify for/against when owner is known
        if owner_tid and our_tid and owner_tid != our_tid:
            suf = "_against"
        elif owner_tid and our_tid and owner_tid == our_tid:
            suf = "_for"
    else:
            suf = ""
        return {"label": f"after_{let}{suf}", "confidence": 0.97, **flags}

    # 3) Fatigue: require a clearly-long shift (relative to player's own mean when available)
    mean_s = float(mean_shift_len_s) if mean_shift_len_s is not None else 45.0
    sd_s = float(sd_shift_len_s) if sd_shift_len_s is not None else None
    zscore = None
    if sd_s is not None and sd_s > 1e-6:
        zscore = float((float(current_shift_elapsed_s) - mean_s) / sd_s)
    flags["fatigue_z"] = zscore
    flags["shift_elapsed_s"] = int(current_shift_elapsed_s)
    flags["shift_mean_s"] = float(mean_s)

    if int(current_shift_elapsed_s) >= max(50, int(mean_s * 1.35)) and (zscore is None or zscore >= 1.0):
        return {"label": "fatigue", "confidence": 0.85, **flags}

    # 4) Everything else is genuinely ambiguous without coach intent, puck control, etc.
    return {"label": "unknown", "confidence": 0.40, **flags}


def get_player_landing_cached(player_id: int, cache_dir: str, *, allow_fetch: bool = True, timeout_s: float = 10.0) -> Dict[str, Any]:
    """
    Best-effort fetch of api-web player landing JSON with local caching.
    Cache file: {cache_dir}/{player_id}.json
    """
    pid = int(player_id)
    if pid <= 0:
        return {}
    cache_dir = str(cache_dir or "").strip() or "artifacts/cache/player_landing"
    ensure_dir(cache_dir)
    fp = os.path.join(cache_dir, f"{pid}.json")
    try:
        if os.path.exists(fp):
            return load_json(fp) or {}
        except Exception:
        pass
    if not allow_fetch:
        return {}
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            try:
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
    except Exception:
        pass
            return data or {}
    except Exception:
        return {}


def load_raw_boxscore_team_meta(raw_dir: str, gamePk: int) -> Dict[str, Any]:
    """
    Extract team metadata from raw boxscore:
      - home/away teamId, abbrev, fullName ("Place CommonName")
    Returns dict with keys: homeTeamId, awayTeamId, homeAbbrev, awayAbbrev, homeName, awayName
    """
    out = {
        "homeTeamId": 0,
        "awayTeamId": 0,
        "homeAbbrev": "",
        "awayAbbrev": "",
        "homeName": "",
        "awayName": "",
    }
    if not raw_dir:
        return out
    raw_dir_fs = raw_dir.replace("/", os.sep).rstrip(os.sep)
    box_path = os.path.join(raw_dir_fs, "boxscore", f"{int(gamePk)}.json")
    if not os.path.exists(box_path):
        return out
    try:
        bs = load_json(box_path) or {}
        ht = bs.get("homeTeam") or {}
        at = bs.get("awayTeam") or {}
        out["homeTeamId"] = _safe_int(ht.get("id"), 0)
        out["awayTeamId"] = _safe_int(at.get("id"), 0)
        out["homeAbbrev"] = str(ht.get("abbrev") or "").strip()
        out["awayAbbrev"] = str(at.get("abbrev") or "").strip()
        hp = str(((ht.get("placeName") or {}).get("default")) or "").strip()
        hc = str(((ht.get("commonName") or {}).get("default")) or "").strip()
        ap = str(((at.get("placeName") or {}).get("default")) or "").strip()
        ac = str(((at.get("commonName") or {}).get("default")) or "").strip()
        out["homeName"] = (hp + " " + hc).strip() or hc or hp
        out["awayName"] = (ap + " " + ac).strip() or ac or ap
    except Exception:
        return out
    return out


def _infer_raw_dir_from_pbpice_path(pbpice_path: str) -> str:
    """
    Given a path like ".../raw/pbpice/pbp_onice_<gamePk>.json", return the ".../raw" directory.
    If not found, return empty string.
    """
    if not pbpice_path:
        return ""
    norm = pbpice_path.replace("\\", "/")
    marker = "/raw/"
    if marker not in norm:
        return ""
    return norm.split(marker)[0] + marker[:-1]  # keep trailing "/raw"


def load_raw_boxscore_and_shiftcharts_meta(
    raw_dir: str, gamePk: int
) -> Tuple[str, str, str, Dict[int, str], Dict[int, str]]:
    """
    Best-effort extraction of:
    - rinkid (arena name)
    - season (string)
    - date (YYYY-MM-DD)
    - player_name_map (full if available)
    - player_pos_map (positionCode)

    Sources (if present):
    - {raw_dir}/boxscore/{gamePk}.json: venue.default, season, gameDate, playerByGameStats.*.{forwards/defense/goalies}
    - {raw_dir}/shiftcharts/{gamePk}.json: firstName/lastName per playerId (better full names)
    """
    rinkid = ""
    season = ""
    date = ""
    name_map: Dict[int, str] = {}
    pos_map: Dict[int, str] = {}

    if not raw_dir:
        return rinkid, season, date, name_map, pos_map

    raw_dir_fs = raw_dir.replace("/", os.sep).rstrip(os.sep)
    box_path = os.path.join(raw_dir_fs, "boxscore", f"{int(gamePk)}.json")
    shifts_path = os.path.join(raw_dir_fs, "shiftcharts", f"{int(gamePk)}.json")

    # boxscore: arena + player positions + short names
    try:
        if os.path.exists(box_path):
            bs = load_json(box_path)
            try:
                rinkid = str(((bs.get("venue") or {}).get("default")) or "").strip()
                except Exception:
                rinkid = ""
                try:
                season = str(bs.get("season") or "").strip()
                except Exception:
                season = ""
            try:
                date = str(bs.get("gameDate") or "").strip()
            except Exception:
                date = ""

            pbg = bs.get("playerByGameStats") or {}
            for team_key in ("homeTeam", "awayTeam"):
                team = pbg.get(team_key) or {}
                for group in ("forwards", "defense", "goalies"):
                    for pr in (team.get(group) or []):
                        try:
                            pid = int(pr.get("playerId"))
                        except Exception:
                            continue
                        if pid <= 0:
                            continue
                        pos = str(pr.get("position") or "").strip()
                        nm = str(((pr.get("name") or {}).get("default")) or "").strip()
                        if pos and pid not in pos_map:
                            pos_map[pid] = pos
                        if nm and pid not in name_map:
                            name_map[pid] = nm
        except Exception:
            pass

    # shiftcharts: full names
    try:
        if os.path.exists(shifts_path):
            sh = load_json(shifts_path)
            if isinstance(sh, list):
                for r in sh:
                    try:
                        pid = int(r.get("playerId"))
                    except Exception:
                        continue
                    if pid <= 0:
                        continue
                    fn = str(r.get("firstName") or "").strip()
                    ln = str(r.get("lastName") or "").strip()
                    full = (fn + " " + ln).strip()
                    if full:
                        name_map[pid] = full  # prefer full name
            except Exception:
                pass

    return rinkid, season, date, name_map, pos_map


def write_csv(path: str, rows: List[Dict[str, Any]], keys: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def stable_sample_ints(candidates: Sequence[int], k: int, seed: int) -> List[int]:
    """Deterministic sample without importing numpy; stable across runs."""
    if k <= 0:
        return []
    cand = list(dict.fromkeys(int(x) for x in candidates))  # unique, preserve order
    if len(cand) <= k:
        return cand
    import random
    rng = random.Random(int(seed))
    # Fisher-Yates partial shuffle
    for i in range(min(k, len(cand) - 1)):
        j = rng.randrange(i, len(cand))
        cand[i], cand[j] = cand[j], cand[i]
    return cand[:k]


# -------------------- Time / strength helpers --------------------

def period_of(sec_game: int) -> int:
    return int(sec_game // SECONDS_PER_PERIOD) + 1


def clock_s_at(sec_game: int) -> int:
    return int(sec_game % SECONDS_PER_PERIOD)


def clock_str(sec_game: int) -> str:
    s = clock_s_at(sec_game)
    mm = int(s // 60)
    ss = int(s % 60)
    return f"{mm}:{ss:02d}"


def is_long_change(sec_game: int) -> int:
    # Long change in periods 2 and 4 (OT treated as 4 here)
    p = period_of(sec_game)
    return int(p in (2, 4))


def strength_for_team(sk_for: int, sk_against: int, goalie_for: int, goalie_against: int) -> str:
    # goalie_for==0 means pulled
    if goalie_for == 0 and goalie_against > 0:
        return "EN_for"
    if goalie_against == 0 and goalie_for > 0:
        return "EN_against"
    if sk_for == sk_against:
        return "5v5" if sk_for == 5 else f"{sk_for}v{sk_against}"
    if sk_for > sk_against:
        return "PP"
    return "PK"


# -------------------- Event parsing helpers --------------------

def _owner_side_for_event(details: Dict[str, Any], home_team_id: Optional[int], away_team_id: Optional[int]) -> Optional[str]:
    tid = details.get("eventOwnerTeamId") or details.get("teamId") or details.get("committingTeamId")
    try:
        tid = int(tid) if tid is not None else None
        except Exception:
        tid = None
    if tid is None:
        return None
    try:
        if home_team_id is not None and int(tid) == int(home_team_id):
            return "home"
        if away_team_id is not None and int(tid) == int(away_team_id):
            return "away"
        except Exception:
        return None
    return None


def _zone_bucket(details: Dict[str, Any]) -> Optional[str]:
    z = details.get("zoneCode") or details.get("zone") or details.get("zone_code")
    if z is None:
        return None
    z = str(z).upper().strip()
    if z in ("O", "OFF", "OFFENSIVE", "OZ"):
        return "OZ"
    if z in ("N", "NEU", "NEUTRAL", "NZ"):
        return "NZ"
    if z in ("D", "DEF", "DEFENSIVE", "DZ"):
        return "DZ"
    return None


def _bump_zone_counter(dst: Dict[str, int], details: Dict[str, Any]) -> None:
    zb = _zone_bucket(details)
    if zb in dst:
        dst[zb] += 1


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
        except Exception:
        return default


def _maybe_correct_onice_for_goal(
    sec: int,
    ev: Dict[str, Any],
    pre_home_set: set,
    pre_away_set: set,
    home_team_id: Optional[int],
    away_team_id: Optional[int],
    team_onice_by_sec: List[Tuple[List[int], List[int]]],
    orders_by_sec: Dict[int, List[Tuple[int, str]]],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
) -> Tuple[set, set]:
    """
    Jitter correction for goals.
    - If a same-second shift-change precedes the goal (sortOrder), or a shift at sec-1, use prior on-ice snapshot.
    - If credited scorer/assist is missing from on-ice set, look back up to 3s and swap in for a cameo.
    """
    det = ev.get("details") or {}
    goal_so = _safe_int(ev.get("sortOrder", 0))

    side_local = _owner_side_for_event(det, home_team_id, away_team_id)

    base_home = set(pre_home_set)
    base_away = set(pre_away_set)

    # shift-change before goal in same second?
    same_sec_shift_before = False
    try:
        evs_this = orders_by_sec.get(sec, [])
        same_sec_shift_before = any((t == "shift-change" and so < goal_so) for so, t in evs_this)
            except Exception:
        same_sec_shift_before = False
    tminus1_has_shift = bool(shift_changes_by_sec.get(sec - 1)) if sec - 1 >= 0 else False

    if same_sec_shift_before or tminus1_has_shift:
        start_prev = sec - 2 if tminus1_has_shift else sec - 1
        lookback = 0
        prev_s = start_prev
        while prev_s >= 0 and lookback < 30:
            if prev_s < len(team_onice_by_sec):
                ph, pa = team_onice_by_sec[prev_s]
                base_home = set(ph)
                base_away = set(pa)
                # prefer stable-ish 5v5 frames if possible
                if len(ph) >= 5 and len(pa) >= 5:
                break
            prev_s -= 1
            lookback += 1

    parts: List[int] = []
    for k in ("scoringPlayerId", "assist1PlayerId", "assist2PlayerId"):
            v = det.get(k)
            try:
                if v is not None:
                    parts.append(int(v))
            except Exception:
            continue

    if parts and side_local in ("home", "away"):
        goal_set = base_home if side_local == "home" else base_away
        missing = next((pid for pid in parts if pid not in goal_set), None)
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
                cand_out = next((pid_now for pid_now in list(goal_set) if pid_now not in prev_side_set), None)
                if cand_out is not None:
                    goal_set.discard(cand_out)
                    goal_set.add(missing)
        if side_local == "home":
            base_home = goal_set
        else:
            base_away = goal_set

    return base_home, base_away


# -------------------- Core builders --------------------

@dataclass
class ParsedGame:
    gamePk: int
    home_team_id: int
    away_team_id: int
    events: List[Dict[str, Any]]


def parse_game(pbp_onice: Dict[str, Any]) -> ParsedGame:
    gamePk = int(pbp_onice.get("gamePk"))
    home_team_id = int(pbp_onice.get("home", {}).get("teamId"))
    away_team_id = int(pbp_onice.get("away", {}).get("teamId"))
    events = list(pbp_onice.get("events") or [])
    return ParsedGame(gamePk=gamePk, home_team_id=home_team_id, away_team_id=away_team_id, events=events)


def build_second_index(game: ParsedGame) -> Tuple[
    List[Tuple[List[int], List[int]]],  # team_onice_by_sec
    List[Dict[str, int]],               # goalie_ids_by_sec
    Dict[int, List[Dict[str, Any]]],    # events_by_sec
    Dict[int, List[Tuple[int, str]]],   # orders_by_sec: (sortOrder, type)
    Dict[int, List[Tuple[int, str]]],   # sso_by_sec: (same_sec_order, type)
    Dict[int, Dict[str, set]],          # shift_changes_by_sec
    Dict[int, Dict[str, Any]],          # end_event_at
    Dict[int, Dict[str, Any]],          # fo_meta
    set,                                # tv_timeout_secs
    int,                                # horizon (max sec)
]:
    events_by_sec: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    orders_by_sec: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    sso_by_sec: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    shift_changes_by_sec: Dict[int, Dict[str, set]] = defaultdict(lambda: {"home_in": set(), "home_out": set(), "away_in": set(), "away_out": set()})
    end_event_at: Dict[int, Dict[str, Any]] = {}
    fo_meta: Dict[int, Dict[str, Any]] = {}
    tv_timeout_secs: set = set()

    max_sec = 0
    for ev in game.events:
        sec = ev.get("sec_game")
        if sec is None:
            continue
        sec = int(sec)
        max_sec = max(max_sec, sec)
        et = str(ev.get("type") or "").lower()
        events_by_sec[sec].append(ev)
        orders_by_sec[sec].append((_safe_int(ev.get("sortOrder", 0)), et))
        sso_by_sec[sec].append((_safe_int(ev.get("same_sec_order", 0)), et))

        # faceoff metadata at second
        if et == "faceoff":
            det = ev.get("details") or {}
            zc = det.get("zoneCode")
            z = "flow"
            if zc:
                zc = str(zc).upper()
                if zc == "O":
                    z = "O"
                elif zc == "D":
                    z = "D"
                elif zc == "N":
                    z = "N"
            fo_meta[sec] = {
                "zone": z,
                "eventOwnerTeamId": det.get("eventOwnerTeamId"),
                "winningPlayerId": det.get("winningPlayerId"),
                "losingPlayerId": det.get("losingPlayerId"),
            }

        # shift-change deltas (already provided)
        if et == "shift-change":
            sc = ev.get("shift_change") or {}
            for k in ("home_in", "home_out", "away_in", "away_out"):
                vals = sc.get(k) or []
                try:
                    shift_changes_by_sec[sec][k].update(int(x) for x in vals)
            except Exception:
                    pass

        # TV timeout signal (if present as stoppage reason)
        if et in ("stoppage", "goalie-stopped"):
            det = ev.get("details") or {}
            reason = str(det.get("reason") or det.get("stoppageReason") or "").lower()
            if "tv" in reason or "media" in reason or "commercial" in reason:
                tv_timeout_secs.add(sec)

        # store last event at second (for break reason/metadata)
        end_event_at[sec] = ev

    # Build per-second on-ice snapshots from event.onice (pre-change snapshots)
    horizon = max_sec
    team_onice_by_sec: List[Tuple[List[int], List[int]]] = [([], []) for _ in range(horizon + 2)]
    goalie_ids_by_sec: List[Dict[str, int]] = [{"home": 0, "away": 0} for _ in range(horizon + 2)]

    last_home: List[int] = []
    last_away: List[int] = []
    last_goalies = {"home": 0, "away": 0}

    for s in range(horizon + 1):
        prev_home = list(last_home)
        prev_away = list(last_away)
        # choose the last event in this second with onice snapshot, preferring sec_phase == pre-change
        chosen = None
        for ev in events_by_sec.get(s, []):
            if isinstance(ev.get("onice"), dict):
                if str(ev.get("sec_phase") or "").lower() == "pre-change":
                    chosen = ev
        if chosen is None:
            for ev in reversed(events_by_sec.get(s, [])):
                if isinstance(ev.get("onice"), dict):
                    chosen = ev
                    break
        if chosen is not None:
            on = chosen.get("onice") or {}
            h = on.get("home") or []
            a = on.get("away") or []
            last_home = [int(x) for x in h]
            last_away = [int(x) for x in a]
            g = (on.get("goalies") or {})
            last_goalies = {
                "home": _safe_int(g.get("home", last_goalies["home"])),
                "away": _safe_int(g.get("away", last_goalies["away"])),
            }
        # Hard cap skaters to 6 (raw feed can glitch to 7+)
        if len(last_home) > 6:
            last_home = cap_skaters_to_six(last_home, prev_home)
        if len(last_away) > 6:
            last_away = cap_skaters_to_six(last_away, prev_away)
        # Data guard (NON-STICKY): some feeds occasionally report 6 skaters while still listing a goalie id.
        # Physically, 6 skaters implies the goalie is not on the ice (pulled / extra attacker) *for that second*.
        # Important: do NOT mutate `last_goalies` here, or you can "stick" goalie_id=0 for long stretches if later
        # on-ice snapshots omit goalie ids. Instead, compute a per-second goalie snapshot with the guard applied.
        goalies_now = dict(last_goalies)
        if len(last_home) >= 6:
            goalies_now["home"] = 0
        if len(last_away) >= 6:
            goalies_now["away"] = 0
        team_onice_by_sec[s] = (list(last_home), list(last_away))
        goalie_ids_by_sec[s] = goalies_now

    return (
        team_onice_by_sec,
        goalie_ids_by_sec,
        events_by_sec,
        orders_by_sec,
        sso_by_sec,
        shift_changes_by_sec,
        end_event_at,
        fo_meta,
        tv_timeout_secs,
        horizon,
    )


def build_credit_by_sec(
    game: ParsedGame,
    team_onice_by_sec: List[Tuple[List[int], List[int]]],
    goalie_ids_by_sec: List[Dict[str, int]],
    events_by_sec: Dict[int, List[Dict[str, Any]]],
    orders_by_sec: Dict[int, List[Tuple[int, str]]],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
) -> Tuple[Dict[int, Dict[str, Any]], List[int], List[int]]:
    """
    Build a per-second directional credit dict:
      credit_by_sec[s] = {
        "home": Counter-like dict of AF/SF/BF/GF etc,
        "away": ...
        "onice_home": list of skaters home (pre-change snapshot for this second)
        "onice_away": list of skaters away
      }
    Also build cumulative goals arrays for score state at token times.
    """
    credit_by_sec: Dict[int, Dict[str, Any]] = {}

    horizon = len(team_onice_by_sec) - 2
    cum_home_goals = [0] * (horizon + 2)
    cum_away_goals = [0] * (horizon + 2)

    seen_goal_keys: set = set()

    for s in range(horizon + 1):
        home_ids, away_ids = team_onice_by_sec[s]
        cb = {
            "home": Counter(),
            "away": Counter(),
            "onice_home": list(home_ids),
            "onice_away": list(away_ids),
        }
        evs = events_by_sec.get(s, [])
        for ev in evs:
            et = str(ev.get("type") or "").lower()
            if et not in SHOT_TYPES and et not in GOAL_TYPES and et not in MICRO_TYPES:
            continue
            det = ev.get("details") or {}

            # micro
            if et in ("giveaway", "takeaway"):
                pid = det.get("playerId") or det.get("player_id") or det.get("actorPlayerId")
                pid = _safe_int(pid, default=0)
                if pid:
                    # store on credit bucket later in token aggregation; here only keep event list
                    pass

            # shots / goals
            if et in SHOT_TYPES or et in GOAL_TYPES:
                if et in BLOCK_TYPES and str(det.get("reason", "")).lower() == "teammate-blocked":
                    continue
                side = _owner_side_for_event(det, game.home_team_id, game.away_team_id)
                if side not in ("home", "away"):
                    continue
                opp = "away" if side == "home" else "home"

                # attempts for owner
                cb[side]["AF"] += 1
                cb[opp]["AA"] += 1

                # shots on goal for owner
                if et in SHOT_ON_GOAL_TYPES or et in GOAL_TYPES:
                    cb[side]["SF"] += 1
                    cb[opp]["SA"] += 1

                # goals (dedupe by (sec, owner team, scorer))
                if et in GOAL_TYPES:
                    owner_tid = det.get("eventOwnerTeamId")
                    scorer = det.get("scoringPlayerId") or det.get("shootingPlayerId")
                    goal_key = (s, owner_tid, scorer)
                    if goal_key in seen_goal_keys:
                        continue
                    seen_goal_keys.add(goal_key)
                    cb[side]["GF"] += 1
                    cb[opp]["GA"] += 1

                # blocked-for/against: attribute by defender side if blocker is known onice
                if et in BLOCK_TYPES:
                    blk_pid = det.get("blockingPlayerId") or det.get("blockedByPlayerId") or det.get("blockerPlayerId") or det.get("blockerId")
                    try:
                        blk_pid = int(blk_pid) if blk_pid is not None else None
                    except Exception:
                        blk_pid = None
                    defender_side = None
                    if blk_pid is not None:
                        if blk_pid in home_ids:
                            defender_side = "home"
                        elif blk_pid in away_ids:
                            defender_side = "away"
                    if defender_side is None:
                        defender_side = opp
                    attacker_side = "away" if defender_side == "home" else "home"
                    cb[defender_side]["BF"] += 1
                    cb[attacker_side]["BA"] += 1

        credit_by_sec[s] = cb

        # cum goals
        cum_home_goals[s + 1] = cum_home_goals[s] + int(cb["home"].get("GF", 0))
        cum_away_goals[s + 1] = cum_away_goals[s] + int(cb["away"].get("GF", 0))

    return credit_by_sec, cum_home_goals, cum_away_goals


def build_windows(
    game: ParsedGame,
    team_onice_by_sec: List[Tuple[List[int], List[int]]],
    goalie_ids_by_sec: List[Dict[str, int]],
    events_by_sec: Dict[int, List[Dict[str, Any]]],
    orders_by_sec: Dict[int, List[Tuple[int, str]]],
    sso_by_sec: Dict[int, List[Tuple[int, str]]],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
    end_event_at: Dict[int, Dict[str, Any]],
    fo_meta: Dict[int, Dict[str, Any]],
    tv_timeout_secs: set,
    horizon: int,
    hard_cap_sec: int,
) -> List[Dict[str, Any]]:
    # strength_global: based on skater counts + goalie presence by second
    def strength_at(sec: int) -> str:
        sec = max(0, min(sec, horizon))
        h_ids, a_ids = team_onice_by_sec[sec]
        gh = _safe_int(goalie_ids_by_sec[sec].get("home", 0))
        ga = _safe_int(goalie_ids_by_sec[sec].get("away", 0))
        gh_present = 1 if gh != 0 else 0
        ga_present = 1 if ga != 0 else 0
        # Preserve full state for CIN / EN regimes
        # Canonical encoding: "{home_skaters}v{away_skaters}_GH{0/1}_GA{0/1}"
        return f"{len(h_ids)}v{len(a_ids)}_GH{gh_present}_GA{ga_present}"

    # breaks: natural + faceoffs
    break_secs: set = set()
    for s, ev in end_event_at.items():
        et = str(ev.get("type") or "").lower()
        if et in ("goal", "penalty", "stoppage", "goalie-stopped", "timeout", "challenge"):
            break_secs.add(int(s))
    # also treat faceoffs as breaks (if no natural break at same second, faceoff will flush)
    for s in fo_meta.keys():
        break_secs.add(int(s))

    windows: List[Dict[str, Any]] = []

    open_s = 0
    last_strength = strength_at(0)
    last_home, last_away = team_onice_by_sec[0]

    last_fo_zone: Optional[str] = None
    last_fo_winner_team_id: Optional[int] = None
    last_fo_winner_player_id: Optional[int] = None
    last_fo_loser_player_id: Optional[int] = None

    last_break_type: Optional[str] = None
    last_break_team_id: Optional[int] = None
    last_break_subtype: Optional[str] = None
    media_timeout_tag_start_sec: Optional[int] = None

    if 0 in fo_meta:
        last_fo_zone = fo_meta[0]["zone"]
        last_fo_winner_team_id = fo_meta[0].get("eventOwnerTeamId")
        last_fo_winner_player_id = fo_meta[0].get("winningPlayerId")
        last_fo_loser_player_id = fo_meta[0].get("losingPlayerId")

    def flush(end_s: int, reason: str):
        nonlocal open_s, last_strength, last_home, last_away, last_fo_zone, last_fo_winner_team_id, last_fo_winner_player_id, last_fo_loser_player_id
        nonlocal last_break_type, last_break_team_id, last_break_subtype, media_timeout_tag_start_sec
        if end_s <= open_s:
            return
        home_end, away_end = team_onice_by_sec[max(0, min(end_s - 1, horizon))]
        # Snapshot goalie presence at window start (for instruments)
        gh0 = _safe_int(goalie_ids_by_sec[open_s].get("home", 0)) if open_s < len(goalie_ids_by_sec) else 0
        ga0 = _safe_int(goalie_ids_by_sec[open_s].get("away", 0)) if open_s < len(goalie_ids_by_sec) else 0
        w = {
            "window_id": f"W{len(windows) + 1:04d}",
            "period": period_of(open_s),
            "start_sec": int(open_s),
            "end_sec": int(end_s),
            "duration": int(end_s - open_s),
            "clock_start": clock_str(open_s),
            "strength_global": last_strength,
            # goalie pull flags (instrumental; do not collapse into strength_global only)
            "home_goalie_present": int(1 if gh0 != 0 else 0),
            "away_goalie_present": int(1 if ga0 != 0 else 0),
            "home_goalie_pulled": int(1 if gh0 == 0 else 0),
            "away_goalie_pulled": int(1 if ga0 == 0 else 0),
            "fo_zone": last_fo_zone or "flow",
            "fo_won_team_id": last_fo_winner_team_id,
            "fo_won_player_id": last_fo_winner_player_id,
            "fo_lost_player_id": last_fo_loser_player_id,
            "start_prev_break_type": (last_break_type or "period_start"),
            "start_prev_break_team_id": last_break_team_id,
            "start_prev_break_subtype": last_break_subtype,
            "media_timeout_start": int(1 if (media_timeout_tag_start_sec is not None and media_timeout_tag_start_sec == open_s) else 0),
            "end_event_type": reason,
            "home_ids_start": list(last_home),
            "away_ids_start": list(last_away),
            "home_ids_end": list(home_end),
            "away_ids_end": list(away_end),
        }
        windows.append(w)

        open_s = end_s
        last_strength = strength_at(end_s)
        last_home, last_away = team_onice_by_sec[end_s]
        last_fo_zone = "flow"
        last_fo_winner_team_id = None
        last_fo_winner_player_id = None
        last_fo_loser_player_id = None
        if media_timeout_tag_start_sec is not None and media_timeout_tag_start_sec == open_s:
            media_timeout_tag_start_sec = None

    # NOTE: `sec == horizon` is a sentinel snapshot (we keep arrays sized horizon+1).
    # Do not let that sentinel second create synthetic "strength" breaks; period end
    # should be labeled as `period_end` via the final flush below.
    s = 1
    while s < horizon:
        cur_strength = strength_at(s)
        elapsed = s - open_s

        # natural break first
        ev = end_event_at.get(s)
        if ev is not None:
            et = str(ev.get("type") or "").lower()
            if et in ("goal", "penalty", "stoppage", "goalie-stopped", "timeout", "challenge"):
                det = ev.get("details") or {}
                sub_reason = str(det.get("reason") or det.get("stoppageReason") or "").lower()
                reason = "icing" if (et in {"stoppage", "goalie-stopped"} and sub_reason == "icing") else et
                by_tid = det.get("byTeamId") or det.get("teamId") or det.get("committingTeamId") or det.get("eventOwnerTeamId")
            try:
                by_tid = int(by_tid) if by_tid is not None else None
            except Exception:
                by_tid = None

            flush(s, reason)
            last_break_type = reason
            last_break_team_id = by_tid
                last_break_subtype = sub_reason if et in {"stoppage", "goalie-stopped"} else None
            if s in tv_timeout_secs:
                media_timeout_tag_start_sec = s
            if s in fo_meta:
                last_fo_zone = fo_meta[s]["zone"]
                last_fo_winner_team_id = fo_meta[s].get("eventOwnerTeamId")
                last_fo_winner_player_id = fo_meta[s].get("winningPlayerId")
                last_fo_loser_player_id = fo_meta[s].get("losingPlayerId")
            s += 1
            continue

        # faceoff-only break
        if s in fo_meta:
            flush(s, "faceoff")
            last_break_type = "faceoff"
            last_break_team_id = None
            last_break_subtype = None
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
    return windows


def aggregate_features_for_token(
    *,
    game: "ParsedGame",
    by_sec_home: Dict[int, set],
    by_sec_away: Dict[int, set],
    events_by_sec: Dict[int, List[Dict[str, Any]]],
    credit_by_sec: Dict[int, Dict[str, Any]],
    team_onice_by_sec: List[Tuple[List[int], List[int]]],
    orders_by_sec: Dict[int, List[Tuple[int, str]]],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
                             side: str,
    p: int,
    win_start: int,
    win_end: int,
    t_token: int,
    t_end: int,
) -> Dict[str, Any]:
    """
    Recompute ALL outcome/target features over the token horizon (t_token, t_end] clipped to window end,
    credited only at seconds where player p is on-ice (per-second snapshots).
    """
    start_target = max(int(win_start), int(t_token) + 1)
    end_target = min(int(t_end), int(win_end))
    if end_target < start_target:
        return {k: 0 for k in (
            "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
            "giveaways_committed","takeaways_forced","hits_personal","blocks_personal","shots_blocked_personal",
            "giveaways_committed_oz","giveaways_committed_nz","giveaways_committed_dz",
            "takeaways_forced_oz","takeaways_forced_nz","takeaways_forced_dz",
            "hits_personal_oz","hits_personal_nz","hits_personal_dz",
            "blocks_personal_oz","blocks_personal_nz","blocks_personal_dz",
            "shots_blocked_personal_oz","shots_blocked_personal_nz","shots_blocked_personal_dz",
        )} | {
            # True exposure for outcome heads: number of seconds in (t_token, t_end] where player is on-ice.
            # NOTE: This can be < seconds_token if the player exits before t_end (important for offsets/rate models).
            "seconds_onice_in_h": 0,
            # POST-horizon overlap evidence (DO NOT use as PRE/CIN inputs)
            "post_teammates_onice_ids_w": [], "post_teammates_onice_w": [], "post_teammates_onice_sec_w": [], "post_teammates_onice_share_raw": [],
            "post_opponents_onice_ids_w": [], "post_opponents_onice_w": [], "post_opponents_onice_sec_w": [], "post_opponents_onice_share_raw": [],
            "post_with_event_GF": [], "post_with_event_GA": [], "post_with_event_SF": [], "post_with_event_SA": [],
        }

    onice_map = by_sec_home if side == "home" else by_sec_away
    opp_map = by_sec_away if side == "home" else by_sec_home

    Y = Counter()
    xgF = 0.0
    xgA = 0.0

    gv_p = tk_p = hp_p = bp_p = sbp_p = 0
    gv_zone = {zb: 0 for zb in ZONE_BUCKETS}
    tk_zone = {zb: 0 for zb in ZONE_BUCKETS}
    hp_zone = {zb: 0 for zb in ZONE_BUCKETS}
    bp_zone = {zb: 0 for zb in ZONE_BUCKETS}
    sbp_zone = {zb: 0 for zb in ZONE_BUCKETS}

    overlap_with: Counter[int] = Counter()
    overlap_vs: Counter[int] = Counter()
    event_copres_with: Dict[int, Counter] = defaultdict(Counter)

    # IMPORTANT: include end_target (inclusive) for (t_token, t_end]
    for s in range(int(start_target), int(end_target) + 1):
        if p not in onice_map.get(int(s), ()):
            continue

        # micro events by actor at second s
        for ev in events_by_sec.get(int(s), []):
            et = str(ev.get("type") or "").lower()
            if et not in MICRO_TYPES:
                continue
            det = ev.get("details") or {}
            if et in ("giveaway", "takeaway"):
                pid = _safe_int(det.get("playerId") or det.get("player_id") or det.get("actorPlayerId"), 0)
                if pid == p:
                    if et == "giveaway":
                        gv_p += 1
                        _bump_zone_counter(gv_zone, det)
                        else:
                        tk_p += 1
                        _bump_zone_counter(tk_zone, det)
            elif et == "hit":
                pid = _safe_int(det.get("hittingPlayerId") or det.get("hitterPlayerId") or det.get("hitterId") or det.get("playerId"), 0)
                if pid == p:
                    hp_p += 1
                    _bump_zone_counter(hp_zone, det)
            elif et == "blocked-shot":
                if str(det.get("reason", "")).lower() == "teammate-blocked":
                                continue
                blk_pid = _safe_int(det.get("blockingPlayerId") or det.get("blockedByPlayerId") or det.get("blockerPlayerId") or det.get("blockerId") or det.get("playerId"), 0)
                if blk_pid == p:
                    bp_p += 1
                    _bump_zone_counter(bp_zone, det)
                shot_pid = _safe_int(det.get("shootingPlayerId") or det.get("shooterId") or det.get("shooter"), 0)
                if shot_pid == p:
                    sbp_p += 1
                    _bump_zone_counter(sbp_zone, det)

        # shots/goals crediting (pre-change snapshots)
        cb = credit_by_sec.get(int(s)) or {}
                    pre_home = list(cb.get("onice_home", ()))
                    pre_away = list(cb.get("onice_away", ()))
        for ev in events_by_sec.get(int(s), []):
            et = str(ev.get("type") or "").lower()
            if et not in SHOT_TYPES and et not in GOAL_TYPES:
                            continue
                        det = ev.get("details") or {}
            if et in BLOCK_TYPES and str(det.get("reason", "")).lower() == "teammate-blocked":
                            continue
            ev_side = _owner_side_for_event(det, game.home_team_id, game.away_team_id)
            if ev_side not in ("home", "away"):
                continue

            # corrected on-ice for goals
                        if et in GOAL_TYPES:
                corr_home, corr_away = _maybe_correct_onice_for_goal(
                    int(s), ev, set(pre_home), set(pre_away),
                    game.home_team_id, game.away_team_id,
                    team_onice_by_sec, orders_by_sec, shift_changes_by_sec,
                )
                            on_for = list(corr_home) if ev_side == "home" else list(corr_away)
                            on_opp = list(corr_away) if ev_side == "home" else list(corr_home)
                        else:
                            on_for = pre_home if ev_side == "home" else pre_away
                            on_opp = pre_away if ev_side == "home" else pre_home

            xg = float(det.get("xg", 0.0)) if det.get("xg") is not None else 0.0
                        if p in on_for:
                            Y["AF"] += 1
                if et in SHOT_ON_GOAL_TYPES or et in GOAL_TYPES:
                                Y["SF"] += 1
                                xgF += xg
                        elif p in on_opp:
                            Y["AA"] += 1
                if et in SHOT_ON_GOAL_TYPES or et in GOAL_TYPES:
                                Y["SA"] += 1
                                xgA += xg

                        if et in BLOCK_TYPES:
                blk_pid = det.get("blockingPlayerId") or det.get("blockedByPlayerId") or det.get("blockerPlayerId") or det.get("blockerId")
                            try:
                                blk_pid = int(blk_pid) if blk_pid is not None else None
                            except Exception:
                                blk_pid = None
                defender_side = None
                            if blk_pid is not None:
                                if blk_pid in pre_home:
                                    defender_side = "home"
                                elif blk_pid in pre_away:
                                    defender_side = "away"
                            if defender_side is None:
                    defender_side = ("away" if ev_side == "home" else "home")
                attacker_side = ("away" if defender_side == "home" else "home")
                if defender_side == side and p in (pre_home if side == "home" else pre_away):
                                Y["BF"] += 1
                if attacker_side == side and p in (pre_home if side == "home" else pre_away):
                                Y["BA"] += 1

                        if et in GOAL_TYPES:
                            if p in on_for:
                                Y["GF"] += 1
                            elif p in on_opp:
                                Y["GA"] += 1

        # co-presence (post evidence)
        our = onice_map.get(int(s), ())
        if p in our:
            for q in our:
                if q == p:
                    continue
                        overlap_with[q] += 1
                if credit_by_sec.get(int(s)):
                            event_copres_with[q]["any"] += 1
                    for k in ("SF", "SA", "GF", "GA"):
                        if (credit_by_sec[int(s)][side].get(k, 0) if int(s) in credit_by_sec else 0):
                            event_copres_with[q][k] += int(credit_by_sec[int(s)][side][k])
            for q in opp_map.get(int(s), ()):
                overlap_vs[q] += 1

    # True on-ice exposure inside the token horizon (t_token, t_end] (already clipped to window end).
    seconds_onice_in_h = sum(1 for s in range(int(start_target), int(end_target) + 1) if p in onice_map.get(int(s), ()))

    def build_weighted_list(overlap: Counter[int], seconds_i: int, ev_map: Dict[int, Counter]) -> Tuple[List[int], List[float], List[int], List[float], Dict[int, Counter]]:
                    if seconds_i <= 0:
            return [], [], [], [], {}
                    if seconds_i < SHORT_WIN_SEC:
                        share_min = SHORT_SHARE_MIN
                        sec_floor_eff = max(2, int(SHORT_SEC_FRAC * seconds_i))
                    else:
                        share_min = RAW_SHARE_MIN
                        sec_floor_eff = SEC_FLOOR_BASE
                    kept: List[Tuple[int, int, float]] = []
                    for q, ov_sec in overlap.items():
                        raw_share = ov_sec / float(seconds_i)
            if ov_sec >= sec_floor_eff and raw_share >= share_min:
                            kept.append((q, ov_sec, raw_share))
                    if not kept and seconds_i < SHORT_WIN_SEC and seconds_i >= 2 and TOPK_FALLBACK > 0 and overlap:
                        qbest, ovbest = max(overlap.items(), key=lambda kv: (kv[1], kv[0]))
                        if ovbest >= 2:
                            kept = [(qbest, ovbest, ovbest / float(seconds_i))]
        ids: List[int] = []
        secs: List[int] = []
        shares: List[float] = []
        w_raw: List[float] = []
                    for q, ov_sec, raw_share in kept:
            ids.append(int(q))
                        secs.append(int(ov_sec))
            shares.append(float(raw_share))
            w_raw.append(raw_share)
        ssum = sum(w_raw)
        ws = [float(w / ssum) for w in w_raw] if ssum > 0 else [float(w) for w in w_raw]
        ev_out = {q: ev_map.get(q, Counter()) for q, _, _ in kept}
        return ids, ws, secs, shares, ev_out

    ids_with, w_with, sec_with, share_with, ev_with = build_weighted_list(overlap_with, seconds_onice_in_h, event_copres_with)
    ids_vs, w_vs, sec_vs, share_vs, _ = build_weighted_list(overlap_vs, seconds_onice_in_h, {})

    return {
        "seconds_onice_in_h": int(seconds_onice_in_h),
        "xGF": float(xgF),
        "xGA": float(xgA),
        "GF": int(Y.get("GF", 0)),
        "GA": int(Y.get("GA", 0)),
        "SF": int(Y.get("SF", 0)),
        "SA": int(Y.get("SA", 0)),
        "AF": int(Y.get("AF", 0)),
        "AA": int(Y.get("AA", 0)),
        "BF": int(Y.get("BF", 0)),
        "BA": int(Y.get("BA", 0)),
        "giveaways_committed": int(gv_p),
        "takeaways_forced": int(tk_p),
        "hits_personal": int(hp_p),
        "blocks_personal": int(bp_p),
        "shots_blocked_personal": int(sbp_p),
        "giveaways_committed_oz": int(gv_zone.get("OZ", 0)),
        "giveaways_committed_nz": int(gv_zone.get("NZ", 0)),
        "giveaways_committed_dz": int(gv_zone.get("DZ", 0)),
        "takeaways_forced_oz": int(tk_zone.get("OZ", 0)),
        "takeaways_forced_nz": int(tk_zone.get("NZ", 0)),
        "takeaways_forced_dz": int(tk_zone.get("DZ", 0)),
        "hits_personal_oz": int(hp_zone.get("OZ", 0)),
        "hits_personal_nz": int(hp_zone.get("NZ", 0)),
        "hits_personal_dz": int(hp_zone.get("DZ", 0)),
        "blocks_personal_oz": int(bp_zone.get("OZ", 0)),
        "blocks_personal_nz": int(bp_zone.get("NZ", 0)),
        "blocks_personal_dz": int(bp_zone.get("DZ", 0)),
        "shots_blocked_personal_oz": int(sbp_zone.get("OZ", 0)),
        "shots_blocked_personal_nz": int(sbp_zone.get("NZ", 0)),
        "shots_blocked_personal_dz": int(sbp_zone.get("DZ", 0)),
        # POST-horizon overlap evidence (DO NOT use as PRE/CIN inputs)
        "post_teammates_onice_ids_w": "|".join(str(x) for x in ids_with),
        "post_teammates_onice_w": "|".join(str(x) for x in w_with),
        "post_teammates_onice_sec_w": "|".join(str(x) for x in sec_with),
        "post_teammates_onice_share_raw": "|".join(str(x) for x in share_with),
        "post_opponents_onice_ids_w": "|".join(str(x) for x in ids_vs),
        "post_opponents_onice_w": "|".join(str(x) for x in w_vs),
        "post_opponents_onice_sec_w": "|".join(str(x) for x in sec_vs),
        "post_opponents_onice_share_raw": "|".join(str(x) for x in share_vs),
        "post_with_event_GF": "|".join(str(int(ev_with.get(q, {}).get("GF", 0))) for q in ids_with),
        "post_with_event_GA": "|".join(str(int(ev_with.get(q, {}).get("GA", 0))) for q in ids_with),
        "post_with_event_SF": "|".join(str(int(ev_with.get(q, {}).get("SF", 0))) for q in ids_with),
        "post_with_event_SA": "|".join(str(int(ev_with.get(q, {}).get("SA", 0))) for q in ids_with),
    }


def generate_tokens_and_rows(
    game: ParsedGame,
    windows: List[Dict[str, Any]],
    team_onice_by_sec: List[Tuple[List[int], List[int]]],
    goalie_ids_by_sec: List[Dict[str, int]],
    events_by_sec: Dict[int, List[Dict[str, Any]]],
    orders_by_sec: Dict[int, List[Tuple[int, str]]],
    sso_by_sec: Dict[int, List[Tuple[int, str]]],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
    fo_meta: Dict[int, Dict[str, Any]],
    end_event_at: Dict[int, Dict[str, Any]],
    credit_by_sec: Dict[int, Dict[str, Any]],
    cum_home_goals: List[int],
    cum_away_goals: List[int],
    horizon_sec: int,
    min_swap: int,
    min_gap_sec: int,
    stable_sec: int,
    rc_require_stable: int,
    post_dwell_sec: int,
    mass_swap_suppress_threshold: int,
    max_tokens: int,
    mode: str,
    opp_change_tokens: int = 0,
    opp_cover_gap_sec: int = 4,
    player_name_map: Optional[Dict[int, str]] = None,
    player_pos_map: Optional[Dict[int, str]] = None,
    season: str = "",
    date: str = "",
    rinkid: Any = None,
    # Deprecated legacy knobs (ROSTER_CHANGE tokenization). Kept for call-site compatibility.
    matchup_stable_sec: int = 6,
    # --- STATE tokenization controls (outcome supervision) ---
    # --- STATE tokenization controls (outcome supervision) ---
    min_exposure_sec: int = 5,
    state_stable_sec: int = 2,
    state_chunk_by_h: int = 0,
) -> List[Dict[str, Any]]:
    """
    Build token rows for each (window, player) using:
    - ENTRY / WINDOW_START / EXIT tokens for selection/termination (coach process) supervision
    - STATE tokens for outcome supervision

    STATE tokens correspond to (mostly) piecewise-constant matchup states:
      state(s) = (teammate_set_excl_player, opponent_set, own_goalie_id, opp_goalie_id, strength_global)

    For STATE tokens, the effective target interval ends at the end of the current matchup-state run:
      Because targets are counted in (t_token, t_end] (inclusive of t_end), we cap at run_end (not run_end+1)
      so we do not include the first second of the next state.
      t_end = min(t_token + H, run_end, win_end_sec, seg_last_on)

    This guarantees that outcome supervision tokens do NOT mix different matchup states inside their horizon.
    Outcomes are computed over (t_token, t_end] and only while player is on-ice, using pre-change
    on-ice snapshots for each second.
    """
    player_name_map = player_name_map or {}
    player_pos_map = player_pos_map or {}

    by_sec_home = {s: set(team_onice_by_sec[s][0]) for s in range(len(team_onice_by_sec))}
    by_sec_away = {s: set(team_onice_by_sec[s][1]) for s in range(len(team_onice_by_sec))}

    # Precompute per-player completed shift length stats (used for fatigue proxy).
    horizon_local = len(team_onice_by_sec) - 2
    shift_stats_home = build_shift_length_stats(by_sec_home, horizon_local)
    shift_stats_away = build_shift_length_stats(by_sec_away, horizon_local)

    # "Seen roster" per side (skaters only) for cheap bench availability features
    home_seen: set = set()
    away_seen: set = set()
    for hh, aa in team_onice_by_sec:
        home_seen.update(int(x) for x in hh)
        away_seen.update(int(x) for x in aa)

    def score_at_start(start_sec: int) -> Tuple[int, int]:
        # goals BEFORE this second
        return cum_home_goals[start_sec], cum_away_goals[start_sec]

    def score_diff_for_side(side: str, start_sec: int) -> int:
        h, a = score_at_start(start_sec)
        return (h - a) if side == "home" else (a - h)

    def score_bucket(diff: int) -> str:
        if diff <= -2:
            return "trail_2p"
        if diff == -1:
            return "trail_1"
        if diff == 0:
            return "tied"
        if diff == 1:
            return "lead_1"
        return "lead_2p"

    def _tok_prio(tok_type: str) -> int:
        return int(TOKEN_PRIORITIES.get(str(tok_type).upper(), 999))

    def onice_elapsed_before_second(p: int, sec: int, onice_map: Dict[int, set]) -> int:
        t = sec - 1
        streak = 0
        while t >= 0 and p in onice_map.get(t, ()):
            streak += 1
            t -= 1
        return streak

    def last_shift_and_rest_before(p: int, sec: int, onice_map: Dict[int, set]) -> Tuple[int, int]:
        if sec <= 0 or p not in onice_map.get(sec, ()):
            return 0, 0
        curr_start = sec
        sec_period = period_of(sec)
        while curr_start - 1 >= 0 and period_of(curr_start - 1) == sec_period and p in onice_map.get(curr_start - 1, ()):
            curr_start -= 1
        t = curr_start - 1
        while t >= 0 and p not in onice_map.get(t, ()):
            t -= 1
        if t < 0:
            return 0, curr_start
        prev_end = t
        while t - 1 >= 0 and p in onice_map.get(t - 1, ()):
            t -= 1
        prev_start = t
        last_len = max(0, prev_end - prev_start + 1)
        rest_len = max(0, curr_start - (prev_end + 1))
        return int(last_len), int(rest_len)

    def _onice_sets_at(sec: int, side: str) -> Tuple[set, set]:
        ss = max(0, min(sec, len(team_onice_by_sec) - 1))
        h, a = team_onice_by_sec[ss]
        return (set(h), set(a)) if side == "home" else (set(a), set(h))

    def _goalie_ids_at(sec: int, side: str) -> Tuple[int, int]:
        ss = max(0, min(sec, len(goalie_ids_by_sec) - 1))
        g = goalie_ids_by_sec[ss]
        return (_safe_int(g.get("home", 0)), _safe_int(g.get("away", 0))) if side == "home" else (_safe_int(g.get("away", 0)), _safe_int(g.get("home", 0)))

    def _strength_team_at(sec: int, side: str) -> str:
        our, opp = _onice_sets_at(sec, side)
        og, tg = _goalie_ids_at(sec, side)
        return strength_for_team(len(our), len(opp), int(og != 0), int(tg != 0))

    # Per-second strength_global for STATE tokenization (must match container encoding).
    horizon_local_s = len(team_onice_by_sec) - 2
    strength_global_by_sec: List[str] = []
    for s in range(horizon_local_s + 1):
        h_ids, a_ids = team_onice_by_sec[max(0, min(s, horizon_local_s))]
        gh = _safe_int(goalie_ids_by_sec[max(0, min(s, len(goalie_ids_by_sec) - 1))].get("home", 0))
        ga = _safe_int(goalie_ids_by_sec[max(0, min(s, len(goalie_ids_by_sec) - 1))].get("away", 0))
        gh_present = 1 if gh != 0 else 0
        ga_present = 1 if ga != 0 else 0
        strength_global_by_sec.append(f"{len(h_ids)}v{len(a_ids)}_GH{gh_present}_GA{ga_present}")

    rows: List[Dict[str, Any]] = []
    m = str(mode or "both").lower().strip()

    # Token-local faceoff zone at/before each second (rink zone code)
    last_fo_zone_by_sec: Dict[int, str] = {}
    last_z = "flow"
    max_s = len(team_onice_by_sec) - 2
    for s in range(max_s + 1):
        if s in fo_meta:
            last_z = str((fo_meta.get(s) or {}).get("zone") or last_z)
        last_fo_zone_by_sec[s] = last_z

    stoppage_types_for_last_change = {
        "goal", "penalty", "icing", "offside", "stoppage", "puck-out-of-play",
        "goalie-stopped", "timeout", "challenge", "period_start", "faceoff",
    }

    stoppage_types_for_exit_reason = {
        "goal", "penalty", "icing", "offside", "stoppage", "puck-out-of-play",
        "goalie-stopped", "timeout", "challenge", "faceoff",
    }

    def _exit_reason_proxy(side: str, t_exit: int) -> str:
        """
        Best-effort proxy for why a shift ended at t_exit.
        Uses only info at/around the exit second; meant for causal supervision, not perfect labeling.
        """
        # Goal for/against at prior second is highly informative
        try:
            cb_prev = credit_by_sec.get(max(0, t_exit - 1)) or {}
            if int((cb_prev.get(side) or {}).get("GA", 0)) > 0:
                return "goal_against"
            if int((cb_prev.get(side) or {}).get("GF", 0)) > 0:
                return "goal_for"
                except Exception:
                    pass

        ev = end_event_at.get(t_exit)
        if isinstance(ev, dict):
            et = str(ev.get("type") or "").lower()
            if et in stoppage_types_for_exit_reason:
                # icing special case
                if et in {"stoppage", "goalie-stopped"}:
                    det = ev.get("details") or {}
                    sub_reason = str(det.get("reason") or det.get("stoppageReason") or "").lower()
                    if sub_reason == "icing":
                        return "icing"
                return et

        # strength regime change at exit second
        try:
            st0 = _strength_team_at(max(0, t_exit - 1), side)
            st1 = _strength_team_at(t_exit, side)
            if st0 != st1:
                return "strength_change"
        except Exception:
            pass

        return "line_change"

    def _bench_rest_bucket(rest_s: int) -> str:
        if rest_s <= 10:
            return "rest_0_10"
        if rest_s <= 30:
            return "rest_11_30"
        if rest_s <= 60:
            return "rest_31_60"
        return "rest_61p"

    for w in windows:
        win_id = w["window_id"]
        start_s, end_s = int(w["start_sec"]), int(w["end_sec"])
        if end_s <= start_s:
            continue

        for side in ("home", "away"):
            onice_map = by_sec_home if side == "home" else by_sec_away
            team_id = game.home_team_id if side == "home" else game.away_team_id

            seen: Counter[int] = Counter()
            secs_on_by_player: Dict[int, List[int]] = defaultdict(list)
                for s in range(start_s, end_s):
                for p in onice_map.get(s, ()):
                    pid = int(p)
                    seen[pid] += 1
                    secs_on_by_player[pid].append(int(s))
            if not seen:
                continue

            for p, sec_i in seen.items():
                if int(sec_i) <= 0:
                    continue
                secs_on = secs_on_by_player.get(int(p), [])
                if not secs_on:
                    continue

                secs_on_sorted = sorted(secs_on)
                segments: List[Tuple[int, int]] = []
                seg_start = secs_on_sorted[0]
                prev_s = secs_on_sorted[0]
                for s in secs_on_sorted[1:]:
                    if s == prev_s + 1:
                        prev_s = s
                        continue
                    segments.append((int(seg_start), int(prev_s)))
                    seg_start = s
                    prev_s = s
                segments.append((int(seg_start), int(prev_s)))

                # Build candidate tokens across all segments, then assign deterministic token_idx per player per window.
                token_specs: List[Dict[str, Any]] = []
                for seg_idx, (seg_s, seg_last_on) in enumerate(segments):
                    t_entry = int(seg_s)
                    t_exit = int(seg_last_on + 1)  # first second off after being on (may equal end_s)

                    # Compute the player's true segment start in game time (may be before this container).
                    seg_true_start = int(seg_s)
                    tt = int(seg_s) - 1
                    while tt >= 0 and p in onice_map.get(tt, ()):
                        seg_true_start = int(tt)
                        tt -= 1

                    # ENTRY: only if the player actually entered within this window
                    if int(seg_true_start) == int(seg_s):
                        token_specs.append({"t": t_entry, "type": "ENTRY", "seg_idx": seg_idx, "seg_s": int(seg_s), "seg_true_start": int(seg_true_start), "seg_last_on": int(seg_last_on), "t_exit": int(t_exit)})

                    # WINDOW_START: only if player is on at window start AND was already on before the window started.
                    # This avoids emitting both WINDOW_START and ENTRY at the same second.
                    if p in onice_map.get(int(start_s), ()) and int(start_s) > 0 and (p in onice_map.get(int(start_s) - 1, ())):
                        token_specs.append({"t": int(start_s), "type": "WINDOW_START", "seg_idx": seg_idx, "seg_s": int(seg_s), "seg_true_start": int(seg_true_start), "seg_last_on": int(seg_last_on), "t_exit": int(t_exit)})
                    if int(start_s) <= int(t_exit) <= int(end_s):
                        token_specs.append({"t": int(t_exit), "type": "EXIT", "seg_idx": seg_idx, "seg_s": int(seg_s), "seg_true_start": int(seg_true_start), "seg_last_on": int(seg_last_on), "t_exit": int(t_exit)})

                    # -------------------- STATE tokens (outcome supervision) --------------------
                    # Build matchup "state runs" inside this player's on-ice segment, then emit STATE tokens
                    # at run starts (and every H seconds for long runs).

                    def _state_key(sec: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int, str]:
                        our, opp = _onice_sets_at(int(sec), side)
                        tm = tuple(sorted(int(x) for x in (our - {int(p)})))
                        op = tuple(sorted(int(x) for x in opp))
                        og, tg = _goalie_ids_at(int(sec), side)
                        sg = strength_global_by_sec[max(0, min(int(sec), len(strength_global_by_sec) - 1))]
                        return tm, op, int(og), int(tg), str(sg)

                    def _state_change_reason(k0: Tuple, k1: Tuple) -> str:
                        tm0, op0, og0, tg0, sg0 = k0
                        tm1, op1, og1, tg1, sg1 = k1
                        parts: List[str] = []
                        if tm0 != tm1:
                            parts.append("team")
                        if op0 != op1:
                            parts.append("opp")
                        if int(og0) != int(og1):
                            parts.append("own_goalie")
                        if int(tg0) != int(tg1):
                            parts.append("opp_goalie")
                        if str(sg0) != str(sg1):
                            parts.append("strength")
                        return "+".join(parts) if parts else "none"

                    # 1) Raw maximal runs of constant state_key (optionally debounced by state_stable_sec).
                    runs: List[Dict[str, Any]] = []
                    if int(seg_s) <= int(seg_last_on):
                        s2 = int(seg_s)
                        cur_k = _state_key(int(s2))
                        run_start = int(s2)
                        s2 += 1
                        while s2 <= int(seg_last_on):
                            k = _state_key(int(s2))
                            if k == cur_k:
                                s2 += 1
                                continue
                            if int(state_stable_sec) > 1:
                                ok = True
                                for t3 in range(int(s2), min(int(seg_last_on), int(s2) + int(state_stable_sec) - 1) + 1):
                                    if _state_key(int(t3)) != k:
                                        ok = False
                                break
                                if not ok:
                                    s2 += 1
                                    continue
                            runs.append({"start": int(run_start), "end": int(s2 - 1), "key": cur_k})
                            cur_k = k
                            run_start = int(s2)
                            s2 += 1
                        runs.append({"start": int(run_start), "end": int(seg_last_on), "key": cur_k})

                    def _dur(rr: Dict[str, Any]) -> int:
                        return int(rr["end"]) - int(rr["start"]) + 1

                    # 2) Merge micro-runs (< min_exposure_sec), prefer forward merge into the next run if it exists.
                    changed = True
                    while changed and len(runs) >= 2:
                        changed = False
                        # merge adjacent identical keys (defensive)
                        merged: List[Dict[str, Any]] = []
                        for rr in runs:
                            if merged and merged[-1]["key"] == rr["key"] and int(rr["start"]) <= int(merged[-1]["end"]) + 1:
                                merged[-1]["end"] = int(rr["end"])
                        else:
                                merged.append(dict(rr))
                        runs = merged

                        for i, rr in enumerate(runs):
                            if _dur(rr) >= int(min_exposure_sec):
                                continue
                            if i < len(runs) - 1:
                                # forward merge (extend next run backward)
                                runs[i + 1]["start"] = int(rr["start"])
                                runs.pop(i)
                                changed = True
                                break
                            if i > 0:
                                # backward merge (extend prev run forward)
                                runs[i - 1]["end"] = int(rr["end"])
                                runs.pop(i)
                                changed = True
                                break

                    # 3) Emit STATE tokens from kept runs (no min-gap throttling; runs already constant).
                    for run_id, rr in enumerate(runs):
                        run_start = int(rr["start"])
                        run_end = int(rr["end"])
                        run_dur = int(run_end - run_start + 1)
                        if run_dur < int(min_exposure_sec):
                            continue
                        next_reason = "segment_end"
                        if run_id < len(runs) - 1:
                            next_reason = _state_change_reason(rr["key"], runs[run_id + 1]["key"])
                        # Merge rule (sanity): if the run starts at an ENTRY/WINDOW_START token time,
                        # do NOT emit a duplicate STATE row at the same second. Instead, attach the run metadata
                        # to that anchor row and set is_outcome_token=1 later.
                        anchor_at_start: Optional[Dict[str, Any]] = None
                        for sp in token_specs:
                            if int(sp.get("t", -1)) == int(run_start) and str(sp.get("type") or "").upper() in ("ENTRY", "WINDOW_START"):
                                anchor_at_start = sp
                                break
                        if anchor_at_start is not None:
                            anchor_at_start["run_id"] = int(run_id)
                            anchor_at_start["run_start"] = int(run_start)
                            anchor_at_start["run_end"] = int(run_end)
                            anchor_at_start["run_duration"] = int(run_dur)
                            anchor_at_start["state_changed_reason"] = str(next_reason)
                            # Optional chunking: emit additional physics slices inside the same run at +H, +2H...
                            if int(state_chunk_by_h) != 0 and int(horizon_sec) > 0:
                                tt_emit = int(run_start) + int(horizon_sec)
                                while int(tt_emit) <= int(run_end):
                                    token_specs.append(
                                        {
                                            "t": int(tt_emit),
                                            "type": "STATE",
                                            "seg_idx": seg_idx,
                                            "seg_s": int(seg_s),
                                            "seg_true_start": int(seg_true_start),
                                            "seg_last_on": int(seg_last_on),
                                            "t_exit": int(t_exit),
                                            "run_id": int(run_id),
                                            "run_start": int(run_start),
                                            "run_end": int(run_end),
                                            "run_duration": int(run_dur),
                                            "state_changed_reason": str(next_reason),
                                        }
                                    )
                                    tt_emit += int(horizon_sec)
                                continue
                            
                        # Default: emit exactly ONE STATE token per real regime (run_start).
                        # Optional: chunk long runs into multiple rows every H seconds (for numerical convenience).
                        tt_emit = int(run_start)
                        while True:
                            token_specs.append(
                                {
                                    "t": int(tt_emit),
                                    "type": "STATE",
                                    "seg_idx": seg_idx,
                                    "seg_s": int(seg_s),
                                    "seg_true_start": int(seg_true_start),
                                    "seg_last_on": int(seg_last_on),
                                    "t_exit": int(t_exit),
                                    "run_id": int(run_id),
                                    "run_start": int(run_start),
                                    "run_end": int(run_end),
                                    "run_duration": int(run_dur),
                                    "state_changed_reason": str(next_reason),
                                }
                            )
                            if int(state_chunk_by_h) != 0 and int(horizon_sec) > 0 and (int(tt_emit) + int(horizon_sec) <= int(run_end)):
                                tt_emit += int(horizon_sec)
                                continue
                            break

                if not token_specs:
                    continue

                # Deduplicate exact duplicates of (t, type) and sort deterministically.
                seen_tt: Dict[Tuple[int, str], Dict[str, Any]] = {}
                uniq_specs: List[Dict[str, Any]] = []
                for spec in token_specs:
                    key = (int(spec["t"]), str(spec["type"]).upper())
                    spec["type"] = str(spec["type"]).upper()
                    if key in seen_tt:
                        continue
                    seen_tt[key] = spec
                    uniq_specs.append(spec)
                kept_specs = sorted(uniq_specs, key=lambda d: (int(d["t"]), _tok_prio(str(d["type"]))))

                for token_idx, spec in enumerate(kept_specs):
                    t_tok = int(spec["t"])
                    tok_type = str(spec["type"])
                    seg_idx = int(spec["seg_idx"])
                    seg_s = int(spec["seg_s"])
                    seg_true_start = int(spec.get("seg_true_start", seg_s))
                    seg_last_on = int(spec["seg_last_on"])
                    t_exit = int(spec["t_exit"])
                    # For EXIT tokens, keep t_token at seg_exit (first off-ice second),
                    # but compute "context-at-token" covariates from the last on-ice second.
                    t_context = int(seg_last_on) if str(tok_type).upper() == "EXIT" else int(t_tok)
                    # For PRE roster snapshots (CIN inputs), use decision-time roster at t_token for most tokens,
                    # but for EXIT tokens use the just-before-exit roster at t_context (last on-ice second).
                    t_roster = int(t_context) if str(tok_type).upper() == "EXIT" else int(t_tok)

                    # Horizon/exposure contract:
                    # - All non-EXIT tokens must NOT include any off-ice seconds in their target interval.
                    #   Because targets are counted in (t_token, t_end] (inclusive of t_end), we cap at seg_last_on.
                    # - STATE tokens are additionally capped by the end of their constant-state run (run_end),
                    #   so their targets do not mix different matchup states.
                    has_run = spec.get("run_end") is not None
                    if has_run:
                        run_end = int(spec.get("run_end", seg_last_on))
                        if int(state_chunk_by_h) != 0 and str(tok_type).upper() == "STATE":
                            # Chunked interior physics rows use H, still capped at run_end.
                            t_end = min(int(end_s), int(t_tok) + int(horizon_sec), int(seg_last_on), int(run_end))
                            else:
                            # One row per real regime (and merged ENTRY/WINDOW_START-at-run-start rows) uses full run.
                            t_end = min(int(end_s), int(seg_last_on), int(run_end))
                    elif str(tok_type).upper() == "EXIT":
                        t_end = min(int(end_s), int(t_tok) + int(horizon_sec))
                    else:
                        t_end = min(int(end_s), int(t_tok) + int(horizon_sec), int(seg_last_on))
                    seconds_token = int(t_end) - int(t_tok)
                    # Keep STATE rows even if seconds_token==0 (e.g., a 1-second state right before exit).
                    # These are useful for logging state transitions; they should be excluded from outcome loss via is_outcome_token.
                    if seconds_token <= 0 and str(tok_type).upper() not in ("EXIT", "STATE"):
                        continue

                    our_set, opp_set = _onice_sets_at(int(t_roster), side)
                    teammates_ids = sorted(int(x) for x in (our_set - {p}))
                    opponents_ids = sorted(int(x) for x in opp_set)
                    own_goalie_id, opp_goalie_id = _goalie_ids_at(int(t_roster), side)

                    # CIN-friendly fixed roster slots (skaters only). Use up to 6 to handle EN 6v5.
                    team_skaters_sorted = sorted(int(x) for x in our_set)
                    opp_skaters_sorted = sorted(int(x) for x in opp_set)
                    num_team_skaters = int(len(team_skaters_sorted))
                    num_opp_skaters = int(len(opp_skaters_sorted))

                    def _slots(arr: List[int], k: int) -> List[int]:
                        out = arr[:k]
                        if len(out) < k:
                            out = out + [0] * (k - len(out))
                        return out

                    team_slots = _slots(team_skaters_sorted, 6)
                    opp_slots = _slots(opp_skaters_sorted, 6)

                    strength_g = str(w.get("strength_global"))
                    strength_team = _strength_team_at(int(t_context), side)

                    outcomes = aggregate_features_for_token(
                        game=game,
                        by_sec_home=by_sec_home,
                        by_sec_away=by_sec_away,
                        events_by_sec=events_by_sec,
                        credit_by_sec=credit_by_sec,
                        team_onice_by_sec=team_onice_by_sec,
                        orders_by_sec=orders_by_sec,
                        shift_changes_by_sec=shift_changes_by_sec,
                        side=side,
                        p=int(p),
                        win_start=int(start_s),
                        win_end=int(end_s),
                        t_token=int(t_tok),
                        t_end=int(t_end),
                    )
                    seconds_onice_in_h = int(outcomes.get("seconds_onice_in_h", 0) or 0)
                    score_diff_tok = score_diff_for_side(side, t_context)

                    sim_features = {
                        "period": int(period_of(t_context)),
                        "clock_s": int(clock_s_at(t_context)),
                        "home_away": bool(side == "home"),
                        "long_change": int(is_long_change(t_context)),
                        "strength_team": strength_team,
                        # score_diff at "context time" (for EXIT tokens, this is just-before-exit)
                        "score_diff": int(score_diff_for_side(side, t_context)),
                    }
                    # "Last meaningful event" context (exclude shift-change; must be close enough to matter).
                    last_et, last_es, last_dt, last_lb, last_z, last_owner_tid = last_non_shift_event_before_adaptive(
                        t_context=int(t_context),
                        events_by_sec=events_by_sec,
                        lookback_primary_sec=6,
                        lookback_extended_sec=10,
                    )
                    sim_features.update(
                        {
                            "last_event_type": str(last_et),
                            "time_since_last_event_s": int(last_dt),
                            "last_event_sec": int(last_es),
                            "last_event_lookback_s": int(last_lb),
                            "last_event_zone": str(last_z),
                            "last_event_owner_team_id": int(last_owner_tid),
                        }
                    )

                    # Shift-change reasoning proxy (fatigue + situation)
                    shift_elapsed = int(onice_elapsed_before_second(p, int(t_context), onice_map))
                    mean_len, sd_len, n_prev = shift_mean_sd_before(
                        shift_stats_home if side == "home" else shift_stats_away,
                        int(p),
                        int(t_context),
                    )
                    # Token-local goalie pulled flags & transitions (high confidence from per-second snapshots)
                    # IMPORTANT: goalie pulled means the goalie is not on-ice (goalie_id == 0), regardless of skater count.
                    # (You can be shorthanded and still pull the goalie; then skaters may be 5, goalie_id==0.)
                    own_pulled = int(1 if int(own_goalie_id) == 0 else 0)
                    opp_pulled = int(1 if int(opp_goalie_id) == 0 else 0)
                    own_prev_pulled = 0
                    opp_prev_pulled = 0
                    if int(t_context) - 1 >= 0:
                        prev_our, prev_opp = _onice_sets_at(int(t_context) - 1, side)
                        prev_own_gid, prev_opp_gid = _goalie_ids_at(int(t_context) - 1, side)
                        own_prev_pulled = int(1 if int(prev_own_gid) == 0 else 0)
                        opp_prev_pulled = int(1 if int(prev_opp_gid) == 0 else 0)
                    own_pull_trans = int(1 if own_prev_pulled != own_pulled else 0)
                    opp_pull_trans = int(1 if opp_prev_pulled != opp_pulled else 0)

                    # Period boundary (very high confidence shift termination context)
                    clk = int(clock_s_at(int(t_context)))
                    is_period_boundary = int(1 if (clk <= 2 or clk >= (SECONDS_PER_PERIOD - 2)) else 0)

                    # Boundary-only container break context (helps detect strength-change driven swaps cleanly)
                    boundary_prev_break_type = str(w.get("start_prev_break_type") or "") if int(t_tok) == int(start_s) else ""
                    sim_features.update(
                        {
                            "shift_elapsed_s": int(shift_elapsed),
                            "shift_prev_n": int(n_prev),
                            "shift_prev_mean_len_s": float(mean_len) if mean_len is not None else None,
                            "shift_prev_sd_len_s": float(sd_len) if sd_len is not None else None,
                        }
                    )
                    bundle = reason_proxy_bundle(
                        token_type=str(tok_type),
                        t_token=int(t_tok),
                        t_context=int(t_context),
                        strength_team=str(strength_team),
                        score_diff=int(score_diff_for_side(side, t_context)),
                        last_event_type=str(last_et),
                        last_event_owner_team_id=int(last_owner_tid),
                        our_team_id=int(team_id),
                        last_event_zone=str(last_z),
                        current_shift_elapsed_s=int(shift_elapsed),
                        mean_shift_len_s=mean_len,
                        sd_shift_len_s=sd_len,
                        position_code=str(player_pos_map.get(int(p)) or ""),
                        is_period_boundary=int(is_period_boundary),
                        own_goalie_pulled=int(own_pulled),
                        opp_goalie_pulled=int(opp_pulled),
                        own_goalie_pull_transition=int(own_pull_trans),
                        opp_goalie_pull_transition=int(opp_pull_trans),
                        boundary_prev_break_type=str(boundary_prev_break_type),
                    )
                    # Focus on EXIT for "why did he leave", but expose change-token proxy too.
                    lbl = str(bundle.get("label") or "na")
                    conf = float(bundle.get("confidence") or 0.0)
                    sim_features["exit_reason_proxy_pre"] = lbl if str(tok_type).upper() == "EXIT" else "na"
                    sim_features["change_reason_proxy_pre"] = (
                        lbl if str(tok_type).upper() in ("ENTRY", "EXIT", "STATE") else "na"
                    )
                    sim_features["exit_reason_conf_pre"] = float(conf) if str(tok_type).upper() == "EXIT" else 0.0
                    sim_features["change_reason_conf_pre"] = float(conf) if str(tok_type).upper() in ("ENTRY", "EXIT", "STATE") else 0.0
                    # Evidence flags (always safe as PRE; helps you filter to high-confidence subsets)
                    for fk in (
                        "is_special_teams","is_after_faceoff","is_after_stoppage","is_after_icing","is_after_goal","is_after_penalty",
                        "is_period_boundary","own_goalie_pulled","opp_goalie_pulled","own_goalie_pull_transition","opp_goalie_pull_transition","boundary_prev_break_type",
                        "zone_O","zone_D","zone_N","score_big","fatigue_z","shift_elapsed_s","shift_mean_s",
                    ):
                        if fk in bundle:
                            sim_features[f"reason_{fk}"] = bundle[fk]

                    last_len, rest_gap = last_shift_and_rest_before(p, int(t_context), onice_map)
                    train_features = dict(sim_features)
                    train_features.update(
                        {
                            "last_shift_len_s": int(last_len),
                            "time_since_last_shift_s": int(rest_gap),
                            "entry_offset_s": int(int(t_context) - int(seg_s)),
                        }
                    )

                    # Exit hazard targets for tokens inside this segment (EXIT tokens have 0).
                    exit_time_s = int(max(0, int(t_exit) - int(t_tok)))
                    exit_event_within_h = int(1 if (exit_time_s > 0 and exit_time_s <= int(seconds_token)) else 0)
                    exit_time_to_exit_s = int(exit_time_s) if exit_event_within_h else int(max(0, seconds_token))

                    # Replacement labeling for EXIT token: entrants on player's team at t_exit vs t_exit-1
                    replacement1_id = 0
                    replacement2_id = 0
                    replacement_count = 0
                    if str(tok_type).upper() == "EXIT":
                        t_prev = max(int(start_s), min(int(t_tok) - 1, int(end_s) - 1))
                        t_next = max(int(start_s), min(int(t_tok), int(end_s)))
                        if 0 <= t_prev < len(team_onice_by_sec) and 0 <= t_next < len(team_onice_by_sec):
                            home_prev, away_prev = team_onice_by_sec[t_prev]
                            home_next, away_next = team_onice_by_sec[t_next]
                            prev_team = set(home_prev) if side == "home" else set(away_prev)
                            next_team = set(home_next) if side == "home" else set(away_next)
                            entrants = sorted(int(x) for x in (next_team - prev_team))
                            replacement_count = int(len(entrants))
                            if len(entrants) >= 1:
                                replacement1_id = int(entrants[0])
                            if len(entrants) >= 2:
                                replacement2_id = int(entrants[1])

                    row = {
                        "season": season,
                        "date": date,
                        "gamePk": int(game.gamePk),
                        "teamId": int(team_id),
                        "team_side": side,
                        "window_id": win_id,
                        "strength_global": strength_g,
                        "playerId": int(p),
                        "token_idx": int(token_idx),
                        "token_type": str(tok_type),
                        "rc_side": "na",
                        "t_token": int(t_tok),
                        "t_context": int(t_context),
                        "t_end": int(t_end),
                        "seconds_token": int(seconds_token),
                        "seconds_onice_in_h": int(seconds_onice_in_h),
                        "is_outcome_token": int(1 if (has_run and int(seconds_token) > 0) else 0),
                        "is_selection_token": int(1 if str(tok_type).upper() in ("ENTRY", "WINDOW_START") else 0),
                        "is_hazard_token": int(1 if str(tok_type).upper() == "EXIT" else 0),
                        "positionCode": (player_pos_map.get(int(p)) or ""),
                        "playerName": (player_name_map.get(int(p)) or ""),
                        # segment index (useful for joining/diagnostics; keep lightweight)
                        "seg_idx": int(seg_idx),
                        # STATE run metadata (audit-friendly; only filled for STATE tokens)
                        "run_id": (int(spec.get("run_id")) if has_run and spec.get("run_id") is not None else None),
                        "run_start": (int(spec.get("run_start")) if has_run and spec.get("run_start") is not None else None),
                        "run_end": (int(spec.get("run_end")) if has_run and spec.get("run_end") is not None else None),
                        "run_duration": (int(spec.get("run_duration")) if has_run and spec.get("run_duration") is not None else None),
                        "state_changed_reason": (str(spec.get("state_changed_reason") or "") if has_run else None),
                        # How long this exact state persists starting at t_token (inclusive), in seconds.
                        # This is the quantity you want when micro-changes exist: a 2-second “reality” gets persist=2.
                        "state_persist_sec": (int(int(spec.get("run_end")) - int(t_tok) + 1) if has_run and spec.get("run_end") is not None else None),
                        "bench_rest_bucket": _bench_rest_bucket(int(rest_gap)) if str(tok_type).upper() == "ENTRY" else "na",
                        "entry_after_faceoff_flag": int(1 if (str(tok_type).upper() == "ENTRY" and int(t_tok) in fo_meta) else 0),
                        # bench availability pressure (sim-safe; derived from roster availability at t_token)
                        "bench_size_at_t": int(len((home_seen if side == "home" else away_seen) - set(our_set))),
                        # PRE roster at decision time (CIN inputs) - fixed slots only
                        "num_team_skaters": num_team_skaters,
                        "num_opp_skaters": num_opp_skaters,
                        "team_slot1": int(team_slots[0]),
                        "team_slot2": int(team_slots[1]),
                        "team_slot3": int(team_slots[2]),
                        "team_slot4": int(team_slots[3]),
                        "team_slot5": int(team_slots[4]),
                        "team_slot6": int(team_slots[5]),
                        "opp_slot1": int(opp_slots[0]),
                        "opp_slot2": int(opp_slots[1]),
                        "opp_slot3": int(opp_slots[2]),
                        "opp_slot4": int(opp_slots[3]),
                        "opp_slot5": int(opp_slots[4]),
                        "opp_slot6": int(opp_slots[5]),
                        "own_goalie_id": int(own_goalie_id),
                        "opp_goalie_id": int(opp_goalie_id),
                        # exit supervision targets (use for selection/exit heads; censored with seconds_token)
                        "exit_event_within_h": int(exit_event_within_h),
                        "exit_time_to_exit_s": int(exit_time_to_exit_s),
                        "exit_reason_proxy": _exit_reason_proxy(side, int(t_exit)) if str(tok_type).upper() == "EXIT" else "na",
                        "replacement1_id": int(replacement1_id),
                        "replacement2_id": int(replacement2_id),
                        "replacement_count": int(replacement_count),
                        # Use true on-ice exposure (not just horizon length) for TOI offsets.
                        "offset_log_toi": float(math.log(max(1, int(seconds_onice_in_h)))),
                        "rinkid": rinkid,
                        **outcomes,
                    }

                    # Carry window/container instruments into every token row (prefixed win_*)
                    # These are decision-time instruments (deployment signals) and safe as PRE inputs.
                    win_prev_type = str(w.get("start_prev_break_type") or "")
                    win_prev_team = w.get("start_prev_break_team_id")
                    win_fo_zone = str(w.get("fo_zone") or "flow")
                    # home last change opportunity (rule-based; conservative)
                    win_home_last_change = False
                    try:
                        if side == "home":
                            spbt = win_prev_type.lower()
                            fo_expected = (spbt in stoppage_types_for_last_change)
                            excluded = spbt in {"strength", "hard_cap", "flow", ""}
                            win_home_last_change = bool(fo_expected and not excluded and win_fo_zone.lower() != "flow")
                            # If home committed icing, cannot change
                            if spbt == "icing" and win_prev_team is not None and int(win_prev_team) == int(game.home_team_id):
                                win_home_last_change = False
            except Exception:
                        win_home_last_change = False
                    row.update(
                        {
                            "win_start_sec": int(w.get("start_sec", 0) or 0),
                            "win_home_goalie_present": int(w.get("home_goalie_present", 0) or 0),
                            "win_away_goalie_present": int(w.get("away_goalie_present", 0) or 0),
                            "win_home_goalie_pulled": int(w.get("home_goalie_pulled", 0) or 0),
                            "win_away_goalie_pulled": int(w.get("away_goalie_pulled", 0) or 0),
                            "win_fo_zone": win_fo_zone,
                            "win_fo_won_team_id": int(w.get("fo_won_team_id") or 0),
                            "win_start_prev_break_type": win_prev_type,
                            "win_start_prev_break_team_id": int(win_prev_team or 0),
                            "win_start_prev_break_subtype": str(w.get("start_prev_break_subtype") or "none") or "none",
                            "win_media_timeout_start": int(w.get("media_timeout_start", 0) or 0),
                            "win_home_last_change_opportunity": int(1 if win_home_last_change else 0),
                            # token-local (at/before t_token) last faceoff zone
                            "tok_last_fo_zone": str(last_fo_zone_by_sec.get(int(t_context), "flow")),
                            # audit helper: distinguish boundary-driven churn vs within-window roster evolution
                            "is_boundary_second": int(1 if int(t_tok) == int(w.get("start_sec", 0) or 0) else 0),
                        }
                    )

                    if m in ("sim", "both"):
                        for k, v in sim_features.items():
                            row[f"sim_{k}"] = v
                    if m in ("train", "both"):
                        for k, v in train_features.items():
                            row[f"train_{k}"] = v

                    rows.append(row)

    # Deterministic global ordering for auditability & train/test split discipline.
    rows.sort(
        key=lambda r: (
            str(r.get("window_id") or ""),
            int(r.get("playerId") or 0),
            int(r.get("t_token") or 0),
            int(TOKEN_PRIORITIES.get(str(r.get("token_type") or "").upper(), 999)),
        )
    )
    return rows


def extract_roster_change_token_times(
    *,
    windows: List[Dict[str, Any]],
    by_sec_home: Dict[int, set],
    by_sec_away: Dict[int, set],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
    min_swap: int,
    min_gap_sec: int,
    stable_sec: int,
    rc_require_stable: int,
    post_dwell_sec: int,
    mass_swap_suppress_threshold: int,
) -> Dict[Tuple[str, int, str], List[int]]:
    """
    Fast extractor: compute kept ROSTER_CHANGE token times per (window_id, playerId, team_side),
    using the exact same debounce/min-gap/mass-swap rules as generate_tokens_and_rows, but without
    computing any outcomes or writing full rows.
    """
    out: Dict[Tuple[str, int, str], List[int]] = defaultdict(list)
    mass_swap_th = max(0, int(mass_swap_suppress_threshold))
    stable_sec_i = int(stable_sec)
    require_stable = int(rc_require_stable) != 0
    post_dwell_i = max(0, int(post_dwell_sec))

    for w in windows:
        win_id = str(w.get("window_id") or "")
        start_s = int(w.get("start_sec") or 0)
        end_s = int(w.get("end_sec") or 0)
        if end_s <= start_s:
                                continue

        for side in ("home", "away"):
            onice_map = by_sec_home if side == "home" else by_sec_away
            # collect players who appear onice at least once in window
            players_in_window: set = set()
            for s in range(int(start_s), int(end_s)):
                players_in_window.update(int(x) for x in (onice_map.get(int(s)) or set()))
            if not players_in_window:
                            continue

            for p in sorted(players_in_window):
                # seconds on-ice for this player in the window
                # Window bounds are treated as half-open [start_sec, end_sec) for token membership.
                secs_on = [s for s in range(int(start_s), int(end_s)) if p in (onice_map.get(int(s)) or set())]
                if not secs_on:
                    continue
                secs_on.sort()

                # build contiguous segments
                segments: List[Tuple[int, int]] = []
                seg_start = secs_on[0]
                prev_s = secs_on[0]
                for s in secs_on[1:]:
                    if s == prev_s + 1:
                        prev_s = s
                        continue
                    segments.append((int(seg_start), int(prev_s)))
                    seg_start = s
                    prev_s = s
                segments.append((int(seg_start), int(prev_s)))

                kept_all: List[int] = []
                for seg_s, seg_last_on in segments:
                    prev_team = set(onice_map.get(int(seg_s), set())) - {p}

                    def stable_team_at(ss: int) -> Optional[set]:
                        if stable_sec_i <= 1:
                            return set(onice_map.get(int(ss), set())) - {p}
                        last = None
                        for tt in range(int(ss), int(ss) + int(stable_sec_i)):
                            if tt > int(seg_last_on):
            return None
                            if p not in (onice_map.get(int(tt), set()) or set()):
        return None
                            cur = set(onice_map.get(int(tt), set())) - {p}
                            if last is None:
                                last = cur
                            elif cur != last:
                                return None
                        return last

                    s = int(seg_s) + 1
                    candidates: List[int] = []
                    while s <= int(seg_last_on):
                        if p not in (onice_map.get(int(s), set()) or set()):
                            s += 1
                        continue
                        sc = shift_changes_by_sec.get(int(s)) or {}
                        explicit_shift = bool(sc.get(f"{side}_in") or sc.get(f"{side}_out"))
                        inferred_shift = bool((onice_map.get(int(s), set()) or set()) != (onice_map.get(int(s) - 1, set()) or set()))
                        if not (explicit_shift or inferred_shift):
                            s += 1
                        continue

                        if mass_swap_th > 0 and int(s) - 1 >= 0:
                            prev_on = set(onice_map.get(int(s) - 1, set()) or set())
                            now_on = set(onice_map.get(int(s), set()) or set())
                            swap_count = len(now_on - prev_on) + len(prev_on - now_on)
                            if int(swap_count) >= int(mass_swap_th):
                                s += 1
                        continue

                        team_now = set(onice_map.get(int(s), set()) or set()) - {p}
                        if len(team_now.symmetric_difference(prev_team)) < int(min_swap):
                            s += 1
                        continue

                        emit_t = int(s)
                        if stable_sec_i > 1:
                            found = False
                            for ss in range(int(s), int(seg_last_on) + 1):
                                st = stable_team_at(int(ss))
                                if st is None:
                        continue
                                if post_dwell_i > 0:
                                    ok_pd = True
                                    for tt in range(int(ss), int(ss) + int(post_dwell_i) + 1):
                                        if tt > int(seg_last_on) or p not in (onice_map.get(int(tt), set()) or set()):
                                            ok_pd = False
                                            break
                                    if not ok_pd:
                        continue
                                if len(set(st).symmetric_difference(prev_team)) >= int(min_swap):
                                    emit_t = int(ss)
                                    team_now = set(st)
                                    found = True
                                    break
                            if not found:
                                if require_stable:
                                    s += 1
                                continue
                                emit_t = int(s)
                                team_now = set(onice_map.get(int(s), set()) or set()) - {p}

                        candidates.append(int(emit_t))
                        prev_team = set(team_now)
                        s = int(emit_t) + 1

                    # min-gap between RC tokens within this segment/player
                    candidates = sorted(dict.fromkeys(int(x) for x in candidates))
                    kept: List[int] = []
                    last_kept = None
                    for tt in candidates:
                        if last_kept is None or int(tt) - int(last_kept) >= int(min_gap_sec):
                            kept.append(int(tt))
                            last_kept = int(tt)
                    kept_all.extend(kept)

                kept_all = sorted(dict.fromkeys(int(x) for x in kept_all))
                if kept_all:
                    out[(win_id, int(p), side)] = kept_all

    return out


# -------------------- Main --------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build window containers and tokenized player rows (CSV-only).")
    ap.add_argument("--in", dest="in_path", required=True, help="Path to pbp_onice_<gamePk>.json")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--hard_cap_sec", type=int, default=0, help="Force container breaks every N seconds (0 = disabled)")
    ap.add_argument("--horizon-sec", type=int, default=20, help="Token horizon in seconds (default 20)")
    ap.add_argument("--min-exposure-sec", type=int, default=1, help="STATE tokenization: minimum run exposure in seconds to emit STATE tokens (default 1; use 5+ to suppress micro states)")
    ap.add_argument("--state-stable-sec", type=int, default=1, help="STATE tokenization: optional dwell (seconds) required before accepting a new state (default 1; >1 debounces flicker)")
    ap.add_argument("--state-chunk-by-h", type=int, default=0, help="If 1, split long STATE runs into multiple STATE rows every H seconds. Default 0 (one STATE per real lineup regime).")
    ap.add_argument("--schema", choices=["slim", "full"], default="slim", help="Output schema for player_tokens_pre: slim keeps only required columns; full includes all diagnostic sim_reason_* fields and any extra columns (default slim).")
    ap.add_argument("--min-swap", type=int, default=2, help="ROSTER_CHANGE threshold (default 2)")
    ap.add_argument("--min-gap-sec", type=int, default=8, help="Minimum seconds between kept ROSTER_CHANGE tokens (default 8)")
    ap.add_argument("--stable-sec", type=int, default=2, help="Debounce ROSTER_CHANGE: require teammate set stable for N seconds before emitting (<=1 disables; default 2)")
    ap.add_argument(
        "--matchup-stable-sec",
        type=int,
        default=6,
        help=(
            "When --opp-change-tokens=1, require BOTH teammate+opponent sets to remain unchanged for N seconds "
            "before emitting a ROSTER_CHANGE (suppresses short-lived opponent response flicker). Default 6."
        ),
    )
    ap.add_argument("--rc-require-stable", type=int, default=1, help="If 1, drop ROSTER_CHANGE candidates that never become stable before segment end (prevents 1-second flicker tokens). Default 1.")
    ap.add_argument("--post-dwell-sec", type=int, default=2, help="Require player remains on-ice for N seconds after emitted ROSTER_CHANGE time (default 2).")
    ap.add_argument("--mass-swap-suppress-threshold", type=int, default=3, help="If >=3 skaters swap on a side at a second, suppress ROSTER_CHANGE emission at that second (anchors cover it). Default 3.")
    ap.add_argument("--opp-change-tokens", type=int, default=0, help="If 1, also emit opponent-driven context updates as ROSTER_CHANGE tokens tagged rc_side=opp (same debounce/min-gap rules). Default 0.")
    ap.add_argument("--opp-cover-gap-sec", type=int, default=4, help="Coverage suppression for opponent-driven updates: skip rc_side=opp ROSTER_CHANGE if any token was emitted within the last N seconds. Default 4.")
    ap.add_argument("--max-tokens", type=int, default=10, help="Max tokens per player per window (default 10)")
    ap.add_argument("--mode", choices=["train", "sim", "both"], default="both", help="Which feature namespaces to output")
    ap.add_argument("--season", default="", help="Optional season label to stamp into CSVs (e.g., 20252026)")
    ap.add_argument("--date", default="", help="Optional game date to stamp into CSVs (YYYY-MM-DD)")
    ap.add_argument("--rinkid", default="", help="Optional rink id/name to stamp into CSVs")
    ap.add_argument("--player-meta-csv", default="artifacts/player_career_years_2017_2025.csv", help="Fallback player metadata CSV to fill playerName/positionCode if raw boxscore/shiftcharts are unavailable. Use empty string to disable.")
    ap.add_argument("--player-landing-cache-dir", default="artifacts/cache/player_landing", help="Cache directory for api-web player landing JSON (used for handedness/age enrichment).")
    ap.add_argument("--fetch-player-landing", type=int, default=1, help="If 1, fetch missing player landing JSON from api-web (cached). If 0, never fetch and leave age blank if unknown. Default 1.")
    ap.add_argument("--write-parquet", type=int, default=1, help="If 1, also write player_tokens_pre_{gamePk}.parquet (default 1).")
    ap.add_argument("--write-windows", action="store_true", help="Also write windows_{gamePk}.csv (containers)")
    ap.add_argument("--write-post", action="store_true", help="Also write player_tokens_post_{gamePk}.csv (post-horizon evidence; do NOT use as PRE inputs)")
    ap.add_argument("--entry-negatives", type=int, default=0, help="If >0, also write entry_selection_{gamePk}.csv with N negative candidates per ENTRY token")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    pbp_onice = load_json(args.in_path)
    game = parse_game(pbp_onice)

    # Prefer raw boxscore/shiftcharts metadata for rink + player meta (truly from raws).
    raw_dir = _infer_raw_dir_from_pbpice_path(str(args.in_path))
    raw_rinkid, raw_season, raw_date, raw_name_map, raw_pos_map = load_raw_boxscore_and_shiftcharts_meta(raw_dir, int(game.gamePk))
    team_meta = load_raw_boxscore_team_meta(raw_dir, int(game.gamePk))

    # season/date stamps: prefer CLI; else raw boxscore/pbp (if available); else keep blank.
    season_stamp = str(args.season or "").strip() or str(raw_season or "").strip()
    date_stamp = str(args.date or "").strip() or str(raw_date or "").strip()
    game_date_obj = _parse_yyyy_mm_dd(date_stamp)

    # rinkid: prefer CLI; else raw arena name; else home team abbrev as deterministic stamp.
    rinkid = str(args.rinkid or "").strip() or str(raw_rinkid or "").strip()
    if not rinkid:
        try:
            rinkid = str((pbp_onice.get("home", {}) or {}).get("abbrev") or game.home_team_id)
                        except Exception:
            rinkid = str(game.home_team_id)

    # Player meta maps: prefer raw; fall back to artifact CSV if provided.
    player_name_map = dict(raw_name_map or {})
    player_pos_map = dict(raw_pos_map or {})
    meta_path = str(args.player_meta_csv or "").strip()
    csv_hand_map: Dict[int, str] = {}
    if meta_path:
        csv_name_map, csv_pos_map = load_player_meta_csv(meta_path)
        csv_hand_map = load_player_handedness_csv(meta_path)
        for k, v in (csv_name_map or {}).items():
            player_name_map.setdefault(int(k), v)
        for k, v in (csv_pos_map or {}).items():
            player_pos_map.setdefault(int(k), v)

    (
        team_onice_by_sec,
        goalie_ids_by_sec,
        events_by_sec,
        orders_by_sec,
        sso_by_sec,
        shift_changes_by_sec,
        end_event_at,
        fo_meta,
        tv_timeout_secs,
        horizon,
    ) = build_second_index(game)

    credit_by_sec, cum_home_goals, cum_away_goals = build_credit_by_sec(
        game, team_onice_by_sec, goalie_ids_by_sec, events_by_sec, orders_by_sec, shift_changes_by_sec
    )

    windows = build_windows(
        game=game,
        team_onice_by_sec=team_onice_by_sec,
        goalie_ids_by_sec=goalie_ids_by_sec,
        events_by_sec=events_by_sec,
        orders_by_sec=orders_by_sec,
        sso_by_sec=sso_by_sec,
        shift_changes_by_sec=shift_changes_by_sec,
        end_event_at=end_event_at,
        fo_meta=fo_meta,
        tv_timeout_secs=tv_timeout_secs,
        horizon=horizon,
        hard_cap_sec=int(args.hard_cap_sec),
    )

    out_windows_csv = ""
    if args.write_windows:
        # windows CSV (containers; optional)
        windows_rows = []
        for w in windows:
            windows_rows.append(
                {
                    "season": args.season,
                    "date": args.date,
                    "gamePk": int(game.gamePk),
                    "rinkid": args.rinkid,
                    **w,
                }
            )
        windows_cols = [
            "season","date","gamePk","rinkid",
            "window_id","period","start_sec","end_sec","duration","clock_start",
            "strength_global","home_goalie_present","away_goalie_present","home_goalie_pulled","away_goalie_pulled",
            "fo_zone","fo_won_team_id","fo_won_player_id","fo_lost_player_id",
            "start_prev_break_type","start_prev_break_team_id","start_prev_break_subtype","media_timeout_start",
            "end_event_type","home_ids_start","away_ids_start","home_ids_end","away_ids_end",
        ]
        extra_w = sorted({k for r in windows_rows for k in r.keys() if k not in windows_cols})
        out_windows_csv = os.path.join(args.out_dir, f"windows_{game.gamePk}.csv")
        write_csv(out_windows_csv, windows_rows, windows_cols + extra_w)

    # tokens CSV
    token_rows = generate_tokens_and_rows(
        game=game,
        windows=windows,
        team_onice_by_sec=team_onice_by_sec,
        goalie_ids_by_sec=goalie_ids_by_sec,
        events_by_sec=events_by_sec,
        orders_by_sec=orders_by_sec,
        sso_by_sec=sso_by_sec,
        shift_changes_by_sec=shift_changes_by_sec,
        fo_meta=fo_meta,
        end_event_at=end_event_at,
        credit_by_sec=credit_by_sec,
        cum_home_goals=cum_home_goals,
        cum_away_goals=cum_away_goals,
        horizon_sec=int(args.horizon_sec),
        min_swap=int(args.min_swap),
        min_gap_sec=int(args.min_gap_sec),
        stable_sec=int(args.stable_sec),
        matchup_stable_sec=int(args.matchup_stable_sec),
        rc_require_stable=int(args.rc_require_stable),
        post_dwell_sec=int(args.post_dwell_sec),
        mass_swap_suppress_threshold=int(args.mass_swap_suppress_threshold),
        max_tokens=int(args.max_tokens),
        opp_change_tokens=int(args.opp_change_tokens),
        opp_cover_gap_sec=int(args.opp_cover_gap_sec),
        mode=str(args.mode),
        player_name_map=player_name_map,
        player_pos_map=player_pos_map,
        season=season_stamp,
        date=date_stamp,
        rinkid=rinkid,
        min_exposure_sec=int(args.min_exposure_sec),
        state_stable_sec=int(args.state_stable_sec),
        state_chunk_by_h=int(args.state_chunk_by_h),
    )

    # --- Enrichment: handedness, age, and team names (for Parquet + easier modeling) ---
    cache_dir = str(args.player_landing_cache_dir or "").strip() or "artifacts/cache/player_landing"
    fetch_landing = int(args.fetch_player_landing) != 0

    # Team identity (from raw boxscore; best-effort)
    home_abbrev = str(team_meta.get("homeAbbrev") or (pbp_onice.get("home", {}) or {}).get("abbrev") or "").strip()
    away_abbrev = str(team_meta.get("awayAbbrev") or (pbp_onice.get("away", {}) or {}).get("abbrev") or "").strip()
    home_name = str(team_meta.get("homeName") or "").strip()
    away_name = str(team_meta.get("awayName") or "").strip()
    home_tid = _safe_int(team_meta.get("homeTeamId"), int(game.home_team_id))
    away_tid = _safe_int(team_meta.get("awayTeamId"), int(game.away_team_id))

    landing_cache: Dict[int, Dict[str, Any]] = {}
    for r in token_rows:
        pid = int(r.get("playerId", 0) or 0)
        side = str(r.get("team_side") or "")

        # team names next to player fields
        if side == "home":
            r["teamAbbrev"] = home_abbrev
            r["teamName"] = home_name
            r["oppTeamId"] = int(away_tid)
            r["oppTeamAbbrev"] = away_abbrev
            r["oppTeamName"] = away_name
        else:
            r["teamAbbrev"] = away_abbrev
            r["teamName"] = away_name
            r["oppTeamId"] = int(home_tid)
            r["oppTeamAbbrev"] = home_abbrev
            r["oppTeamName"] = home_name

        # handedness + birthDate from meta csv first; else api-web player landing (cached on disk).
        hand = str((csv_hand_map.get(pid) or "")).strip()
        birth_date_obj: Optional[datetime.date] = None
        if (not hand or game_date_obj is not None) and pid > 0:
            if pid not in landing_cache:
                landing_cache[pid] = get_player_landing_cached(pid, cache_dir, allow_fetch=fetch_landing)
            land = landing_cache.get(pid) or {}
            if not hand:
                hand = str(land.get("shootsCatches") or land.get("shoots_catches") or land.get("handedness") or "").strip()
            bd = str(land.get("birthDate") or land.get("birthdate") or land.get("birth_date") or "").strip()
            birth_date_obj = _parse_yyyy_mm_dd(bd)
        r["handedness"] = hand

        age = compute_age_years(birth_date_obj, game_date_obj) if game_date_obj is not None else None
        r["age"] = int(age) if age is not None else ""

    def _dedupe_train_sim_identical_columns(rows: List[Dict[str, Any]]) -> List[str]:
        """
        If mode=both and train_* columns are identical to sim_* columns for all rows,
        drop the redundant train_* columns.
        Returns the list of dropped column names.
        """
        if not rows:
            return []
        keys: set = set()
        for r in rows:
            keys.update(r.keys())
        sim_bases = {k[4:] for k in keys if k.startswith("sim_")}
        train_bases = {k[6:] for k in keys if k.startswith("train_")}
        common = sorted(sim_bases & train_bases)
        dropped: List[str] = []
        for base in common:
            sk = f"sim_{base}"
            tk = f"train_{base}"
            same = True
            for r in rows:
                if r.get(sk) != r.get(tk):
                    same = False
                    break
            if same:
                dropped.append(tk)
        if dropped:
            for r in rows:
                for k in dropped:
                    if k in r:
                        del r[k]
        return dropped

    # Canonical, auditable column ordering:
    # Who/Where → Window context → Token semantics → Token timing/state → Segment debug → PRE roster/entities
    # → instruments/constraints → exposure/offsets → outcomes → exit/replacement supervision.
    tok_cols_core = [
        # 1) Identity / grouping (never model inputs)
        "season","date","rinkid","gamePk",
        "teamId","team_side","teamAbbrev","teamName","oppTeamId","oppTeamAbbrev","oppTeamName",
        "playerId","playerName","positionCode","handedness","age",
        "window_id","strength_global","token_idx","t_token",

        # 2) Token semantics + timing
        "token_type","seg_idx",
        "is_outcome_token","is_selection_token","is_hazard_token",
        "run_id","run_start","run_end","run_duration","state_changed_reason","state_persist_sec",
        "t_context","t_end","seconds_token","seconds_onice_in_h",

        # 3) Container/window context (stable within window; decision-time instruments)
        "win_start_sec",
        "win_fo_zone","win_fo_won_team_id",
        "win_start_prev_break_type","win_start_prev_break_team_id","win_start_prev_break_subtype",
        "win_home_last_change_opportunity","win_media_timeout_start",
        "win_home_goalie_pulled","win_away_goalie_pulled",
        "tok_last_fo_zone","is_boundary_second",

        # 4) PRE roster & entities (CIN core; fixed slots only)
        "num_team_skaters","num_opp_skaters",
        "team_slot1","team_slot2","team_slot3","team_slot4","team_slot5","team_slot6",
        "opp_slot1","opp_slot2","opp_slot3","opp_slot4","opp_slot5","opp_slot6",
        "own_goalie_id","opp_goalie_id",

        # 5) Decision-time constraints
        "bench_rest_bucket","bench_size_at_t","entry_after_faceoff_flag",

        # 6) Exposure & offsets
        "offset_log_toi",

        # 7) Targets (what happened next)
            "xGF","xGA","GF","GA","SF","SA","AF","AA","BF","BA",
            "giveaways_committed","takeaways_forced","hits_personal","blocks_personal","shots_blocked_personal",
            "giveaways_committed_oz","giveaways_committed_nz","giveaways_committed_dz",
            "takeaways_forced_oz","takeaways_forced_nz","takeaways_forced_dz",
            "hits_personal_oz","hits_personal_nz","hits_personal_dz",
            "blocks_personal_oz","blocks_personal_nz","blocks_personal_dz",
            "shots_blocked_personal_oz","shots_blocked_personal_nz","shots_blocked_personal_dz",

        # 8) Exit / replacement supervision
        "exit_event_within_h","exit_time_to_exit_s","exit_reason_proxy",
        "replacement_count","replacement1_id","replacement2_id",
    ]

    schema = str(args.schema or "slim").lower().strip()
    sim_cols = [
        # Slim, sim-safe context
        "sim_period","sim_clock_s",
        "sim_strength_team","sim_score_diff","sim_long_change","sim_home_away",
        "sim_last_event_type","sim_time_since_last_event_s","sim_last_event_zone","sim_last_event_owner_team_id",
        "sim_shift_elapsed_s",
        "sim_exit_reason_proxy_pre","sim_exit_reason_conf_pre",
        "sim_change_reason_proxy_pre","sim_change_reason_conf_pre",
    ]
    if schema == "full":
        sim_cols.extend(
            [
                # more detailed diagnostics
                "sim_last_event_sec","sim_last_event_lookback_s",
                "sim_shift_prev_n","sim_shift_prev_mean_len_s","sim_shift_prev_sd_len_s",
                "sim_reason_is_special_teams","sim_reason_is_after_faceoff","sim_reason_is_after_stoppage","sim_reason_is_after_icing","sim_reason_is_after_goal","sim_reason_is_after_penalty",
                "sim_reason_is_period_boundary","sim_reason_own_goalie_pulled","sim_reason_opp_goalie_pulled","sim_reason_own_goalie_pull_transition","sim_reason_opp_goalie_pull_transition","sim_reason_boundary_prev_break_type",
                "sim_reason_zone_O","sim_reason_zone_D","sim_reason_zone_N","sim_reason_score_big","sim_reason_fatigue_z","sim_reason_shift_elapsed_s","sim_reason_shift_mean_s",
            ]
        )
    train_cols = [
        "train_last_shift_len_s","train_time_since_last_shift_s","train_entry_offset_s",
    ]

    # Insert sim/train blocks right after exposure fields (keep timing/exposure together).
    insert_at = tok_cols_core.index("seconds_onice_in_h") + 1
    mode = str(args.mode).lower()
    cols_mode_block: List[str] = []
    if mode in ("sim", "both"):
        cols_mode_block.extend(sim_cols)
    if mode in ("train", "both"):
        cols_mode_block.extend(train_cols)
    tok_cols_core = tok_cols_core[:insert_at] + cols_mode_block + tok_cols_core[insert_at:]

    dropped_train: List[str] = []
    if mode == "both":
        dropped_train = _dedupe_train_sim_identical_columns(token_rows)
        if dropped_train:
            tok_cols_core = [c for c in tok_cols_core if c not in set(dropped_train)]
    post_cols = [
        "season","date","gamePk","teamId","team_side","window_id","strength_global","playerId","token_idx","token_type","t_token","t_end","seconds_token",
        "post_teammates_onice_ids_w","post_teammates_onice_w","post_teammates_onice_sec_w","post_teammates_onice_share_raw",
        "post_opponents_onice_ids_w","post_opponents_onice_w","post_opponents_onice_sec_w","post_opponents_onice_share_raw",
        "post_with_event_GF","post_with_event_GA","post_with_event_SF","post_with_event_SA",
    ]

    # --- Row ordering (for human review) ---
    # Within each window, list players in the order they first appear in that window (carried-at-start first),
    # and then list that player's tokens in time order. This is only an output sort; it does not change token_idx.
    def _window_num(wid: Any) -> int:
        s = str(wid or "")
        digs = "".join(ch for ch in s if ch.isdigit())
        return int(digs) if digs else 0

    token_type_priority = {
        "ENTRY": 0,
        "WINDOW_START": 1,
        "STATE": 2,
        "EXIT": 3,
    }

    first_t: Dict[Tuple[str, str, int], int] = {}
    for r in token_rows:
        k = (str(r.get("window_id") or ""), str(r.get("team_side") or ""), int(r.get("playerId") or 0))
        t = int(r.get("t_token") or 0)
        if k not in first_t or t < first_t[k]:
            first_t[k] = t

    token_rows = sorted(
        token_rows,
        key=lambda r: (
            str(r.get("season") or ""),
            str(r.get("date") or ""),
            int(r.get("gamePk") or 0),
            _window_num(r.get("window_id")),
            # player ordering within window
            first_t.get(
                (str(r.get("window_id") or ""), str(r.get("team_side") or ""), int(r.get("playerId") or 0)),
                int(r.get("t_token") or 0),
            ),
            int(r.get("playerId") or 0),
            # token ordering within player
            int(r.get("t_token") or 0),
            token_type_priority.get(str(r.get("token_type") or "").upper(), 99),
            int(r.get("token_idx") or 0),
        ),
    )

    # Write PRE (CIN-safe) token table (default + recommended)
    extra_pre: List[str] = []
    if schema == "full":
        extra_pre = sorted({k for r in token_rows for k in r.keys() if k not in tok_cols_core and not str(k).startswith("post_")})
    out_pre_csv = os.path.join(args.out_dir, f"player_tokens_pre_{game.gamePk}.csv")
    write_csv(out_pre_csv, token_rows, tok_cols_core + extra_pre)

    # Also write Parquet (requested for modeling; preserves ordering and types better than CSV).
    if int(args.write_parquet) != 0:
        try:
            import pandas as pd
                except Exception:
            pd = None
        if pd is not None:
            out_pre_parquet = os.path.join(args.out_dir, f"player_tokens_pre_{game.gamePk}.parquet")
            cols_all = tok_cols_core + extra_pre
            df = pd.DataFrame(token_rows)
            # ensure deterministic column order (include missing columns as NA)
            for c in cols_all:
                if c not in df.columns:
                    df[c] = None
            # Keep PRE columns only (avoid writing post_* list columns that don't Parquet-convert cleanly).
            df = df[cols_all]
            df.to_parquet(out_pre_parquet, index=False)

    out_post_csv = ""
    if args.write_post:
        # Write POST evidence table (overlap/chemistry over horizon) separately to prevent leakage
        extra_post = sorted({k for r in token_rows for k in r.keys() if k not in post_cols and str(k).startswith("post_")})
        out_post_csv = os.path.join(args.out_dir, f"player_tokens_post_{game.gamePk}.csv")
        write_csv(out_post_csv, token_rows, post_cols + extra_post)

    # ENTRY selection supervision (optional)
    if int(args.entry_negatives) > 0:
        # Build "seen roster" by team side across the game (skaters only)
        home_seen: set = set()
        away_seen: set = set()
        for hh, aa in team_onice_by_sec:
            home_seen.update(int(x) for x in hh)
            away_seen.update(int(x) for x in aa)

        entry_rows: List[Dict[str, Any]] = []
        group_id = 0
        for r in token_rows:
            if str(r.get("token_type", "")).upper() != "ENTRY":
                continue
            group_id += 1
            side = str(r.get("team_side"))
            t_tok = int(r.get("t_token", 0) or 0)
            chosen_pid = int(r.get("playerId", 0) or 0)
            if chosen_pid <= 0:
                continue
            our_set, _ = (
                (set(team_onice_by_sec[t_tok][0]), set(team_onice_by_sec[t_tok][1]))
                if side == "home"
                else (set(team_onice_by_sec[t_tok][1]), set(team_onice_by_sec[t_tok][0]))
            )
            seen_roster = home_seen if side == "home" else away_seen
            bench_candidates = sorted(int(x) for x in (seen_roster - our_set) if int(x) != chosen_pid)
            # Deterministic negative sample
            seed = int(game.gamePk) * 1000003 + int(chosen_pid) * 97 + int(t_tok)
            negs = stable_sample_ints(bench_candidates, int(args.entry_negatives), seed)

            # Positive row
            entry_rows.append(
                {
                    "group_id": int(group_id),
                    "season": r.get("season", ""),
                    "date": r.get("date", ""),
                    "gamePk": int(game.gamePk),
                    "teamId": int(r.get("teamId", 0) or 0),
                    "team_side": side,
                    "window_id": r.get("window_id", ""),
                    "t_token": int(t_tok),
                    "chosen_playerId": int(chosen_pid),
                    "candidate_playerId": int(chosen_pid),
                    "entered": 1,
                    # a few key instruments/context for selection head
                    "strength_global": r.get("strength_global", ""),
                    "strength_team": r.get("sim_strength_team", ""),
                    "score_diff": r.get("sim_score_diff", ""),
                    "tok_last_fo_zone": r.get("tok_last_fo_zone", ""),
                    "win_home_last_change_opportunity": r.get("win_home_last_change_opportunity", 0),
                    "bench_size_at_t": int(len(bench_candidates)),
                }
            )
            # Negatives
            for cand in negs:
                entry_rows.append(
                    {
                        "group_id": int(group_id),
                        "season": r.get("season", ""),
                        "date": r.get("date", ""),
                        "gamePk": int(game.gamePk),
                        "teamId": int(r.get("teamId", 0) or 0),
                        "team_side": side,
                        "window_id": r.get("window_id", ""),
                        "t_token": int(t_tok),
                        "chosen_playerId": int(chosen_pid),
                        "candidate_playerId": int(cand),
                        "entered": 0,
                        "strength_global": r.get("strength_global", ""),
                        "strength_team": r.get("sim_strength_team", ""),
                        "score_diff": r.get("sim_score_diff", ""),
                        "tok_last_fo_zone": r.get("tok_last_fo_zone", ""),
                        "win_home_last_change_opportunity": r.get("win_home_last_change_opportunity", 0),
                        "bench_size_at_t": int(len(bench_candidates)),
                    }
                )

        entry_cols = [
            "group_id","season","date","gamePk","teamId","team_side","window_id","t_token",
            "chosen_playerId","candidate_playerId","entered",
            "strength_global","strength_team","score_diff","tok_last_fo_zone",
            "win_home_last_change_opportunity","bench_size_at_t",
        ]
        out_entry_csv = os.path.join(args.out_dir, f"entry_selection_{game.gamePk}.csv")
        write_csv(out_entry_csv, entry_rows, entry_cols)
        print(f"Wrote {out_entry_csv} ({len(entry_rows)} rows; {group_id} ENTRY groups)")

    if out_windows_csv:
        print(f"Wrote {out_windows_csv}")
    print(f"Wrote {out_pre_csv} ({len(token_rows)} token-rows)")
    if out_post_csv:
        print(f"Wrote {out_post_csv}")


if __name__ == "__main__":
    main()

