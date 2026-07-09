# build_pbp_with_onice.py
# Creates a per-game JSON that aligns every play-by-play event AND any pure shift-change
# moment with the exact on-ice skater sets and per-player time-on-ice at that instant.
# Usage:
#   python build_pbp_with_onice.py --game 2024020849 --raw /path/to/raw --out /path/to/out
#
# Expected raw layout (mirrors your current scripts):
#   <raw>/pbp/<gamePk>.json         # NHL api-web play-by-play JSON
#   <raw>/boxscore/<gamePk>.json    # box score JSON (for team ids/abbrevs, positions)
#   <raw>/shiftcharts/<gamePk>.json # shift charts JSON (or {"data":[...]} )
#
# Output:
#   <out>/pbp_onice_<gamePk>.json
#
# Notes / invariants:
# - On-ice snapshot at second s is the state **after** all changes taking effect at s (post-change).
# - If there are PBP plays at s, we DO NOT emit a synthetic "shift_change". We attach the delta to
#   the FIRST PBP play at s; subsequent plays at s carry empty shift_change lists.
# - Plays within the same second are sorted by hockey priority:
#     GOAL (1) > PENALTY (2) > STOPPAGE/PUCK-OUT/GOALIE-STOPPED/ICING/OFFSIDE/TIMEOUT/CHALLENGE (3) > FACEOFF (4) > other (5)
# - For determinism, each play gets `same_sec_order` (0,1,2,...).
# - NEW: For PBP plays at a second:
#     * The FIRST play uses the PRE-change on-ice snapshot and `sec_phase="pre-change"`.
#     * Subsequent plays at the same second use POST-change on-ice and `sec_phase="post-change"`.
#   Faceoffs still get `faceoff_anchor=True`.
#
import os, json, argparse, bisect
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

SECONDS_PER_PERIOD = 20 * 60
MAX_SECONDS = SECONDS_PER_PERIOD * 6  # generous game horizon

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def resolve_path(root: str, cands: List[str]) -> str:
    for rel in cands:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return os.path.join(root, cands[0])

def sec_from_period_clock(period: int, mmss: str) -> int:
    try:
        m, s = str(mmss).split(":")
        return (int(period) - 1) * SECONDS_PER_PERIOD + int(m) * 60 + int(s)
    except Exception:
        return 0

def iter_apiweb_plays(gc: Dict[str, Any]):
    for p in (gc or {}).get("plays", []) or []:
        pd = p.get("periodDescriptor") or {}
        sec = sec_from_period_clock(int(pd.get("number", 1)), p.get("timeInPeriod", "00:00"))
        so = int(p.get("sortOrder", 0))
        yield sec, so, p

def get_home_away_ids(box: Dict[str, Any]) -> Tuple[int, int, str, str]:
    h = (box.get("homeTeam") or {})
    a = (box.get("awayTeam") or {})
    return int(h.get("id")), int(a.get("id")), str(h.get("abbrev")), str(a.get("abbrev"))

def build_player_maps_from_box(box: Dict[str, Any]):
    pid_to_tid: Dict[int, int] = {}
    pid_to_pos: Dict[int, str] = {}
    pgs = (box.get("playerByGameStats") or {})
    for side in ("homeTeam", "awayTeam"):
        team = (box.get(side) or {})
        tid = team.get("id")
        stats = (pgs.get(side) or {})
        for group, code in (("forwards", "F"), ("defense", "D"), ("goalies", "G")):
            for p in (stats.get(group) or []):
                pid = p.get("playerId")
                if isinstance(pid, int):
                    pid_to_tid[pid] = int(tid)
                    pid_to_pos[pid] = code
    return pid_to_tid, pid_to_pos

def sec_from_period_clock_generic(period: int, mmss: str) -> int:
    try:
        m, s = str(mmss).split(":")
        return (int(period) - 1) * SECONDS_PER_PERIOD + int(m) * 60 + int(s)
    except Exception:
        return 0

def build_onice_index(shift_rows: List[Dict[str, Any]]):
    """Returns per-second sorted list of playerIds on ice (all players, including goalies)."""
    horizon = MAX_SECONDS
    onice_sets: List[set] = [set() for _ in range(horizon + 1)]
    for r in shift_rows:
        pid = r.get("playerId") or r.get("player_id")
        period = r.get("period") or r.get("periodNumber")
        start = r.get("startTime") or r.get("start_time")
        end = r.get("endTime") or r.get("end_time")
        if not (isinstance(pid, int) and period and start and end):
            continue
        s = sec_from_period_clock_generic(int(period), str(start))
        e = sec_from_period_clock_generic(int(period), str(end))
        if e <= s:
            continue
        s = max(0, min(s, horizon))
        e = max(0, min(e, horizon))
        for t in range(s, e):
            onice_sets[t].add(int(pid))
    return [sorted(list(s)) for s in onice_sets]

def build_player_shift_intervals(shift_rows: List[Dict[str, Any]]) -> Dict[int, List[Tuple[int, int]]]:
    """Per-player intervals (start_sec, end_sec)."""
    intervals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for r in shift_rows:
        pid = r.get("playerId") or r.get("player_id")
        period = r.get("period") or r.get("periodNumber")
        start = r.get("startTime") or r.get("start_time")
        end = r.get("endTime") or r.get("end_time")
        if not (isinstance(pid, int) and period and start and end):
            continue
        s = sec_from_period_clock_generic(int(period), str(start))
        e = sec_from_period_clock_generic(int(period), str(end))
        if e > s:
            intervals[int(pid)].append((s, e))
    for pid, lst in intervals.items():
        lst.sort()
    return intervals

def toi_for_player_at(pid: int, sec: int, intervals: List[Tuple[int, int]]) -> Optional[int]:
    """Returns seconds-on-ice for pid at 'sec' if on-ice, else None. Uses binary search over intervals."""
    if not intervals:
        return None
    starts = [a for a, _ in intervals]
    i = bisect.bisect_right(starts, sec) - 1
    if i >= 0:
        s, e = intervals[i]
        if s <= sec < e:
            return sec - s
    return None

def build_goalies_by_team(onice: List[List[int]], pid_to_tid: Dict[int,int], pid_to_pos: Dict[int,str], horizon: int):
    goalies_by_team: List[Dict[int, int]] = [dict() for _ in range(horizon + 1)]
    for s in range(horizon + 1):
        counts = defaultdict(int)
        for pid in onice[s]:
            if (pid_to_pos.get(pid) or "").upper() == "G":
                tid = pid_to_tid.get(pid)
                if isinstance(tid, int):
                    counts[tid] += 1
        goalies_by_team[s] = dict(counts)
    return goalies_by_team

# ---------- New: priority for same-second ordering ----------
def play_priority(typ: str) -> int:
    t = (typ or "").lower()
    if t == "goal": return 1
    if t == "penalty": return 2
    if t in {"stoppage","puck-out-of-play","goalie-stopped","icing","offside","timeout","challenge"}:
        return 3
    if t == "faceoff": return 4
    return 5

def build(game_pk: int, raw_dir: str, out_dir: str) -> str:
    # Resolve paths
    pbp_path   = resolve_path(raw_dir, [os.path.join("pbp", f"{game_pk}.json"), os.path.join("playbyplay", f"{game_pk}.json"), f"pbp_{game_pk}.json"])
    box_path   = resolve_path(raw_dir, [os.path.join("boxscore", f"{game_pk}.json"), os.path.join("box", f"{game_pk}.json"), f"box_{game_pk}.json"])
    shifts_path= resolve_path(raw_dir, [os.path.join("shiftcharts", f"{game_pk}.json"), os.path.join("shifts", f"{game_pk}.json"), f"shifts_{game_pk}.json"])

    pbp = load_json(pbp_path)
    box = load_json(box_path)
    shifts = load_json(shifts_path)
    shift_rows = (shifts.get("data") if isinstance(shifts, dict) and "data" in shifts else shifts) or []

    home_id, away_id, home_abbr, away_abbr = get_home_away_ids(box)
    pid_to_tid, pid_to_pos = build_player_maps_from_box(box)

    onice = build_onice_index(shift_rows)
    horizon = min(MAX_SECONDS, len(onice) - 1)
    _ = build_goalies_by_team(onice, pid_to_tid, pid_to_pos, horizon)  # kept for parity; not used directly here
    player_intervals = build_player_shift_intervals(shift_rows)

    # Pre-index plays by second with deterministic priority
    plays_by_sec: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for sec, so, p in iter_apiweb_plays(pbp):
        plays_by_sec[sec].append({"sortOrder": so, "raw": p})

    for sec in plays_by_sec:
        # Strict global ordering by native sortOrder only (no type prioritization)
        plays_by_sec[sec].sort(key=lambda r: int(r["sortOrder"]))
        for i, r in enumerate(plays_by_sec[sec]):
            r["same_sec_order"] = i

    # Helper: skater sets excluding goalies, per side (POST-CHANGE snapshot at sec)
    def skaters_at(sec: int) -> Tuple[List[int], List[int]]:
        home, away = [], []
        for pid in onice[min(max(sec,0), horizon)]:
            if (pid_to_pos.get(pid) or "").upper() == "G":
                continue
            tid = pid_to_tid.get(pid)
            if tid == home_id:
                home.append(pid)
            elif tid == away_id:
                away.append(pid)
        return sorted(home), sorted(away)

    # Helper: goalies at sec (one per team if present)
    def goalies_at(sec: int) -> Dict[str, Optional[int]]:
        s = min(max(sec,0), horizon)
        ghome = None
        gaway = None
        for pid in onice[s]:
            if (pid_to_pos.get(pid) or "").upper() == "G":
                tid = pid_to_tid.get(pid)
                if tid == home_id:
                    ghome = pid
                elif tid == away_id:
                    gaway = pid
        return {"home": ghome, "away": gaway}

    # Build event stream over 0..horizon
    events_out: List[Dict[str, Any]] = []
    prev_home, prev_away = skaters_at(0)

    last_pb_so: float = -1.0
    for sec in range(0, horizon + 1):
        cur_home, cur_away = skaters_at(sec)
        # Detect shift delta vs previous second
        home_out = [p for p in prev_home if p not in cur_home]
        home_in  = [p for p in cur_home if p not in prev_home]
        away_out = [p for p in prev_away if p not in cur_away]
        away_in  = [p for p in cur_away if p not in prev_away]
        any_change = bool(home_in or home_out or away_in or away_out)

        # If there is NO PBP at this second but lines changed, emit a pure shift_change
        if any_change and sec not in plays_by_sec:
            toi_map = {}
            for pid in cur_home + cur_away:  # TOI for players ON after the change
                toi = toi_for_player_at(pid, sec, player_intervals.get(pid, []))
                if toi is not None:
                    toi_map[str(pid)] = int(toi)
            events_out.append({
                "type": "shift_change",
                "period": (sec // SECONDS_PER_PERIOD) + 1,
                "timeInPeriod": f"{(sec % SECONDS_PER_PERIOD)//60:02d}:{(sec % SECONDS_PER_PERIOD)%60:02d}",
                "sec_game": sec,
                # Assign a synthetic sortOrder just after the last seen PBP play so global order stays monotonic
                "sortOrder": float(last_pb_so + 0.0001),
                "same_sec_order": 0,
                "details": {},
                "onice": {"home": cur_home, "away": cur_away, "goalies": goalies_at(sec)},
                "shift_change": {"home_in": home_in, "home_out": home_out, "away_in": away_in, "away_out": away_out},
                "toi_by_player": toi_map,
                "sec_phase": "post-change",
                "faceoff_anchor": False
            })

        # Emit PBP plays (if any)
        first_play = True
        for rec in plays_by_sec.get(sec, []):
            p = rec["raw"]
            pd = p.get("periodDescriptor") or {}
            typ = (p.get("typeDescKey") or "").lower()

            if first_play:
                # FIRST play at this second: use PRE-change snapshot (correct credit for goals/shots)
                onice_payload = {"home": prev_home, "away": prev_away, "goalies": goalies_at(sec)}
                toi_map = {}
                for pid in prev_home + prev_away:
                    toi = toi_for_player_at(pid, sec, player_intervals.get(pid, []))
                    if toi is not None:
                        toi_map[str(pid)] = int(toi)
                sc_payload = {
                    "home_in": home_in, "home_out": home_out,
                    "away_in": away_in, "away_out": away_out,
                } if any_change else {"home_in": [], "home_out": [], "away_in": [], "away_out": []}
                sec_phase = "pre-change"
            else:
                # Subsequent plays at same second: POST-change snapshot
                onice_payload = {"home": cur_home, "away": cur_away, "goalies": goalies_at(sec)}
                toi_map = {}
                for pid in cur_home + cur_away:
                    toi = toi_for_player_at(pid, sec, player_intervals.get(pid, []))
                    if toi is not None:
                        toi_map[str(pid)] = int(toi)
                sc_payload = {"home_in": [], "home_out": [], "away_in": [], "away_out": []}
                sec_phase = "post-change"

            events_out.append({
                "type": typ,
                "period": int(pd.get("number", 1)),
                "timeInPeriod": p.get("timeInPeriod"),
                "sec_game": sec,
                "sortOrder": float(p.get("sortOrder", 0)),
                "same_sec_order": int(rec.get("same_sec_order", 0)),
                "details": p.get("details") or {},
                "onice": onice_payload,
                "shift_change": sc_payload,
                "toi_by_player": toi_map,
                "sec_phase": sec_phase,
                "faceoff_anchor": (typ == "faceoff"),
            })
            # Track last native sortOrder for positioning synthetic shift_change rows
            try:
                last_pb_so = max(last_pb_so, float(p.get("sortOrder", 0)))
            except Exception:
                pass
            first_play = False

        prev_home, prev_away = cur_home, cur_away

    # Final: strictly sort the full event stream by sortOrder, then sec_game
    events_out.sort(key=lambda e: (float(e.get("sortOrder", 0)), int(e.get("sec_game", 0)), int(e.get("same_sec_order", 0))))

    # Compose final doc
    out_doc = {
        "gamePk": game_pk,
        "home": {"teamId": home_id, "abbrev": home_abbr},
        "away": {"teamId": away_id, "abbrev": away_abbr},
        "events": events_out
    }

    ensure_out_dir(out_dir)
    out_path = os.path.join(out_dir, f"pbp_onice_{game_pk}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2, separators=(",", ": "))

    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Emit per-game PBP+on-ice JSON with shift-change events and per-player TOI at each event.")
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--raw", type=str, required=True, help="Directory containing pbp/, boxscore/, shiftcharts/")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    path = build(args.game, args.raw, args.out)
    print(f"Wrote {path}")
