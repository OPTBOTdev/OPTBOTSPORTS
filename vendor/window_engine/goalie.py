#!/usr/bin/env python3
import argparse, json, time, random
from typing import Dict, List, Any, Optional, Tuple, Iterable
from datetime import datetime
import requests

APIWEB = "https://api-web.nhle.com/v1"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# ---------------- Logging ----------------
def log(msg: str, quiet: bool = False):
    if not quiet:
        print(msg, flush=True)

# ---------------- HTTP ----------------
class Http:
    def __init__(self, retries=3, backoff=0.6, timeout=30, quiet=False):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self.quiet = quiet

    def get_json(self, url: str, params: dict | None = None) -> Any:
        last_err = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429,) or r.status_code >= 500:
                    raise requests.RequestException(f"HTTP {r.status_code}")
                r.raise_for_status()
            except Exception as e:
                last_err = e
                if attempt == self.retries:
                    raise
                sleep_s = self.backoff * (2 ** (attempt - 1)) + random.random() * 0.25
                log(f"[http] retry {attempt}/{self.retries} after error on {url}: {e} (sleep {sleep_s:.2f}s)", self.quiet)
                time.sleep(sleep_s)
        raise last_err if last_err else RuntimeError("Unknown HTTP error")

# --------------- helpers ---------------
def safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def iso_date_from_box(box: dict) -> Optional[str]:
    s = (box.get("gameDate") or box.get("startTimeUTC") or "").replace("Z", "")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.split(".")[0]).date().isoformat()
    except Exception:
        return None

def parse_toi_seconds(s: Any) -> int:
    """Parse 'MM:SS' or 'HH:MM:SS' to seconds."""
    if not isinstance(s, str) or not s:
        return 0
    parts = s.split(":")
    try:
        if len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            return m * 60 + sec
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + sec
    except Exception:
        return 0
    return 0

def parse_pair(s: Any) -> Tuple[int, int]:
    """
    Parse strings like '25/27' → (saves, shots).
    Returns (0, 0) on failure.
    """
    if not isinstance(s, str) or "/" not in s:
        return 0, 0
    left, right = s.split("/", 1)
    return safe_int(left), safe_int(right)

# ---- team & schedule discovery ----
STATIC_ABBRS = ["ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM",
                "FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT",
                "SEA","SJS","STL","TBL","TOR","UTA","VAN","VGK","WSH","WPG"]

def normalize_team_filter(team_args: Optional[List[str]]) -> Optional[List[str]]:
    if not team_args:
        return None
    out = []
    for t in team_args:
        t = (t or "").strip().upper()
        if not t:
            continue
        # allow comma-separated
        out.extend([p.strip().upper() for p in t.split(",") if p.strip()])
    # de-dupe and keep only plausible abbrs
    out = [t for t in dict.fromkeys(out)]
    return out or None

def normalize_type_filter(type_args: Optional[List[str]]) -> Optional[List[str]]:
    if not type_args:
        return None
    out: List[str] = []
    for t in type_args:
        t = (t or "").strip()
        if not t:
            continue
        parts = [p.strip().upper() for p in t.split(",") if p.strip()]
        for p in parts:
            if p in ("1", "PRE", "PRESEASON"): out.append("PRE"); continue
            if p in ("2", "REG", "REGULAR"):   out.append("REG"); continue
            if p in ("3", "PST", "PLAYOFFS", "POSTSEASON"): out.append("PST"); continue
            # keep any other code as-is
            out.append(p)
    # de-dupe while preserving order
    out = [t for t in dict.fromkeys(out)]
    return out or None

def fetch_teams_from_standings(http: Http, quiet: bool) -> List[dict]:
    log("[teams] fetching standings/now …", quiet)
    rows = []
    try:
        data = http.get_json(f"{APIWEB}/standings/now")
        if isinstance(data, dict) and isinstance(data.get("standings"), list):
            for rec in data["standings"]:
                tid = rec.get("teamId")
                abbr = rec.get("teamAbbrev") or rec.get("triCode")
                name = rec.get("teamName") or rec.get("teamFullName")
                if isinstance(tid, int) and isinstance(abbr, str):
                    rows.append({"id": tid, "abbrev": abbr, "name": name})
    except Exception as e:
        log(f"[teams] standings/now error: {e}", quiet)

    if rows:
        abbrs_preview = ", ".join(sorted({r["abbrev"] for r in rows})[:12])
        log(f"[teams] discovered {len(rows)} teams (e.g., {abbrs_preview}…)", quiet)
        return rows

    log("[teams] WARNING: no teams from standings/now; using static list", quiet)
    return [{"id": None, "abbrev": a, "name": ""} for a in STATIC_ABBRS]

def _game_type_code(g: dict) -> str:
    raw = g.get("seasonType", g.get("gameType", g.get("gameState", "")))
    code = str(raw).upper() if not isinstance(raw, int) else str(raw)
    if code in ("1","PRE","PRESEASON"): return "PRE"
    if code in ("2","REG","REGULAR"):   return "REG"
    if code in ("3","PST","PLAYOFFS","POSTSEASON"): return "PST"
    return code

def get_club_schedule(http: Http, abbr: str, season: str, quiet: bool) -> List[dict | int]:
    url = f"{APIWEB}/club-schedule-season/{abbr}/{season}"
    try:
        data = http.get_json(url)
    except Exception as e:
        log(f"[sched] {abbr} ERROR fetching schedule: {e}", quiet)
        return []

    raw: List[dict | int] = []
    if isinstance(data, dict):
        if isinstance(data.get("games"), list):
            raw.extend(data["games"])
        if isinstance(data.get("gameWeek"), list):
            for wk in data["gameWeek"]:
                if isinstance(wk, dict) and isinstance(wk.get("games"), list):
                    raw.extend(wk["games"])
    elif isinstance(data, list):
        raw.extend(data)

    by_id: dict[int, dict | int] = {}
    for g in raw:
        if isinstance(g, int):
            by_id[g] = g
        elif isinstance(g, dict):
            gid = g.get("id") or g.get("gameId") or g.get("gamePk")
            if isinstance(gid, int):
                by_id[gid] = g

    typed = [g for g in by_id.values() if isinstance(g, dict)]
    if typed:
        pre_n = sum(1 for g in typed if _game_type_code(g) == "PRE")
        reg_n = sum(1 for g in typed if _game_type_code(g) == "REG")
        pst_n = sum(1 for g in typed if _game_type_code(g) == "PST")
        oth_n = len(typed) - pre_n - reg_n - pst_n
        log(f"[sched] {abbr} -> total={len(by_id)} (typed={len(typed)}; PRE={pre_n}, REG={reg_n}, PST={pst_n}, other={oth_n})", quiet)
    else:
        log(f"[sched] {abbr} -> total={len(by_id)} (no type fields)", quiet)

    return list(by_id.values())

def fetch_boxscore_apiweb(http: Http, gamePk: int) -> dict | None:
    try:
        return http.get_json(f"{APIWEB}/gamecenter/{gamePk}/boxscore")
    except Exception:
        return None

def _is_reg_from_box(box: dict | None) -> bool:
    if not isinstance(box, dict):
        return False
    gt = box.get("gameType") or (box.get("gameInfo") or {}).get("gameType")
    if isinstance(gt, int):  return gt == 2
    if isinstance(gt, str):  return gt.strip().upper() in ("2", "REG", "REGULAR")
    return False

def _game_type_from_box(box: dict | None) -> str:
    """Return standardized game type code: 'PRE', 'REG', 'PST' (or empty if unknown)."""
    if not isinstance(box, dict):
        return ""
    gt = box.get("gameType") or (box.get("gameInfo") or {}).get("gameType")
    if isinstance(gt, int):
        if gt == 1: return "PRE"
        if gt == 2: return "REG"
        if gt == 3: return "PST"
        return str(gt).upper()
    if isinstance(gt, str):
        val = gt.strip().upper()
        if val in ("1", "PRE", "PRESEASON"): return "PRE"
        if val in ("2", "REG", "REGULAR"):   return "REG"
        if val in ("3", "PST", "PLAYOFFS", "POSTSEASON"): return "PST"
        return val
    return ""

def _iter_abbrs(teams: List[dict], only_abbrs: Optional[List[str]]) -> Iterable[str]:
    all_abbrs = sorted({t["abbrev"] for t in teams if t.get("abbrev")})
    if not only_abbrs:
        return all_abbrs
    only_set = set(a.upper() for a in only_abbrs)
    # Filter and warn if any requested abbr not found
    selected = [a for a in all_abbrs if a.upper() in only_set]
    missing = [a for a in only_set if a not in set(all_abbrs)]
    if missing:
        log(f"[teams] WARNING: requested team(s) not found in discovery: {', '.join(sorted(missing))}")
    return selected

def discover_league_games(http: Http, season: str, quiet: bool,
                          only_abbrs: Optional[List[str]] = None,
                          allowed_types: Optional[List[str]] = None) -> List[Tuple[int, str]]:
    teams = fetch_teams_from_standings(http, quiet)
    abbrs = list(_iter_abbrs(teams, only_abbrs))
    log(f"[games] discovering games via club schedules for {len(abbrs)} team(s): {', '.join(abbrs)}", quiet)

    seen: dict[int, str] = {}
    for i, abbr in enumerate(abbrs, 1):
        sched = get_club_schedule(http, abbr, season, quiet)
        for g in sched:
            if isinstance(g, int):
                gid, dt_iso = g, ""
            else:
                gid = g.get("id") or g.get("gameId") or g.get("gamePk")
                dt_iso = ""
                for key in ("startTimeUTC", "gameDate"):
                    v = isinstance(g, dict) and g.get(key)
                    if v:
                        try:
                            dt_iso = datetime.fromisoformat(str(v).replace("Z","").split(".")[0]).date().isoformat()
                            break
                        except Exception:
                            pass
            if isinstance(gid, int) and gid not in seen:
                seen[gid] = dt_iso
        log(f"[games] processed {i}/{len(abbrs)} teams … (unique so far: {len(seen)})", quiet)
        time.sleep(0.03)

    gids = list(seen.keys())
    filtered: dict[int, str] = {}
    allowed = [t.upper() for t in (allowed_types or ["REG"])]
    log(f"[games] resolving dates & filtering to {','.join(allowed)} using boxscores for {len(gids)} games …", quiet)
    for idx, gid in enumerate(gids, 1):
        box = fetch_boxscore_apiweb(http, gid)
        gt = _game_type_from_box(box)
        if allowed and gt.upper() not in set(allowed):
            continue
        date_iso = (iso_date_from_box(box) or "") or seen.get(gid) or ""
        filtered[gid] = date_iso
        if idx % 100 == 0:
            log(f"[games]   checked {idx}/{len(gids)}", quiet)
        time.sleep(0.02)

    games = [(gid, filtered.get(gid) or "") for gid in filtered.keys()]
    games.sort(key=lambda t: (t[1] if t[1] else "9999-12-31", t[0]))
    log(f"[games] total unique games discovered (types={','.join(allowed)}): {len(games)}", quiet)
    return games

# -------- robust goalie parsing ----------
def _goalie_identity(p: dict) -> Tuple[int, str]:
    # ID fallbacks
    gid = (p.get("playerId")
           or (p.get("player") or {}).get("id")
           or (p.get("person") or {}).get("id")
           or (p.get("player") or {}).get("playerId")
           or (p.get("person") or {}).get("playerId")
           or p.get("id"))
    goalie_id = safe_int(gid)

    # Name fallbacks
    name = ""
    cand = p.get("name")
    if isinstance(cand, dict):
        cand = cand.get("default")
    if isinstance(cand, str) and cand.strip():
        name = cand.strip()
    else:
        for c in [
            (p.get("player") or {}).get("fullName"),
            (p.get("player") or {}).get("name"),
            (p.get("person") or {}).get("fullName"),
            (p.get("person") or {}).get("name"),
        ]:
            if isinstance(c, str) and c.strip():
                name = c.strip()
                break
    if not name:
        fn = p.get("firstName"); ln = p.get("lastName")
        if isinstance(fn, dict): fn = fn.get("default")
        if isinstance(ln, dict): ln = ln.get("default")
        if isinstance(fn, str) or isinstance(ln, str):
            name = f"{(fn or '').strip()} {(ln or '').strip()}".strip()

    return goalie_id, name

def _goalie_stats_block(p: dict) -> Tuple[int, int, int, bool]:
    """
    Return (shotsAgainst, saves, toi_seconds, starter_raw).

    Supports all of:
      - stats.goalieStats.*
      - p.goalieStats.*
      - top-level: shotsAgainst, saves, toi/timeOnIce, saveShotsAgainst ("25/27")
      - if total missing, sum from evenStrengthShotsAgainst/powerPlayShotsAgainst/shorthandedShotsAgainst
        where values are "saves/shots" strings.
    """
    stats = p.get("stats") or {}
    gs = stats.get("goalieStats") or p.get("goalieStats") or {}

    # Primary reads
    shots = safe_int(gs.get("shotsAgainst") or gs.get("shots"))
    saves = safe_int(gs.get("saves"))

    # --- Top-level fallbacks ---
    if shots == 0:
        shots = safe_int(p.get("shotsAgainst") or p.get("shots"))
    if saves == 0:
        saves = safe_int(p.get("saves"))

    # Parse "saveShotsAgainst": "25/27"
    ssa = p.get("saveShotsAgainst") or gs.get("saveShotsAgainst")
    if (saves == 0 or shots == 0) and isinstance(ssa, str) and "/" in ssa:
        s_left, s_right = parse_pair(ssa)
        if saves == 0: saves = s_left
        if shots == 0: shots = s_right

    # As a last resort, sum components if available
    if shots == 0 or saves == 0:
        comp_keys = [
            "evenStrengthShotsAgainst",
            "powerPlayShotsAgainst",
            "shorthandedShotsAgainst",
        ]
        total_saves = 0
        total_shots = 0
        for key in comp_keys:
            v = p.get(key) or gs.get(key)
            if isinstance(v, str) and "/" in v:
                l, r = parse_pair(v)
                total_saves += l
                total_shots += r
        if total_shots > 0:
            if shots == 0: shots = total_shots
            if saves == 0: saves = total_saves

    # TOI fallbacks: check gs, then stats, then top-level
    toi_s = 0
    for key in ("timeOnIce", "toi"):
        if key in gs:
            toi_s = parse_toi_seconds(gs.get(key)); break
        if key in stats:
            toi_s = parse_toi_seconds(stats.get(key)); break
    if toi_s == 0:
        toi_s = parse_toi_seconds(p.get("toi") or p.get("timeOnIce"))

    # Starter flag can live anywhere
    starter_raw = bool(
        p.get("starter") or gs.get("starter") or stats.get("starter")
    )

    return shots, saves, toi_s, starter_raw

def _compute_starter(goalie_rows_team: List[dict]) -> Optional[int]:
    """
    Decide starter goalie_id for a team in this game:
      1) if any row has starter_raw True → choose that
      2) else goalie with max TOI
      3) else goalie with shots>0
      4) else None
    """
    for r in goalie_rows_team:
        if r.get("starter_raw"):
            return r["goalie_id"]
    if goalie_rows_team:
        cand = max(goalie_rows_team, key=lambda r: r.get("toi_seconds", 0))
        if cand.get("toi_seconds", 0) > 0:
            return cand["goalie_id"]
    for r in goalie_rows_team:
        if r.get("shots_all", 0) > 0:
            return r["goalie_id"]
    return None

def build_goalie_pregame_sv(http: Http, season: str, pause: float, quiet: bool,
                            team_filter: Optional[List[str]] = None,
                            type_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    games = discover_league_games(http, season, quiet, only_abbrs=team_filter, allowed_types=type_filter)

    # cumulative history per goalie_id
    hist: Dict[int, Dict[str, int]] = {}  # {"shots": int, "saves": int, "games": int}
    out: List[Dict[str, Any]] = []

    total = len(games)
    log(f"[process] iterating {total} games …", quiet)
    for idx, (gamePk, date_iso) in enumerate(games, 1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            log(f"[process] game {idx}/{total} → gamePk={gamePk} date={date_iso or 'unknown'}", quiet)

        box = fetch_boxscore_apiweb(http, gamePk)
        if not isinstance(box, dict):
            log(f"[process]   WARNING: no boxscore for {gamePk}", quiet)
            time.sleep(pause)
            continue

        home_meta = box.get("homeTeam") or {}
        away_meta = box.get("awayTeam") or {}
        game_date = (iso_date_from_box(box) or "") or date_iso or ""
        pgs = (box.get("playerByGameStats") or {})

        # Collect goalie rows per side first (so we can decide 'starter')
        per_side_rows: Dict[str, List[dict]] = {"homeTeam": [], "awayTeam": []}

        for side_key, team_meta in (("homeTeam", home_meta), ("awayTeam", away_meta)):
            team_abbr = team_meta.get("abbrev") or team_meta.get("triCode") or ""
            team_id   = safe_int(team_meta.get("id"))

            # If user filtered teams, keep only games where at least one side matches
            if team_filter and team_abbr and team_abbr.upper() not in set(a.upper() for a in team_filter):
                # still parse both sides if the *other* side is filtered; we’ll handle below
                pass

            team_block = pgs.get(side_key, {}) or {}
            goalies_list = team_block.get("goalies") or []
            if not isinstance(goalies_list, list):
                goalies_list = []

            for p in goalies_list:
                if not isinstance(p, dict):
                    continue
                gid, name = _goalie_identity(p)
                if gid <= 0:
                    continue
                shots_all, saves_all, toi_s, starter_raw = _goalie_stats_block(p)
                per_side_rows[side_key].append({
                    "season": season,
                    "date": game_date,
                    "gamePk": gamePk,
                    "team_id": team_id,
                    "team_abbr": team_abbr,
                    "goalie_id": gid,
                    "goalie_name": name,
                    "shots_all": shots_all,
                    "saves_all": saves_all,
                    "toi_seconds": toi_s,
                    "starter_raw": starter_raw,
                })

        # If filtering by team, skip games that involve none of the requested teams
        if team_filter:
            side_abbrs = {r["team_abbr"].upper() for side in per_side_rows.values() for r in side}
            if side_abbrs.isdisjoint({t.upper() for t in team_filter}):
                time.sleep(pause)
                continue

        # Decide starters per side, then write rows with pregame + flags, then accumulate
        for side_key in ("homeTeam", "awayTeam"):
            team_rows = per_side_rows[side_key]
            if team_filter:
                # keep only rows for requested teams if filter is present
                team_rows = [r for r in team_rows if r["team_abbr"].upper() in {t.upper() for t in team_filter}]
            if not team_rows:
                continue

            starter_id = _compute_starter(team_rows)

            for row in team_rows:
                gid = row["goalie_id"]
                h = hist.setdefault(gid, {"shots": 0, "saves": 0, "games": 0})
                prior_shots = h["shots"]; prior_saves = h["saves"]; prior_games = h["games"]
                pregame_sv_all = (prior_saves / prior_shots) if prior_shots > 0 else None

                played = (row["toi_seconds"] > 0) or (row["shots_all"] > 0) or (row["saves_all"] > 0)
                starter = (starter_id == gid)

                out.append({
                    "season": season,
                    "date": row["date"],
                    "gamePk": row["gamePk"],
                    "team_id": row["team_id"],
                    "team_abbr": row["team_abbr"],
                    "goalie_id": gid,
                    "goalie_name": row["goalie_name"],
                    "shots_all": row["shots_all"],
                    "saves_all": row["saves_all"],
                    "toi_seconds": row["toi_seconds"],
                    "played": played,
                    "starter": starter,
                    "prior_games": prior_games,
                    "prior_shots_all": prior_shots,
                    "prior_saves_all": prior_saves,
                    "pregame_sv_all": pregame_sv_all,
                })

                # update cumulative only if they actually played
                if played:
                    h["shots"] += row["shots_all"]
                    h["saves"] += row["saves_all"]
                    h["games"] += 1

        time.sleep(pause)

    log(f"[done] built {len(out)} goalie-game rows", quiet)
    return out

# --------------- CLI ---------------
def main():
    ap = argparse.ArgumentParser(description="Pregame cumulative SV% & starter flags for NHL goalies (api-web only).")
    ap.add_argument("--season", required=True, help="Season like 20242025")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--pause", type=float, default=0.12, help="Seconds to sleep between requests")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    ap.add_argument("--team", action="append",
                    help="Filter by team abbr (e.g., TOR). You can repeat or comma-separate. Example: --team TOR --team BOS,MTL")
    ap.add_argument("--type", action="append",
                    help="Game type(s) to include. Accepts PRE, REG, PST (repeat or comma-separate). Default: REG")
    args = ap.parse_args()

    http = Http(quiet=args.quiet)
    team_filter = normalize_team_filter(args.team)
    type_filter = normalize_type_filter(args.type)
    if team_filter:
        log(f"[filter] limiting to team(s): {', '.join(team_filter)}", args.quiet)
    if type_filter:
        log(f"[filter] including game type(s): {', '.join(type_filter)}", args.quiet)
    else:
        log(f"[filter] including game type(s): REG (default)", args.quiet)

    data = build_goalie_pregame_sv(http, args.season, args.pause, args.quiet,
                                   team_filter=team_filter,
                                   type_filter=type_filter)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log(f"[write] Wrote {args.out} with {len(data)} rows.", args.quiet)

if __name__ == "__main__":
    main()
