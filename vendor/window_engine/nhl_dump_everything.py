#!/usr/bin/env python3
"""
NHL Capability Dumper — pulls EVERYTHING useful for a date range and writes:
  • Raw JSON for each endpoint (so you never lose fields)
  • Wide CSVs that include the union of all discovered keys (no guessing headers)

Endpoints covered per gamePk:
  - statsapi: /api/v1/game/{gamePk}/feed/live                → PBP (events + coordinates)
  - nhl stats REST: /stats/rest/en/shiftcharts?gameId=...    → shift charts (per-player TOI windows)
  - api-web: /v1/gamecenter/{gamePk}/boxscore                → player & team boxscore (enrichment/fallback)
Additionally per team (optional):
  - statsapi: /api/v1/teams/{teamId}?expand=team.roster      → roster & player metadata

Usage examples:
  python nhl_dump_everything.py --start 2025-02-10 --end 2025-02-12 --out artifacts/dumps --dump-raw
  python nhl_dump_everything.py --game 2024010086 --out artifacts/dumps --dump-raw

Outputs (under --out):
  raw/  → raw JSON blobs (pbp_*.json, shiftcharts_*.json, boxscore_*.json, roster_*.json)
  csv/  → wide CSVs  (pbp_*.csv, shiftcharts_*.csv, boxscore_players_*.csv, boxscore_teams_*.csv, roster_*.csv)

Safe to re-run: files are overwritten per game/team.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
import datetime as dt

import time
import random
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Please `pip install requests` first.")
    sys.exit(1)

STATSAPI = "https://statsapi.web.nhl.com/api/v1"
STATSREST = "https://api.nhle.com/stats/rest/en"
APIWEB   = "https://api-web.nhle.com/v1"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)

# -------------------- utils --------------------

def makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dump_raw_json(obj: Any, path: str) -> None:
    # ensure parent directories exist
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def flatten(d: Any, parent_key: str = "", out: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten nested dict/list → single-level dict using dot-keys.
       Lists of scalars → semicolon-joined; lists of dicts → JSON string.
    """
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            nk = f"{parent_key}.{k}" if parent_key else k
            flatten(v, nk, out)
    elif isinstance(d, list):
        if all(not isinstance(x, (dict, list)) for x in d):
            out[parent_key] = ";".join("" if x is None else str(x) for x in d)
        else:
            out[parent_key] = json.dumps(d, ensure_ascii=False)
    else:
        out[parent_key] = d
    return out

@dataclass
class Http:
    session: requests.Session
    retries: int = 3
    backoff: float = 0.6

    def get(self, url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Optional[requests.Response]:
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=timeout)
                if r.status_code == 200:
                    return r
                # retry on 5xx or throttle responses
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.RequestException(f"HTTP {r.status_code}")
                # 4xx that's not 429: don't retry
                return r
            except Exception as e:
                if attempt == self.retries:
                    print(f"GET failed {url}: {e}")
                    return None
                sleep_s = self.backoff * (2 ** (attempt - 1)) + random.random() * 0.25
                time.sleep(sleep_s)
        return None

# -------------------- schedule (gamePk discovery) --------------------

def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def fetch_schedule_games(http: Http, start: str, end: str) -> List[int]:
    """Use statsapi schedule to discover gamePk between start/end (YYYY-MM-DD)."""
    url = f"{STATSAPI}/schedule"
    params = {"startDate": start, "endDate": end}
    r = http.get(url, params=params)
    if not r:
        return []
    data = r.json() or {}
    games: List[int] = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            gamePk = g.get("gamePk")
            if isinstance(gamePk, int):
                games.append(gamePk)
    return games

# -------------------- per-game pulls --------------------

def fetch_pbp_statsapi(http: Http, gamePk: int) -> Optional[Dict[str, Any]]:
    r = http.get(f"{STATSAPI}/game/{gamePk}/feed/live")
    if not r:
        return None
    return r.json()

def extract_pbp_rows_from_statsapi(feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    plays = (feed or {}).get("liveData", {}).get("plays", {}).get("allPlays", [])
    out: List[Dict[str, Any]] = []
    for p in plays:
        row: Dict[str, Any] = {}
        row.update(flatten(p.get("about", {}), "about"))
        row.update(flatten(p.get("result", {}), "result"))
        row.update(flatten(p.get("coordinates", {}), "coordinates"))
        row.update(flatten(p.get("team", {}), "team"))
        row["players_json"] = json.dumps(p.get("players", []), ensure_ascii=False)
        # convenience ids
        row["gamePk"] = (feed or {}).get("gamePk")
        row["eventIdx"] = p.get("about", {}).get("eventIdx")
        row["eventId" ] = p.get("about", {}).get("eventId")
        out.append(row)
    return out

# Add api-web PBP fallback

def fetch_pbp_apiweb(http: Http, gamePk: int) -> Optional[Dict[str, Any]]:
    r = http.get(f"{APIWEB}/gamecenter/{gamePk}/play-by-play")
    if not r:
        return None
    return r.json()


def extract_pbp_rows_from_apiweb(gc: Dict[str, Any], gamePk: int) -> List[Dict[str, Any]]:
    plays = (gc or {}).get("plays", []) or []
    out: List[Dict[str, Any]] = []
    for p in plays:
        row: Dict[str, Any] = {}
        # Common fields
        row["gamePk"] = gamePk
        row["eventId"] = p.get("eventId") or p.get("eventIdx")
        row["type"] = p.get("typeDescKey") or p.get("typeCode")
        # Period/time
        row.update(flatten(p.get("periodDescriptor", {}), "period"))
        # Details include coords, result, strength, etc.
        row.update(flatten(p.get("details", {}), "details"))
        # Keep players array as JSON
        row["players_json"] = json.dumps(p.get("players", []), ensure_ascii=False)
        out.append(row)
    return out

# ---- PBP health validators ----
def validate_pbp_apiweb(gc: Dict[str, Any]) -> Dict[str, Any]:
    plays = (gc or {}).get("plays", []) or []
    icing_pending = False
    n_after_icing_faceoffs = 0
    n_faceoff = n_goal = n_icing = 0
    has_coords = has_zone = has_situation = False

    for p in plays:
        t = (p.get("typeDescKey") or "").lower()
        det = p.get("details", {}) or {}
        # sticky icing: explicit icing OR stoppage(reason=icing)
        is_icing = (t == "icing") or (t == "stoppage" and (det.get("reason") or "").lower() == "icing")
        if is_icing:
            n_icing += 1
            icing_pending = True
        elif t == "faceoff":
            n_faceoff += 1
            if icing_pending:
                n_after_icing_faceoffs += 1
            icing_pending = False
        elif t == "goal":
            n_goal += 1
        # do not reset icing_pending on other admin events

        if det.get("xCoord") is not None:
            has_coords = True
        if det.get("zoneCode"):
            has_zone = True
        # situationCode is on the play root for api-web
        if p.get("situationCode"):
            has_situation = True

    return {
        "source": "api-web",
        "plays": len(plays),
        "faceoffs": n_faceoff,
        "goals": n_goal,
        "icings": n_icing,
        "after_icing_faceoffs": n_after_icing_faceoffs,
        "has_coords": has_coords,
        "has_zone": has_zone,
        "has_situation": has_situation,
    }

def validate_pbp_statsapi(feed: Dict[str, Any]) -> Dict[str, Any]:
    allp = (feed or {}).get("liveData", {}).get("plays", {}).get("allPlays", []) or []
    n_faceoff = sum(1 for p in allp if (p.get("result", {}) or {}).get("eventTypeId") == "FACEOFF")
    n_goal    = sum(1 for p in allp if (p.get("result", {}) or {}).get("eventTypeId") == "GOAL")
    n_icing   = sum(1 for p in allp if (p.get("result", {}) or {}).get("eventTypeId") == "ICING")
    return {
        "source": "statsapi",
        "plays": len(allp),
        "faceoffs": n_faceoff,
        "goals": n_goal,
        "icings": n_icing,
        "after_icing_faceoffs": None,
        "has_coords": any(p.get("coordinates") for p in allp),
        "has_zone": False,
        "has_situation": any(((p.get("result", {}) or {}).get("strength")) for p in allp),
    }

# stats REST — shift charts

def fetch_shiftcharts(http: Http, gamePk: int, debug: bool = False) -> Optional[List[Dict[str, Any]]]:
    url = f"{STATSREST}/shiftcharts"
    params = {"cayenneExp": f"gameId={gamePk}"}
    if debug:
        print(f"[shifts-debug] GET {url} params={params}")
    r = http.get(url, params=params)
    if not r:
        if debug:
            print(f"[shifts-debug] request failed or no response for game {gamePk}")
        return None
    if debug:
        try:
            size = len(r.text)
        except Exception:
            size = -1
        print(f"[shifts-debug] status={r.status_code} bytes={size}")
    try:
        data = r.json() or {}
    except Exception as e:
        if debug:
            print(f"[shifts-debug] JSON parse error for game {gamePk}: {e}")
        return None
    if debug:
        if isinstance(data, dict):
            keys = list(data.keys())
            n = len(data.get("data", []) if isinstance(data.get("data"), list) else [])
            print(f"[shifts-debug] payload keys={keys} rows_in_data={n}")
            if n == 0:
                # Show a small preview of payload for troubleshooting
                preview = {k: (type(v).__name__) for k, v in list(data.items())[:5]}
                print(f"[shifts-debug] empty data; payload preview={preview}")
        else:
            print(f"[shifts-debug] unexpected payload type: {type(data).__name__}")
    return data.get("data", [])

# api-web — boxscore (enrichment)

def fetch_boxscore_apiweb(http: Http, gamePk: int) -> Optional[Dict[str, Any]]:
    r = http.get(f"{APIWEB}/gamecenter/{gamePk}/boxscore")
    if not r:
        return None
    return r.json()

# statsapi — team roster (optional)

def fetch_team_roster(http: Http, team_id: int) -> Optional[Dict[str, Any]]:
    r = http.get(f"{STATSAPI}/teams/{team_id}", params={"expand": "team.roster"})
    if not r:
        return None
    return r.json()

# -------------------- writers (wide CSVs) --------------------

def write_wide_csv(rows: List[Dict[str, Any]], path: str) -> int:
    # ensure parent directories exist
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    if not rows:
        # still write headerless file for traceability
        with open(path, "w", newline="", encoding="utf-8") as f:
            pass
        return 0
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)

# PBP → wide CSV + raw

def write_pbp_everything(http: Http, gamePk: int, raw_dir: str, csv_dir: str, dump_raw: bool, raw_only: bool) -> int:
    # IMPORTANT: downstream window/build scripts expect api-web PBP shape
    # (top-level "plays" array). Prefer api-web, fall back to statsapi only
    # if api-web is unavailable for a given game.
    gc = fetch_pbp_apiweb(http, gamePk)
    raw_path = os.path.join(raw_dir, "pbp", f"{gamePk}.json")
    csv_path = os.path.join(csv_dir, "pbp", f"pbp_{gamePk}.csv")
    if gc is not None:
        if dump_raw or raw_only:
            dump_raw_json(gc, raw_path)
        # write health
        health = validate_pbp_apiweb(gc)
        hp = Path(raw_dir) / "pbp_health" / f"{gamePk}.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        health.update({"gameId": int(gamePk)})
        hp.write_text(json.dumps(health, indent=2))
        if raw_only:
            print(f"[pbp] raw-only written for game {gamePk}")
            return 0
        rows = extract_pbp_rows_from_apiweb(gc, gamePk)
    else:
        feed = fetch_pbp_statsapi(http, gamePk)
        if feed is None:
            print(f"[pbp] no feed for game {gamePk}")
            return 0
        if dump_raw or raw_only:
            dump_raw_json(feed, raw_path)
        # write health
        health = validate_pbp_statsapi(feed)
        hp = Path(raw_dir) / "pbp_health" / f"{gamePk}.json"
        hp.parent.mkdir(parents=True, exist_ok=True)
        health.update({"gameId": int(gamePk)})
        hp.write_text(json.dumps(health, indent=2))
        if raw_only:
            print(f"[pbp] raw-only written for game {gamePk}")
            return 0
        rows = extract_pbp_rows_from_statsapi(feed)
    n = write_wide_csv(rows, csv_path)
    print(f"[pbp] game {gamePk}: {n} plays")
    return n

# Shiftcharts → wide CSV + raw

def write_shiftcharts_everything(http: Http, gamePk: int, raw_dir: str, csv_dir: str, dump_raw: bool, raw_only: bool, debug_shifts: bool = False) -> int:
    data = fetch_shiftcharts(http, gamePk, debug=debug_shifts)
    if data is None:
        print(f"[shifts] no data for game {gamePk}")
        return 0
    raw_path = os.path.join(raw_dir, "shiftcharts", f"{gamePk}.json")
    csv_path = os.path.join(csv_dir, "shiftcharts", f"shiftcharts_{gamePk}.csv")
    if dump_raw or raw_only:
        dump_raw_json(data, raw_path)
        if debug_shifts:
            try:
                sz = os.path.getsize(raw_path) if os.path.exists(raw_path) else 0
            except Exception:
                sz = 0
            print(f"[shifts-debug] wrote raw to {raw_path} (bytes={sz})")
    if raw_only:
        print(f"[shifts] raw-only written for game {gamePk}")
        return 0
    rows = [flatten(d) for d in data]
    n = write_wide_csv(rows, csv_path)
    if debug_shifts and n == 0:
        print(f"[shifts-debug] zero rows written for game {gamePk}; csv_path={csv_path}")
    print(f"[shifts] game {gamePk}: {n} rows")
    return n

# Boxscore → players_wide.csv, teams_wide.csv + raw

def write_boxscore_everything(http: Http, gamePk: int, raw_dir: str, csv_dir: str, dump_raw: bool, raw_only: bool) -> Tuple[int, int]:
    data = fetch_boxscore_apiweb(http, gamePk)
    if data is None:
        print(f"[box] no data for game {gamePk}")
        return (0, 0)
    raw_path = os.path.join(raw_dir, "boxscore", f"{gamePk}.json")
    if dump_raw or raw_only:
        dump_raw_json(data, raw_path)
    if raw_only:
        print(f"[box] raw-only written for game {gamePk}")
        return (0, 0)

    teams_rows: List[Dict[str, Any]] = []
    for side_key in ("homeTeam", "awayTeam"):
        t = (data.get("boxscore", {}) or {}).get(side_key, {}) or {}
        row: Dict[str, Any] = {"side": side_key}
        row.update(flatten(t))
        teams_rows.append(row)
    teams_csv_path = os.path.join(csv_dir, "boxscore", "teams", f"boxscore_teams_{gamePk}.csv")
    n_teams = write_wide_csv(teams_rows, teams_csv_path)

    players_rows: List[Dict[str, Any]] = []
    pstats = (data.get("playerByGameStats", {}) or {})
    for side_key in ("homeTeam", "awayTeam"):
        team = pstats.get(side_key, {}) or {}
        abbrev = team.get("abbrev") or team.get("triCode")
        for group in ("forwards", "defense", "goalies"):
            for p in team.get(group, []) or []:
                person = p.get("player") or p.get("person") or {}
                stats   = p.get("stats") or {}
                row = {
                    "side": side_key,
                    "abbrev": abbrev,
                    "group": group,
                    "playerId": person.get("id") or person.get("idCode") or person.get("playerId"),
                    "name": person.get("name") or person.get("fullName") or "",
                    "raw_stats_json": json.dumps(stats, ensure_ascii=False),
                    "onIce_json": json.dumps(p.get("onIce", []), ensure_ascii=False),
                    "onIcePlusMinusStats_json": json.dumps(p.get("onIcePlusMinusStats", {}), ensure_ascii=False),
                }
                if isinstance(stats.get("skaterStats"), dict):
                    for k, v in stats["skaterStats"].items():
                        row[f"skater_{k}"] = v
                if isinstance(stats.get("goalieStats"), dict):
                    for k, v in stats["goalieStats"].items():
                        row[f"goalie_{k}"] = v
                players_rows.append(row)
    players_csv_path = os.path.join(csv_dir, "boxscore", "players", f"boxscore_players_{gamePk}.csv")
    n_players = write_wide_csv(players_rows, players_csv_path)
    print(f"[box] game {gamePk}: teams={n_teams} rows, players={n_players} rows")
    return (n_teams, n_players)

# Roster (optional) → wide CSV + raw

def write_roster_everything(http: Http, team_id: int, raw_dir: str, csv_dir: str, dump_raw: bool, raw_only: bool) -> int:
    data = fetch_team_roster(http, team_id)
    if data is None:
        print(f"[roster] no data for team {team_id}")
        return 0
    raw_path = os.path.join(raw_dir, "roster", f"{team_id}.json")
    if dump_raw or raw_only:
        dump_raw_json(data, raw_path)
    if raw_only:
        print(f"[roster] raw-only written for team {team_id}")
        return 0
    team = (data or {}).get("teams", [{}])[0]
    roster = (team or {}).get("roster", {}).get("roster", [])
    rows: List[Dict[str, Any]] = []
    for r in roster:
        row = {}
        row.update(flatten(team.get("teamName", ""), "teamName"))
        person = r.get("person", {})
        row.update(flatten(person, "person"))
        row.update(flatten(r.get("position", {}), "position"))
        rows.append(row)
    roster_csv_path = os.path.join(csv_dir, "roster", f"roster_{team_id}.csv")
    n = write_wide_csv(rows, roster_csv_path)
    print(f"[roster] team {team_id}: {n} players")
    return n

# ---- Teams meta (conference/division) ---------------------------------

def fetch_teams_meta(http: Http, season_id: Optional[str] = None) -> dict:
    """Return {teamId: {abbrev,name,conference,division}} using api-web only (no statsapi calls)."""
    meta: Dict[str, Dict[str, Any]] = {}
    # Prefer api-web teams index
    r_idx = http.get(f"{APIWEB}/teams")
    if r_idx:
        try:
            data = r_idx.json() or {}
        except Exception:
            data = {}
        rows = data.get("teams") if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for t in (rows or []):
            tid = t.get("id") or t.get("teamId")
            if not isinstance(tid, int):
                continue
            meta[str(tid)] = {
                "abbrev": t.get("abbrev") or t.get("triCode") or t.get("teamAbbrev") or "",
                "name":   t.get("name") or t.get("teamName") or t.get("commonName") or "",
                "conference": str(t.get("conferenceAbbrev") or t.get("conference") or "").upper(),
                "division":   str(t.get("divisionAbbrev")   or t.get("division")   or "").upper(),
            }
    # Fallback to api-web standings if needed
    if not meta:
        r3 = http.get(f"{APIWEB}/standings/now")
        if r3:
            try:
                data3 = r3.json() or {}
            except Exception:
                data3 = {}
            rows = data3.get("standings") if isinstance(data3, dict) else (data3 if isinstance(data3, list) else [])
            for rec in (rows or []):
                tid = rec.get("teamId")
                if not isinstance(tid, int):
                    continue
                meta[str(tid)] = {
                    "abbrev": rec.get("teamAbbrev") or rec.get("triCode") or "",
                    "name":   rec.get("teamName") or rec.get("teamFullName") or rec.get("commonName") or "",
                    "conference": str(rec.get("conferenceAbbrev") or rec.get("conferenceName") or "").upper(),
                    "division":   str(rec.get("divisionAbbrev")   or rec.get("divisionName")   or "").upper(),
                }
    return meta


def write_teams_meta(http: Http, raw_dir: str, season_id: Optional[str] = None) -> int:
    """Write raw/teams_meta.json (safe to overwrite)."""
    meta = fetch_teams_meta(http, season_id)
    if not meta:
        print("[teams_meta] no data; skipping write")
        return 0
    path = os.path.join(raw_dir, "teams_meta.json")
    dump_raw_json(meta, path)
    print(f"[teams_meta] wrote {path} with {len(meta)} teams")
    return len(meta)

def season_from_gamepk(gamePk: int) -> str:
	try:
		y = int(str(gamePk)[:4])
		return f"{y}{y+1}"
	except Exception:
		return ""


def iso_date_from_box(box: dict) -> Optional[str]:
	s = (box.get("gameDate") or box.get("startTimeUTC") or "").replace("Z", "")
	if not s:
		return None
	try:
		return dt.datetime.fromisoformat(s.split(".")[0]).date().isoformat()
	except Exception:
		return None

def fetch_club_schedule_season(http: Http, team_abbr: str, season: str) -> Optional[Dict[str, Any]]:
    """Fetch api-web club season schedule for a team abbreviation (e.g., 'TOR', 'FLA')."""
    # api-web path typically: /v1/club-schedule-season/{TEAM}/{SEASON}
    r = http.get(f"{APIWEB}/club-schedule-season/{team_abbr}/{season}")
    if not r:
        return None
    try:
        return r.json()
    except Exception:
        return None

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Dump NHL endpoints (raw JSON + wide CSVs)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--game", type=int, help="Single gamePk to dump")
    g.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", type=str, default="artifacts/dumps", help="Output directory root")
    ap.add_argument("--dump-raw", action="store_true", help="Also write raw JSON alongside CSVs")
    ap.add_argument("--raw-only", action="store_true", help="Write only raw JSON (no CSVs)")
    ap.add_argument("--rosters", action="store_true", help="Also dump team rosters for teams in the games")
    # Optional: also build schedules JSON for a specific team/season without dumping all games
    ap.add_argument("--team", type=str, help="Team abbreviation (e.g., TOR) to also include in schedules JSON")
    ap.add_argument("--season", type=str, help="Season string like 20242025 for schedules JSON")
    ap.add_argument("--debug-shifts", action="store_true", help="Print verbose diagnostics for shiftcharts fetch/write")
    args = ap.parse_args()

    # HTTP session with UA
    s = requests.Session()
    s.headers.update({"User-Agent": UA})
    http = Http(s)

    out_root = args.out
    raw_dir = os.path.join(out_root, "raw")
    csv_dir = os.path.join(out_root, "csv")
    makedirs(raw_dir); makedirs(csv_dir)

    # Always refresh teams_meta so downstream builders have conferences/divisions
    try:
        write_teams_meta(http, raw_dir)
    except Exception as e:
        print(f"[teams_meta] failed: {e}")

    games: List[int] = []
    if args.game:
        games = [args.game]
    else:
        if not args.end:
            ap.error("--end is required when using --start")
        games = fetch_schedule_games(http, args.start, args.end)
        if not games:
            print("No games found for that range.")
            return

    # Effective raw write flag
    dump_raw_effective = args.dump_raw or args.raw_only

    # Optional: track teams seen for roster pulls
    teams_seen: set[int] = set()
    # Schedules collector for rest/b2b JSON
    schedules_by_team: Dict[str, set[str]] = defaultdict(set)
    season_hint: Optional[str] = None
    # Collect basic teams meta (abbr/name) from boxscores and club schedules as a fallback
    teams_meta_seen: Dict[str, Dict[str, Any]] = {}

    # Optionally collect team schedule for schedules JSON upfront
    if args.team and args.season:
        sched = fetch_club_schedule_season(http, args.team, args.season)
        if isinstance(sched, dict):
            club_elems = (sched.get("games") or sched.get("gameWeek", [])) or []
            for gobj in club_elems:
                # Some payloads nest under weeks; support both flat and nested
                if isinstance(gobj, dict) and "games" in gobj:
                    it = gobj.get("games") or []
                else:
                    it = [gobj]
                for gg in it:
                    try:
                        d_iso = iso_date_from_box(gg) or (gg.get("startTimeUTC") or gg.get("gameDate") or "")[:10]
                        ht = (gg.get("homeTeam") or {}).get("id")
                        at = (gg.get("awayTeam") or {}).get("id")
                        habbr = (gg.get("homeTeam") or {}).get("abbrev") or (gg.get("homeTeam") or {}).get("triCode")
                        aabbr = (gg.get("awayTeam") or {}).get("abbrev") or (gg.get("awayTeam") or {}).get("triCode")
                        if isinstance(ht, int) and d_iso:
                            schedules_by_team[str(ht)].add(d_iso)
                            if habbr:
                                teams_meta_seen.setdefault(str(ht), {"abbrev": habbr, "name": "", "conference": "", "division": ""})
                        if isinstance(at, int) and d_iso:
                            schedules_by_team[str(at)].add(d_iso)
                            if aabbr:
                                teams_meta_seen.setdefault(str(at), {"abbrev": aabbr, "name": "", "conference": "", "division": ""})
                    except Exception:
                        continue
            season_hint = args.season

    for gamePk in games:
        # Guard: ensure gamePk is an int; skip bad entries
        if not isinstance(gamePk, int):
            try:
                gamePk = int((gamePk or {}).get("id"))
            except Exception:
                print(f"[skip] invalid game id: {gamePk}")
                continue
        print(f"=== GAME {gamePk} ===")
        # PBP
        write_pbp_everything(http, gamePk, raw_dir, csv_dir, dump_raw_effective, args.raw_only)
        # Shift charts
        write_shiftcharts_everything(http, gamePk, raw_dir, csv_dir, dump_raw_effective, args.raw_only, debug_shifts=args.debug_shifts)
        # Boxscore
        box_n = write_boxscore_everything(http, gamePk, raw_dir, csv_dir, dump_raw_effective, args.raw_only)

        # Try to extract team ids from boxscore for roster pulls
        try:
            box_json_path = os.path.join(raw_dir, "boxscore", f"{gamePk}.json")
            jb = None
            if os.path.exists(box_json_path):
                with open(box_json_path, "r", encoding="utf-8") as f:
                    jb = json.load(f)
        except Exception:
            jb = None
        if jb is None:
            jb = fetch_boxscore_apiweb(http, gamePk)
        # Collect schedules (date + team ids)
        if isinstance(jb, dict):
            d_iso = iso_date_from_box(jb)
            home = (jb.get("homeTeam") or {}).get("id")
            away = (jb.get("awayTeam") or {}).get("id")
            habbr = (jb.get("homeTeam") or {}).get("abbrev") or (jb.get("homeTeam") or {}).get("triCode")
            aabbr = (jb.get("awayTeam") or {}).get("abbrev") or (jb.get("awayTeam") or {}).get("triCode")
            if isinstance(home, int) and d_iso:
                schedules_by_team[str(home)].add(d_iso)
                if habbr:
                    teams_meta_seen.setdefault(str(home), {"abbrev": habbr, "name": "", "conference": "", "division": ""})
            if isinstance(away, int) and d_iso:
                schedules_by_team[str(away)].add(d_iso)
                if aabbr:
                    teams_meta_seen.setdefault(str(away), {"abbrev": aabbr, "name": "", "conference": "", "division": ""})
            if season_hint is None:
                season_hint = season_from_gamepk(gamePk)
            # NEW: auto-enrich schedules with full club season for both teams
            try:
                season_auto = season_hint or season_from_gamepk(gamePk)
                abbr_h = ((jb.get("homeTeam") or {}).get("abbrev") or (jb.get("homeTeam") or {}).get("triCode"))
                abbr_a = ((jb.get("awayTeam") or {}).get("abbrev") or (jb.get("awayTeam") or {}).get("triCode"))
                for abbr in [abbr_h, abbr_a]:
                    if not abbr or not season_auto:
                        continue
                    sched_club = fetch_club_schedule_season(http, abbr, season_auto)
                    if isinstance(sched_club, dict):
                        club_elems2 = (sched_club.get("games") or sched_club.get("gameWeek", [])) or []
                        for ge in club_elems2:
                            it2 = (ge.get("games") or []) if isinstance(ge, dict) and "games" in ge else [ge]
                            for gg2 in it2:
                                try:
                                    d2 = iso_date_from_box(gg2) or (gg2.get("startTimeUTC") or gg2.get("gameDate") or "")[:10]
                                    ht2 = (gg2.get("homeTeam") or {}).get("id")
                                    at2 = (gg2.get("awayTeam") or {}).get("id")
                                    habbr2 = (gg2.get("homeTeam") or {}).get("abbrev") or (gg2.get("homeTeam") or {}).get("triCode")
                                    aabbr2 = (gg2.get("awayTeam") or {}).get("abbrev") or (gg2.get("awayTeam") or {}).get("triCode")
                                    if isinstance(ht2, int) and d2:
                                        schedules_by_team[str(ht2)].add(d2)
                                        if habbr2:
                                            teams_meta_seen.setdefault(str(ht2), {"abbrev": habbr2, "name": "", "conference": "", "division": ""})
                                    if isinstance(at2, int) and d2:
                                        schedules_by_team[str(at2)].add(d2)
                                        if aabbr2:
                                            teams_meta_seen.setdefault(str(at2), {"abbrev": aabbr2, "name": "", "conference": "", "division": ""})
                                except Exception:
                                    continue
            except Exception:
                pass

        # Track teams for optional roster dump
        try:
            for side in ("homeTeam", "awayTeam"):
                tid = ((jb.get("homeTeam" if side=="homeTeam" else "awayTeam", {}) or {}).get("id")) if isinstance(jb, dict) else None
                if isinstance(tid, int):
                    teams_seen.add(tid)
        except Exception:
            pass

    # Optional roster dump for all seen teams
    if args.rosters and teams_seen:
        for tid in sorted(teams_seen):
            write_roster_everything(http, tid, raw_dir, csv_dir, dump_raw_effective, args.raw_only)

    # Write schedules JSON
    if schedules_by_team and season_hint:
        out_sched = {tid: sorted(list(dates)) for tid, dates in schedules_by_team.items()}
        sched_path = os.path.join(raw_dir, f"schedules_{season_hint}.json")
        with open(sched_path, "w", encoding="utf-8") as f:
            json.dump(out_sched, f, indent=2)
        print(f"[schedules] wrote {sched_path} with {sum(len(v) for v in out_sched.values())} dates across {len(out_sched)} teams")

    # Re-attempt teams_meta with season_hint if missing or empty
    try:
        tm_path = os.path.join(raw_dir, "teams_meta.json")
        need_write = True
        if os.path.exists(tm_path):
            try:
                cur = json.load(open(tm_path, "r", encoding="utf-8"))
                need_write = not (isinstance(cur, dict) and len(cur) > 0)
            except Exception:
                need_write = True
        if need_write:
            # Try api-web-only write first
            wrote = write_teams_meta(http, raw_dir, season_hint)
            # If still nothing, write fallback from seen teams (abbr-only)
            if wrote == 0 and teams_meta_seen:
                fallback_path = os.path.join(raw_dir, "teams_meta.json")
                dump_raw_json(teams_meta_seen, fallback_path)
                print(f"[teams_meta] wrote fallback {fallback_path} with {len(teams_meta_seen)} teams (abbr only)")
    except Exception as e:
        print(f"[teams_meta] retry failed: {e}")

    print(f"Done. Wrote outputs under: {out_root}")

if __name__ == "__main__":
    main()
