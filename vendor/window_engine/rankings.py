#!/usr/bin/env python3
"""
Build evolving NHL standings for a season from api-web (REG season only).

Writes three CSVs into --out:
  - game_results_<season>.csv
  - standings_by_date_<season>.csv                 (end-of-day snapshots)
  - standings_after_each_game_<season>.csv         (immediately after each finished game)

Hardening vs earlier versions
-----------------------------
* Deterministic ordering by (ISO date, gamePk) for all processing.
* REG-season only: gather candidate IDs from /club-schedule-season, but include
  a game only if the boxscore says it's FINAL (or has a gameOutcome).
* "OT/SO" detection primarily from PBP (period >= 4 or any shootout events),
  with a fallback to box.gameInfo.gameOutcome.
* Stable league ranking (points → point% → GD → GF → team_id) while emitting
  competition-style ranks (1,2,2,4) plus a strict unique rank.
* Safer JSON parsing + retries.

Usage
-----
  python build_season_standings.py --season 20242025 --out ./artifacts/standings

Notes
-----
* Points = 2 win, 1 OT/SO loss, 0 regulation loss.
* Only finished games are counted. In-progress or future games are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

APIWEB = "https://api-web.nhle.com/v1"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ---------------- Logging ----------------

def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


# ---------------- HTTP ----------------

class Http:
    def __init__(self, retries: int = 3, backoff: float = 0.6, timeout: int = 30, quiet: bool = False):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self.quiet = quiet

    def get_json(self, url: str, params: dict | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (429,) or r.status_code >= 500:
                    raise requests.RequestException(f"HTTP {r.status_code}")
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt == self.retries:
                    raise
                sleep_s = self.backoff * (2 ** (attempt - 1)) + random.random() * 0.25
                log(f"[http] retry {attempt}/{self.retries} on {url}: {e} (sleep {sleep_s:.2f}s)", self.quiet)
                time.sleep(sleep_s)
        raise last_err if last_err else RuntimeError("Unknown HTTP error")


# ---------------- Helpers ----------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return 0


def iso_date_from_box(box: dict) -> Optional[str]:
    # Use startTimeUTC if present; both are ISO strings with 'Z'.
    s = (box.get("gameDate") or box.get("startTimeUTC") or "").replace("Z", "")
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.split(".")[0]).date().isoformat()
    except Exception:  # noqa: BLE001
        return None


def _game_type_code(g: dict) -> str:
    raw = g.get("seasonType", g.get("gameType", g.get("gameState", "")))
    code = str(raw).upper() if not isinstance(raw, int) else str(raw)
    if code in ("1", "PRE", "PRESEASON"):
        return "PRE"
    if code in ("2", "REG", "REGULAR"):
        return "REG"
    if code in ("3", "PST", "PLAYOFFS", "POSTSEASON"):
        return "PST"
    return code


def fetch_teams(http: Http, quiet: bool) -> List[str]:
    # Try /standings/now for team list; else fall back to static list
    try:
        data = http.get_json(f"{APIWEB}/standings/now")
        if isinstance(data, dict) and isinstance(data.get("standings"), list):
            def _abbr_from_row(r: dict) -> Optional[str]:
                v = r.get("teamAbbrev") or r.get("triCode") or r.get("abbrev")
                # some endpoints use {"default": "TOR"} shape
                if isinstance(v, dict):
                    v = v.get("default") or v.get("abbrev") or v.get("triCode")
                if isinstance(v, str) and v.strip():
                    return v.strip().upper()
                return None

            abbrs = sorted({a for r in data["standings"] if isinstance(r, dict) for a in [_abbr_from_row(r)] if a})
            if abbrs:
                log(f"[teams] discovered {len(abbrs)} via /standings/now", quiet)
                return abbrs
    except Exception as e:  # noqa: BLE001
        log(f"[teams] standings/now error: {e}", quiet)
    abbrs = [
        "ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET","EDM",
        "FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT","PHI","PIT",
        "SEA","SJS","STL","TBL","TOR","UTA","VAN","VGK","WSH","WPG",
    ]
    log(f"[teams] using static list ({len(abbrs)})", quiet)
    return abbrs


def get_team_schedule_ids(http: Http, abbr: str, season: str, quiet: bool) -> List[int]:
    url = f"{APIWEB}/club-schedule-season/{abbr}/{season}"
    try:
        data = http.get_json(url)
    except Exception as e:  # noqa: BLE001
        log(f"[sched] {abbr} error: {e}", quiet)
        return []
    raw: List[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("games"), list):
            raw += data["games"]
        if isinstance(data.get("gameWeek"), list):
            for wk in data["gameWeek"]:
                if isinstance(wk, dict) and isinstance(wk.get("games"), list):
                    raw += wk["games"]
    elif isinstance(data, list):
        raw += data
    ids: set[int] = set()
    for g in raw:
        if isinstance(g, int):
            ids.add(g)
        elif isinstance(g, dict):
            if _game_type_code(g) == "REG":
                gid = g.get("id") or g.get("gameId") or g.get("gamePk")
                if isinstance(gid, int):
                    ids.add(gid)
    return sorted(ids)


def fetch_box(http: Http, gid: int) -> Optional[dict]:
    try:
        return http.get_json(f"{APIWEB}/gamecenter/{gid}/boxscore")
    except Exception:  # noqa: BLE001
        return None


def fetch_pbp(http: Http, gid: int) -> Optional[dict]:
    try:
        return http.get_json(f"{APIWEB}/gamecenter/{gid}/play-by-play")
    except Exception:  # noqa: BLE001
        return None


def is_final(box: Optional[dict]) -> bool:
    if not isinstance(box, dict):
        return False
    # gameState is often "FINAL"; gameOutcome is OT/SO text when applicable
    state = str((box.get("gameState") or "").upper())
    if state in {"FINAL", "OFF"}:  # OFF shows up post-final on api-web
        return True
    info = box.get("gameInfo") or {}
    if str(info.get("gameOutcome") or "").upper() in {"OT", "SO", "SHOOTOUT"}:
        return True
    # If both teams have integer scores and the clock is 0 in P3+, treat as final (guard)
    try:
        hp = box.get("periodDescriptor") or {}
        if int(hp.get("number") or 0) >= 3 and safe_int((box.get("clock") or {}).get("timeRemaining") == 0):
            return True
    except Exception:
        pass
    return False


def went_beyond_reg(pbp: Optional[dict], box: Optional[dict]) -> bool:
    # Primary: PBP max period >= 4 OR contains shootout events
    if isinstance(pbp, dict):
        plays = pbp.get("plays") or []
        max_pd = 0
        shootout = False
        for p in plays:
            pd = (p.get("periodDescriptor") or {})
            max_pd = max(max_pd, safe_int(pd.get("number")))
            t = (p.get("typeDescKey") or "").lower()
            if "shootout" in t or t == "shootout":
                shootout = True
        if max_pd >= 4 or shootout:
            return True
    # Fallback: box.gameInfo.gameOutcome in {OT, SO}
    if isinstance(box, dict):
        info = box.get("gameInfo") or {}
        outc = (info.get("gameOutcome") or "").upper()
        if outc in ("OT", "SO", "SHOOTOUT"):
            return True
    return False


# ---------------- Ranking ----------------

def _rank_sort_tuple(tid: int, pts: int, gp: int, gf: int, ga: int) -> Tuple:
    point_pct = (pts / (2.0 * gp)) if gp > 0 else 0.0
    return (-pts, -point_pct, -(gf - ga), -gf, tid)


def rank_league(points_by_team: Dict[int, int], gp: Dict[int, int], gf: Dict[int, int], ga: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return (competition_rank, unique_rank) dicts.

    * competition_rank: 1,2,2,4 style (ties share rank).
    * unique_rank: strict 1..N using tie-break tuple (points → point% → GD → GF → team_id).
    """
    teams = set(points_by_team.keys()) | set(gp.keys())
    ordered = sorted(((tid, points_by_team[tid]) for tid in teams), key=lambda kv: _rank_sort_tuple(kv[0], kv[1], gp[kv[0]], gf[kv[0]], ga[kv[0]]))

    comp: Dict[int, int] = {}
    uniq: Dict[int, int] = {}
    last_pts: Optional[int] = None
    comp_rank = 0
    for i, (tid, pts) in enumerate(ordered, 1):
        uniq[tid] = i  # strict 1..N
        if last_pts is None or pts != last_pts:
            comp_rank = i
            last_pts = pts
        comp[tid] = comp_rank
    return comp, uniq


# ---------------- Core build ----------------

def build_standings_for_season(season: str, out_dir: str, pause: float, quiet: bool) -> None:
    ensure_dir(out_dir)
    http = Http(quiet=quiet)

    # 1) discover all REG game IDs
    abbrs = fetch_teams(http, quiet)
    seen: set[int] = set()
    game_ids: List[int] = []
    for i, abbr in enumerate(abbrs, 1):
        ids = get_team_schedule_ids(http, abbr, season, quiet)
        for gid in ids:
            if gid not in seen:
                seen.add(gid)
                game_ids.append(gid)
        if i % 4 == 0 or i == len(abbrs):
            log(f"[sched] processed {i}/{len(abbrs)} teams … games so far: {len(game_ids)}", quiet)
        time.sleep(0.02)

    # 2) fetch minimal info for each game (date, teams, score) — FINAL only
    games: List[Dict[str, Any]] = []
    for idx, gid in enumerate(sorted(game_ids), 1):
        box = fetch_box(http, gid)
        if not isinstance(box, dict):
            continue
        if not is_final(box):  # skip in-progress/future
            continue

        date_iso = iso_date_from_box(box) or ""
        h = box.get("homeTeam") or {}
        a = box.get("awayTeam") or {}
        home_id = safe_int(h.get("id") or h.get("teamId"))
        away_id = safe_int(a.get("id") or a.get("teamId"))
        home_abbr = (h.get("abbrev") or h.get("triCode") or "")
        away_abbr = (a.get("abbrev") or a.get("triCode") or "")
        hs = safe_int(h.get("score"))
        as_ = safe_int(a.get("score"))

        pbp = fetch_pbp(http, gid)
        bor = went_beyond_reg(pbp, box)

        games.append({
            "gamePk": gid,
            "date": date_iso,
            "home_id": home_id,
            "home_abbr": home_abbr,
            "away_id": away_id,
            "away_abbr": away_abbr,
            "home_goals": hs,
            "away_goals": as_,
            "beyond_reg": bor,
        })

        if idx % 100 == 0:
            log(f"[box] fetched {idx}/{len(game_ids)}", quiet)
        time.sleep(pause)

    # Sort deterministically (date then gamePk)
    games.sort(key=lambda r: (r["date"], r["gamePk"]))

    # 3) iterate games + update table, capturing snapshots
    gp: Dict[int, int] = defaultdict(int)
    w: Dict[int, int] = defaultdict(int)
    l: Dict[int, int] = defaultdict(int)
    otl: Dict[int, int] = defaultdict(int)
    pts: Dict[int, int] = defaultdict(int)
    gf: Dict[int, int] = defaultdict(int)
    ga: Dict[int, int] = defaultdict(int)

    def point_pct_of(tid: int) -> float:
        g = gp[tid]
        return (pts[tid] / (2.0 * g)) if g > 0 else 0.0

    game_rows: List[Dict[str, Any]] = []
    per_game_snapshots: List[Dict[str, Any]] = []  # standings right after each game
    per_date_snapshots: List[Dict[str, Any]] = []  # standings at end of each date

    cur_date: Optional[str] = None

    def flush_date_snapshot(date_iso: str) -> None:
        if not date_iso:
            return
        comp_rank, uniq_rank = rank_league(pts, gp, gf, ga)
        for tid in set(list(gp.keys()) + list(pts.keys())):
            per_date_snapshots.append({
                "date": date_iso,
                "team_id": tid,
                "gp": gp[tid],
                "wins": w[tid],
                "losses": l[tid],
                "ot_losses": otl[tid],
                "points": pts[tid],
                "point_pct": round(point_pct_of(tid), 6),
                "goals_for": gf[tid],
                "goals_against": ga[tid],
                "goal_diff": gf[tid] - ga[tid],
                "league_rank": comp_rank.get(tid),
                "league_rank_unique": uniq_rank.get(tid),
            })

    for g in games:
        if cur_date is None:
            cur_date = g["date"]
        if g["date"] != cur_date:
            flush_date_snapshot(cur_date)
            cur_date = g["date"]

        hid, aid = g["home_id"], g["away_id"]
        hs, as_ = g["home_goals"], g["away_goals"]
        bor = g["beyond_reg"]

        # record game
        game_rows.append({
            "date": g["date"],
            "gamePk": g["gamePk"],
            "home_id": hid,
            "home_abbr": g["home_abbr"],
            "home_goals": hs,
            "away_id": aid,
            "away_abbr": g["away_abbr"],
            "away_goals": as_,
            "beyond_reg": int(bor),
            "goal_diff": abs(hs - as_),
            "winner_team_id": (hid if hs > as_ else (aid if as_ > hs else 0)),
        })

        # update counts
        gp[hid] += 1
        gp[aid] += 1
        gf[hid] += hs
        ga[hid] += as_
        gf[aid] += as_
        ga[aid] += hs

        if hs > as_:
            w[hid] += 1
            if bor:
                otl[aid] += 1
                pts[aid] += 1
            else:
                l[aid] += 1
            pts[hid] += 2
        elif as_ > hs:
            w[aid] += 1
            if bor:
                otl[hid] += 1
                pts[hid] += 1
            else:
                l[hid] += 1
            pts[aid] += 2
        else:
            # Safeguard: treat as OT/SO tie (1+1) – shouldn't occur in REG-season data.
            otl[hid] += 1
            otl[aid] += 1
            pts[hid] += 1
            pts[aid] += 1

        # per-game snapshot (immediately after this game)
        comp_rank, uniq_rank = rank_league(pts, gp, gf, ga)
        for tid in (hid, aid):
            per_game_snapshots.append({
                "date": g["date"],
                "after_gamePk": g["gamePk"],
                "team_id": tid,
                "gp": gp[tid],
                "wins": w[tid],
                "losses": l[tid],
                "ot_losses": otl[tid],
                "points": pts[tid],
                "point_pct": round(point_pct_of(tid), 6),
                "goals_for": gf[tid],
                "goals_against": ga[tid],
                "goal_diff": gf[tid] - ga[tid],
                "league_rank": comp_rank.get(tid),
                "league_rank_unique": uniq_rank.get(tid),
            })

    # final date snapshot
    flush_date_snapshot(cur_date or "")

    # 4) write CSVs
    ensure_dir(out_dir)

    def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
        if not rows:
            open(path, "w").close()
            return
        fields = sorted({k for r in rows for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    game_csv = os.path.join(out_dir, f"game_results_{season}.csv")
    daily_csv = os.path.join(out_dir, f"standings_by_date_{season}.csv")
    after_each_csv = os.path.join(out_dir, f"standings_after_each_game_{season}.csv")

    write_csv(game_rows, game_csv)
    write_csv(per_date_snapshots, daily_csv)
    write_csv(per_game_snapshots, after_each_csv)

    log(f"[write] {game_csv} ({len(game_rows)} rows)")
    log(f"[write] {daily_csv} ({len(per_date_snapshots)} rows)")
    log(f"[write] {after_each_csv} ({len(per_game_snapshots)} rows)")


# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build evolving NHL standings for a season (api-web).")
    ap.add_argument("--season", required=True, help="Season like 20242025")
    ap.add_argument("--out", default="artifacts/standings", help="Output directory")
    ap.add_argument("--pause", type=float, default=0.03, help="Sleep between requests (seconds)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    build_standings_for_season(args.season, args.out, args.pause, args.quiet)


if __name__ == "__main__":
    main()
