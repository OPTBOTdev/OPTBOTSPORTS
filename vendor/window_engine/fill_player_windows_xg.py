import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------
# Helpers: IO
# -------------------------------

def _read_windows_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize common canonical columns
    if "gamePk" not in df.columns and "gameid" in df.columns:
        df["gamePk"] = df["gameid"]
    if "teamId" not in df.columns and "teamid" in df.columns:
        df["teamId"] = df["teamid"]
    # Coerce time columns if present
    for c in ["start_sec", "end_sec", "seconds", "clock_s", "period"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Player id present for player-level windows/rollups
    if "playerId" in df.columns:
        df["playerId"] = pd.to_numeric(df["playerId"], errors="coerce").fillna(0).astype(int)
    # Normalize common 0/1 flags that sometimes come through as NaN/blank in rollups
    for c in [
        "fo_O_start","fo_D_start","fo_seen_start","fo_won_start","fo_lost_start",
        "ai_OZ_start","ai_DZ_start",
        "after_icing","after_icing_by_team","after_icing_by_opponent",
        "long_change","home_last_change_opportunity",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def _read_shots_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize expected columns
    # Event type
    event_col = None
    for c in ["event_type", "type", "eventType", "event"]:
        if c in df.columns:
            event_col = c
            break
    if event_col is None:
        df["event_type"] = ""
        event_col = "event_type"
    df[event_col] = df[event_col].astype(str).str.upper()
    # Normalize many vendor spellings into NHL-style buckets
    def _norm_evt(v: str) -> str:
        t = str(v).upper()
        if "GOAL" in t:
            return "GOAL"
        if ("SHOT" in t and ("ON" in t or "SOG" in t)) or t in {"SHOT", "SOG", "SHOT_ON_GOAL", "SHOT-ON-GOAL"}:
            return "SHOT"
        if "MISS" in t:
            return "MISSED_SHOT"
        if "BLOCK" in t:
            return "BLOCKED_SHOT"
        return t
    df[event_col] = df[event_col].map(_norm_evt)

    # xG: prefer xg_use, then fall back to other known names
    if "xg_use" in df.columns:
        df["xg"] = df["xg_use"]
    elif "xg" in df.columns:
        pass
    else:
        for c in ["xG", "xg_value", "x_expected"]:
            if c in df.columns:
                df["xg"] = df[c]
                break
    if "xg" not in df.columns:
        df["xg"] = 0.0
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0.0)

    # Shooter / Scorer
    shooter_col = None
    for c in ["shooterId", "shooter_id", "shootingPlayerId", "shooting_player_id", "playerId", "player_id", "scorerId", "scorer_id"]:
        if c in df.columns:
            shooter_col = c
            break
    if shooter_col is None:
        df["shooterId"] = 0
        shooter_col = "shooterId"
    df[shooter_col] = pd.to_numeric(df[shooter_col], errors="coerce").fillna(0).astype(int)

    # Scorer
    if "scorerId" in df.columns:
        df["scorerId"] = pd.to_numeric(df["scorerId"], errors="coerce").fillna(0).astype(int)
    else:
        df["scorerId"] = df[shooter_col]

    # Assists
    for a in ["assist1PlayerId", "assist2PlayerId"]:
        if a in df.columns:
            df[a] = pd.to_numeric(df[a], errors="coerce").fillna(0).astype(int)
        else:
            df[a] = 0

    # Timing
    sec_col = None
    for c in ["sec_game", "sec", "second", "game_seconds"]:
        if c in df.columns:
            sec_col = c
            break
    derived = False
    if sec_col is None:
        # Try to derive from period + sec_in_period or timeInPeriod (mm:ss)
        period_col = None
        for c in ["period", "Period", "periodNumber"]:
            if c in df.columns:
                period_col = c
                break
        if period_col is not None and ("sec_in_period" in df.columns or "timeInPeriod" in df.columns):
            per = pd.to_numeric(df[period_col], errors="coerce").fillna(1).astype(int)
            if "sec_in_period" in df.columns:
                sip = pd.to_numeric(df["sec_in_period"], errors="coerce").fillna(0).astype(int)
            else:
                # Parse mm:ss
                def _mmss_to_sec(v: str) -> int:
                    try:
                        m, s = str(v).split(":")
                        return int(m) * 60 + int(s)
                    except Exception:
                        return 0
                sip = df["timeInPeriod"].astype(str).map(_mmss_to_sec).astype(int)
            df["sec_game"] = (per - 1) * 1200 + sip
            sec_col = "sec_game"
            derived = True
    if sec_col is None:
        # Last resort
        df["sec_game"] = 0
        sec_col = "sec_game"
    df[sec_col] = pd.to_numeric(df[sec_col], errors="coerce").fillna(0).astype(int)
    # If present but looks broken (mostly zeros) and we can derive, try again
    if (df[sec_col] == 0).mean() > 0.9 and not derived:
        period_col = None
        for c in ["period", "Period", "periodNumber"]:
            if c in df.columns:
                period_col = c
                break
        if period_col is not None and ("sec_in_period" in df.columns or "timeInPeriod" in df.columns):
            per = pd.to_numeric(df[period_col], errors="coerce").fillna(1).astype(int)
            if "sec_in_period" in df.columns:
                sip = pd.to_numeric(df["sec_in_period"], errors="coerce").fillna(0).astype(int)
            else:
                def _mmss_to_sec(v: str) -> int:
                    try:
                        m, s = str(v).split(":")
                        return int(m) * 60 + int(s)
                    except Exception:
                        return 0
                sip = df["timeInPeriod"].astype(str).map(_mmss_to_sec).astype(int)
            df["sec_game"] = (per - 1) * 1200 + sip
            sec_col = "sec_game"

    # Team ids (optional, not strictly needed for own_* stats)
    for c in ["teamId", "team_id", "shooterTeamId", "for_team_id", "eventOwnerTeamId"]:
        if c in df.columns:
            df["teamId"] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
            break
    if "teamId" not in df.columns:
        df["teamId"] = 0

    # Normalize unified column names for downstream
    df = df.rename(columns={
        event_col: "event_type",
        shooter_col: "shooterId",
        sec_col: "sec_game",
    })
    # Ensure is_goal flag exists and is aligned with event_type/feeds
    if "isGoal" in df.columns and "is_goal" not in df.columns:
        df["is_goal"] = pd.to_numeric(df["isGoal"], errors="coerce").fillna(0).astype(int)
    if "is_goal" not in df.columns:
        df["is_goal"] = 0
    # If event_type says GOAL, force is_goal=1
    df.loc[df["event_type"].astype(str).str.upper().eq("GOAL"), "is_goal"] = 1
    # Sort order if present
    so_col = None
    for c in ["sortOrder", "sortorder", "sort_order"]:
        if c in df.columns:
            so_col = c
            break
    if so_col:
        df["sortOrder"] = pd.to_numeric(df[so_col], errors="coerce").fillna(0.0).astype(float)
    else:
        df["sortOrder"] = 0.0
    return df


def _read_ages_json(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either flat {pid: age} or {players: {pid: {age_on_asof}}}
        out: Dict[str, float] = {}
        if isinstance(data, dict) and "players" in data and isinstance(data["players"], dict):
            for k, v in data["players"].items():
                try:
                    pid = str(int(k))
                except Exception:
                    continue
                try:
                    if isinstance(v, dict) and "age_on_asof" in v:
                        out[pid] = float(v["age_on_asof"])
                except Exception:
                    pass
        else:
            for k, v in (data or {}).items():
                try:
                    out[str(int(k))] = float(v)
                except Exception:
                    pass
        return out
    except Exception:
        return {}


def _read_birthdates_json(path: Optional[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
    """Return (birthDate_map, age_on_asof_map) if present in JSON.
    Expected formats:
      - {"players": {"8478402": {"birthDate": "1997-05-31", "age_on_asof": 20}, ...}}
      - {"8478402": "1997-05-31", ...}
      - {"8478402": {"birthDate": "1997-05-31"}, ...}
    """
    if not path:
        return {}, {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        players = data.get("players") if isinstance(data, dict) else None
        mapping_src = players if isinstance(players, dict) else (data if isinstance(data, dict) else {})
        bd: Dict[str, str] = {}
        aos: Dict[str, float] = {}
        for k, v in (mapping_src or {}).items():
            try:
                pid = str(int(k))
            except Exception:
                continue
            dob = None
            if isinstance(v, str):
                dob = v
            elif isinstance(v, dict):
                dob = v.get("birthDate") or v.get("playerBirthDate")
                try:
                    if "age_on_asof" in v and v["age_on_asof"] is not None:
                        aos[pid] = float(v["age_on_asof"])
                except Exception:
                    pass
            if isinstance(dob, str) and len(dob) >= 10:
                bd[pid] = dob[:10]
        return bd, aos
    except Exception:
        return {}, {}


def _compute_age_years(dob_str: Optional[str], on_date_str: Optional[str]) -> Optional[float]:
    if not dob_str or not on_date_str:
        return None
    try:
        dob = pd.to_datetime(str(dob_str)).date()
        day = pd.to_datetime(str(on_date_str)).date()
        return float(((day - dob).days) / 365.2425)
    except Exception:
        return None


# -------------------------------
# Core logic
# -------------------------------

def _build_contributions(
    shots_df: pd.DataFrame,
    onice: Optional[dict] = None,
    *,
    debug: bool = False,
    onice_json_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (contrib_df, own_contrib_df).

    contrib_df contains per-shot event rows used to compute xGF/xGA within windows.
    own_contrib_df has per-(playerId, sec_game) contributions:
      - own_xg, own_gf, own_sf, own_af, own_primary_assists, own_secondary_assists
    """
    if shots_df.empty:
        contrib_empty = pd.DataFrame(columns=[
            "sec_game", "teamId", "xg_sog", "is_goal"
        ])
        empty = pd.DataFrame(columns=[
            "playerId", "sec_game",
            "own_xg", "own_gf", "own_sf", "own_af",
            "own_primary_assists", "own_secondary_assists",
        ])
        return contrib_empty, empty

    events = shots_df.copy()

    # Define simple event class flags
    # Treat SHOT + GOAL as SOG; attempts add MISS/BLOCKED if present
    event_type = events["event_type"].astype(str).str.upper()
    def _flag(col: str) -> pd.Series:
        if col in events.columns:
            return pd.to_numeric(events[col], errors="coerce").fillna(0).astype(int)
        return pd.Series(0, index=events.index, dtype=int)

    is_goal = (event_type == "GOAL") | (_flag("is_goal") == 1)
    is_shot = (event_type == "SHOT") | (_flag("is_shot") == 1)
    is_miss = (event_type == "MISSED_SHOT") | (_flag("is_miss") == 1)
    is_block = (event_type == "BLOCKED_SHOT") | (_flag("is_block") == 1)

    # SOG = shot or goal
    is_sog = (is_shot | is_goal) | (_flag("is_sog") == 1)
    # Attempts = sog + miss + block; also treat any row with a numeric xg as an attempt fallback
    xg_series = pd.to_numeric(events.get("xg", 0.0), errors="coerce").fillna(0.0)
    is_attempt = (is_sog | is_miss | is_block) | (_flag("is_attempt") == 1) | (xg_series > 0)

    # Shooter-based contributions
    shooter_events = events[["sec_game", "shooterId", "xg"]].copy()
    shooter_events.rename(columns={"shooterId": "playerId"}, inplace=True)
    shooter_events["playerId"] = pd.to_numeric(shooter_events["playerId"], errors="coerce").fillna(0).astype(int)

    # own_xg should include all attempts (SOG + MISS + BLOCK)
    shooter_events["own_xg"] = shooter_events["xg"].where(is_attempt, 0.0)
    shooter_events["own_gf"] = is_goal.astype(int)
    shooter_events["own_sf"] = is_sog.astype(int)
    shooter_events["own_af"] = is_attempt.astype(int)

    shooter_events = shooter_events[[
        "playerId", "sec_game", "own_xg", "own_gf", "own_sf", "own_af"
    ]]

    # Assist-based contributions (only on goals)
    goal_rows = events[is_goal.values].copy()
    assist_frames: List[pd.DataFrame] = []
    if not goal_rows.empty:
        if "assist1PlayerId" in goal_rows.columns:
            a1 = goal_rows[["sec_game", "assist1PlayerId"]].rename(columns={"assist1PlayerId": "playerId"})
            a1["playerId"] = pd.to_numeric(a1["playerId"], errors="coerce").fillna(0).astype(int)
            a1 = a1[a1["playerId"] != 0]
            a1["own_primary_assists"] = 1
            assist_frames.append(a1[["playerId", "sec_game", "own_primary_assists"]])
        if "assist2PlayerId" in goal_rows.columns:
            a2 = goal_rows[["sec_game", "assist2PlayerId"]].rename(columns={"assist2PlayerId": "playerId"})
            a2["playerId"] = pd.to_numeric(a2["playerId"], errors="coerce").fillna(0).astype(int)
            a2 = a2[a2["playerId"] != 0]
            a2["own_secondary_assists"] = 1
            assist_frames.append(a2[["playerId", "sec_game", "own_secondary_assists"]])

    if assist_frames:
        assists = assist_frames[0]
        for fr in assist_frames[1:]:
            assists = pd.concat([assists, fr], ignore_index=True)
        # Merge shooter and assists on (playerId, sec)
        own = pd.merge(
            shooter_events,
            assists,
            on=["playerId", "sec_game"],
            how="outer",
        ).fillna({
            "own_xg": 0.0, "own_gf": 0, "own_sf": 0, "own_af": 0,
            "own_primary_assists": 0, "own_secondary_assists": 0,
        })
    else:
        own = shooter_events.copy()
        own["own_primary_assists"] = 0
        own["own_secondary_assists"] = 0

    own["own_xg"] = pd.to_numeric(own["own_xg"], errors="coerce").fillna(0.0)
    for c in ["own_gf", "own_sf", "own_af", "own_primary_assists", "own_secondary_assists"]:
        own[c] = pd.to_numeric(own[c], errors="coerce").fillna(0).astype(int)

    # Build per-event contribution rows for team xGF/xGA within windows
    # Use xG for every attempt (SOG + MISS + BLOCK). Goals get epsilon if xg missing/zero.
    xg_series = pd.to_numeric(events["xg"], errors="coerce").fillna(0.0)
    xg_vals = xg_series.values
    sec_game_series = events["sec_game"].astype(int)
    contrib_df = pd.DataFrame({
        "sec_game": sec_game_series,
        "teamId": pd.to_numeric(events.get("teamId", 0), errors="coerce").fillna(0).astype(int),
        "shooterId": pd.to_numeric(events.get("shooterId", 0), errors="coerce").fillna(0).astype(int),
        "xg_attempt": np.where(is_attempt.values, xg_vals, 0.0),
        "is_goal": is_goal.astype(int).values,
    })
    # carry scorerId if available (for precise GF credit)
    if "scorerId" in events.columns:
        contrib_df["scorerId"] = pd.to_numeric(events["scorerId"], errors="coerce").fillna(0).astype(int)
    # Carry attempt class flags so routed own_* can align with team GF/SF/AF
    contrib_df["is_sog"] = is_sog.astype(int).values
    contrib_df["is_attempt"] = is_attempt.astype(int).values
    # Pass assist IDs through for goal fallbacks
    for a in ("assist1PlayerId", "assist2PlayerId"):
        if a in events.columns:
            contrib_df[a] = pd.to_numeric(events[a], errors="coerce").fillna(0).astype(int)
        else:
            contrib_df[a] = 0
    # Ensure GF/GA alignment: for goals with zero xg, inject small epsilon so xGA/xGF are not 0 when GF/GA==1
    goal_eps = 1e-6
    mask_goal_zero = (contrib_df["is_goal"] == 1) & (pd.to_numeric(contrib_df["xg_attempt"], errors="coerce").fillna(0.0) <= 0.0)
    if mask_goal_zero.any():
        contrib_df.loc[mask_goal_zero, "xg_attempt"] = goal_eps
    return contrib_df, own


def _load_onice_index_from_pbp(onice_json_path: Optional[str]) -> Tuple[Dict[int, Dict[str, set]], Dict[int, int]]:
    """Build two indices from pbp_onice JSON:
    - onice_idx: {sec_game: {"home": set(pids), "away": set(pids)}}
    - prev_sec:  {sec_game: previous play's sec_game}
    If unavailable or malformed, returns empty dicts.
    """
    idx: Dict[int, Dict[str, set]] = {}
    prev_sec: Dict[int, int] = {}
    # Extra context embedded into prev_sec under special keys to avoid breaking callers
    shift_change_secs: Dict[int, int] = {}
    goal_onice_by_sec: Dict[int, Dict[str, set]] = {}
    goal_scorer_by_sec: Dict[int, List[int]] = {}
    goal_assists_by_sec: Dict[int, Tuple[int, int]] = {}
    onice_pre: Dict[int, Dict[str, set]] = {}
    onice_post: Dict[int, Dict[str, set]] = {}
    shift_so_min: Dict[int, float] = {}
    goal_so_min: Dict[int, float] = {}
    if not onice_json_path or not os.path.exists(onice_json_path):
        return idx, prev_sec
    try:
        with open(onice_json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # pbp_onice may be a list or a dict with various keys
        if isinstance(doc, list):
            plays = doc
        elif isinstance(doc, dict):
            plays = doc.get("plays") or doc.get("events") or doc.get("data") or []
        else:
            plays = []

        def _get_sec(p: dict) -> int:
            for k in ("sec_game", "sec", "second"):
                if k in p:
                    try:
                        return int(p.get(k) or 0)
                    except Exception:
                        pass
            # fallback from period + timeInPeriod "MM:SS"
            try:
                per = int((p.get("periodDescriptor") or {}).get("number") or p.get("period") or 1)
                mmss = str(p.get("timeInPeriod") or "00:00")
                m, s = mmss.split(":")
                return (per - 1) * 1200 + int(m) * 60 + int(s)
            except Exception:
                return 0

        # Sort by (sec, sortOrder) to compute previous-play map
        def _get_so(p: dict) -> float:
            try:
                return float(p.get("sortOrder") or 0.0)
            except Exception:
                return 0.0
        plays_sorted = sorted(plays, key=lambda x: (_get_sec(x), _get_so(x)))

        # Track previous DISTINCT second: for multiple plays at the same second,
        # prev_sec should point to the last distinct sec before this second.
        last_distinct_sec = None
        current_sec = None
        for p in plays_sorted:
            sec = _get_sec(p)
            if current_sec is None or sec != current_sec:
                # entering a new second; remember previous distinct second
                current_sec = sec
            # set only once per sec (first encounter), don't overwrite with same-sec plays
            if sec not in prev_sec:
                prev_sec[sec] = last_distinct_sec if last_distinct_sec is not None else max(0, sec - 1)
                # update last_distinct_sec AFTER assigning prev for this new second
                last_distinct_sec = sec
            oi = p.get("onice") or {}
            home_lst = oi.get("home") or []
            away_lst = oi.get("away") or []
            g_home = (oi.get("goalies") or {}).get("home")
            g_away = (oi.get("goalies") or {}).get("away")
            if not isinstance(home_lst, list) or not isinstance(away_lst, list):
                continue
            # Keep the latest snapshot per second for generic lookup
            idx[sec] = {
                "home": set(int(x) for x in home_lst if isinstance(x, int)),
                "away": set(int(x) for x in away_lst if isinstance(x, int)),
                "goalies": {
                    "home": int(g_home) if isinstance(g_home, int) else (int(g_home) if str(g_home).isdigit() else 0),
                    "away": int(g_away) if isinstance(g_away, int) else (int(g_away) if str(g_away).isdigit() else 0),
                },
            }
            # First snapshot seen per second → pre
            if sec not in onice_pre:
                onice_pre[sec] = {
                    "home": set(int(x) for x in home_lst if isinstance(x, int)),
                    "away": set(int(x) for x in away_lst if isinstance(x, int)),
                    "goalies": {
                        "home": int(g_home) if isinstance(g_home, int) else (int(g_home) if str(g_home).isdigit() else 0),
                        "away": int(g_away) if isinstance(g_away, int) else (int(g_away) if str(g_away).isdigit() else 0),
                    },
                }
            # Continuously update post with the latest snapshot
            onice_post[sec] = {
                "home": set(int(x) for x in home_lst if isinstance(x, int)),
                "away": set(int(x) for x in away_lst if isinstance(x, int)),
                "goalies": {
                    "home": int(g_home) if isinstance(g_home, int) else (int(g_home) if str(g_home).isdigit() else 0),
                    "away": int(g_away) if isinstance(g_away, int) else (int(g_away) if str(g_away).isdigit() else 0),
                },
            }
            # Track shift_change seconds explicitly
            try:
                if str(p.get("type", "")).lower() == "shift_change":
                    shift_change_secs[sec] = 1
                    so = _get_so(p)
                    if sec not in shift_so_min or so < shift_so_min[sec]:
                        shift_so_min[sec] = so
            except Exception:
                pass
            # Capture the goal event's own on-ice for this second (pre-change snapshot for credit)
            try:
                if str(p.get("type", "")).lower() == "goal":
                    goal_onice_by_sec[sec] = {
                        "home": set(int(x) for x in home_lst if isinstance(x, int)),
                        "away": set(int(x) for x in away_lst if isinstance(x, int)),
                    }
                    # capture scorer id and assists from details
                    try:
                        scorer = (p.get("details") or {}).get("scoringPlayerId")
                        if scorer is not None:
                            sid = int(scorer)
                            goal_scorer_by_sec.setdefault(sec, []).append(sid)
                    except Exception:
                        pass
                    try:
                        det = p.get("details") or {}
                        a1 = int(det.get("assist1PlayerId") or 0)
                        a2 = int(det.get("assist2PlayerId") or 0)
                        goal_assists_by_sec[sec] = (a1, a2)
                    except Exception:
                        pass
                    so = _get_so(p)
                    if sec not in goal_so_min or so < goal_so_min[sec]:
                        goal_so_min[sec] = so
            except Exception:
                pass
    except Exception:
        return {}, {}
    # Stash extra maps under reserved keys inside prev_sec for downstream use without changing signature
    prev_sec["__shift_change_secs__"] = shift_change_secs  # type: ignore[index]
    prev_sec["__goal_onice__"] = goal_onice_by_sec        # type: ignore[index]
    prev_sec["__goal_scorer_by_sec__"] = goal_scorer_by_sec  # type: ignore[index]
    prev_sec["__goal_assists_by_sec__"] = goal_assists_by_sec  # type: ignore[index]
    prev_sec["__onice_pre__"] = onice_pre                 # type: ignore[index]
    prev_sec["__onice_post__"] = onice_post               # type: ignore[index]
    prev_sec["__shift_so_min__"] = shift_so_min           # type: ignore[index]
    prev_sec["__goal_so_min__"] = goal_so_min             # type: ignore[index]
    return idx, prev_sec


def _aggregate_own_into_windows(windows_df: pd.DataFrame, own_df: pd.DataFrame, events_df: Optional[pd.DataFrame] = None, onice_idx: Optional[Dict[int, Dict[str, set]]] = None, prev_sec_map: Optional[Dict[int, int]] = None, debug: bool = False) -> pd.DataFrame:
    if windows_df.empty or own_df.empty:
        # Add zero columns to windows
        for c in [
            "own_xg_window", "own_gf_window", "own_sf_window", "own_af_window",
            "own_primary_assists_window", "own_secondary_assists_window",
        ]:
            if c not in windows_df.columns:
                windows_df[c] = 0.0 if c == "own_xg_window" else 0
        windows_df["own_xg_window"] = pd.to_numeric(windows_df["own_xg_window"], errors="coerce").fillna(0.0)
        for c in [
            "own_gf_window", "own_sf_window", "own_af_window",
            "own_primary_assists_window", "own_secondary_assists_window",
        ]:
            windows_df[c] = pd.to_numeric(windows_df[c], errors="coerce").fillna(0).astype(int)
        return windows_df

    # We need start/end seconds and playerId in windows
    required = ["playerId", "start_sec", "end_sec"]
    for req in required:
        if req not in windows_df.columns:
            raise KeyError(f"windows csv missing required column: {req}")

    # Prepare index to map back results
    windows_df = windows_df.copy()
    windows_df["__row_id__"] = np.arange(len(windows_df))

    # Keep only relevant columns for own_df
    own = own_df[[
        "playerId", "sec_game",
        "own_xg", "own_gf", "own_sf", "own_af",
        "own_primary_assists", "own_secondary_assists",
    ]].copy()

    # Fast path: group own events by player to reduce comparisons
    results = {
        "own_xg_window": np.zeros(len(windows_df), dtype=float),
        "own_gf_window": np.zeros(len(windows_df), dtype=int),
        "own_sf_window": np.zeros(len(windows_df), dtype=int),
        "own_af_window": np.zeros(len(windows_df), dtype=int),
        "own_primary_assists_window": np.zeros(len(windows_df), dtype=int),
        "own_secondary_assists_window": np.zeros(len(windows_df), dtype=int),
    }

    # Build dictionary of events per player for quick filtering
    own_by_player: Dict[int, pd.DataFrame] = {}
    for pid, grp in own.groupby("playerId", sort=False):
        own_by_player[int(pid)] = grp.sort_values("sec_game")

    # Iterate windows by player groups to minimize overhead
    for pid, win_grp in windows_df.groupby("playerId", sort=False):
        pid = int(pid) if not pd.isna(pid) else 0
        if pid == 0 or pid not in own_by_player:
            continue
        ev = own_by_player[pid]
        if ev.empty:
            continue
        # For each window, slice ev within [start_sec, end_sec)
        for idx, w in win_grp.iterrows():
            start_s = int(w.get("start_sec", 0))
            end_s = int(w.get("end_sec", 0))
            row_i = int(w["__row_id__"])
            # Filter by sec_game range
            mask = (ev["sec_game"] >= start_s) & (ev["sec_game"] < end_s)
            if not mask.any():
                continue
            block = ev.loc[mask]
            # Aggregate
            results["own_xg_window"][row_i] += float(block["own_xg"].sum())
            results["own_gf_window"][row_i] += int(block["own_gf"].sum())
            results["own_sf_window"][row_i] += int(block["own_sf"].sum())
            results["own_af_window"][row_i] += int(block["own_af"].sum())
            if "own_primary_assists" in block.columns:
                results["own_primary_assists_window"][row_i] += int(block["own_primary_assists"].sum())
            if "own_secondary_assists" in block.columns:
                results["own_secondary_assists_window"][row_i] += int(block["own_secondary_assists"].sum())

    # Attach results
    for c, arr in results.items():
        windows_df[c] = arr

    windows_df.drop(columns=["__row_id__"], inplace=True)

    # Compute opponent goalie_id at window start using on-ice map
    try:
        if onice_idx and ("start_sec" in windows_df.columns):
            goalie_col = []
            for _, w in windows_df.iterrows():
                s0 = int(pd.to_numeric(w.get("start_sec"), errors="coerce"))
                team_id = int(pd.to_numeric(w.get("teamId"), errors="coerce")) if "teamId" in windows_df.columns else 0
                g_id = 0
                s_idx = onice_idx.get(s0) if isinstance(onice_idx, dict) else None
                if s_idx and isinstance(s_idx, dict):
                    # Determine side by majority teamId among players present in this row's window at s0
                    home_set = s_idx.get("home", set()) or set()
                    away_set = s_idx.get("away", set()) or set()
                    # Count which side matches window teamId
                    def _side_matches(side_set: set) -> int:
                        cnt = 0
                        for pid in side_set:
                            for tup in rows_by_player.get(pid, []):
                                _i, _s0, _s1 = tup
                                if _s0 <= s0 < _s1 and int(windows_df.loc[_i, "teamId"]) == team_id:
                                    cnt += 1; break
                        return cnt
                    side_home = _side_matches(home_set)
                    side_away = _side_matches(away_set)
                    goalies = s_idx.get("goalies", {}) if isinstance(s_idx.get("goalies"), dict) else {}
                    if side_home > side_away:
                        g_id = int(goalies.get("away", 0) or 0)
                    elif side_away > side_home:
                        g_id = int(goalies.get("home", 0) or 0)
                goalie_col.append(int(g_id))
            windows_df["opponent_goalie_id_start"] = goalie_col
        else:
            # Ensure column exists even if we cannot compute
            if "opponent_goalie_id_start" not in windows_df.columns:
                windows_df["opponent_goalie_id_start"] = 0
    except Exception:
        # Do not fail xG fill if goalie inference has issues
        if "opponent_goalie_id_start" not in windows_df.columns:
            windows_df["opponent_goalie_id_start"] = 0

    # Compute xGF/xGA if shots events are provided
    if events_df is not None and not events_df.empty:
        # Ensure required columns
        if "teamId" not in windows_df.columns:
            raise KeyError("windows csv missing required column: teamId for xGF/xGA computation")
        windows_df["teamId"] = pd.to_numeric(windows_df["teamId"], errors="coerce").fillna(0).astype(int)
        ev = events_df[["sec_game", "teamId", "shooterId", "xg_attempt"]].copy()
        # carry flags/assist ids for routed own_* alignment with GF/SF/AF/AA
        for c in ["is_goal", "is_sog", "is_attempt", "assist1PlayerId", "assist2PlayerId"]:
            if c in events_df.columns and c not in ev.columns:
                ev[c] = events_df[c]
        ev["sec_game"] = pd.to_numeric(ev["sec_game"], errors="coerce").fillna(0).astype(int)
        ev["teamId"] = pd.to_numeric(ev["teamId"], errors="coerce").fillna(0).astype(int)
        ev["shooterId"] = pd.to_numeric(ev["shooterId"], errors="coerce").fillna(0).astype(int)

        # If teamId is missing (0), infer from the shooter's team in windows at that second
        if (ev["teamId"] == 0).any() and "playerId" in windows_df.columns and "start_sec" in windows_df.columns and "end_sec" in windows_df.columns:
            # Build per-player windows with teamId
            win_cols = [c for c in ["playerId", "teamId", "start_sec", "end_sec"] if c in windows_df.columns]
            wmap = windows_df[win_cols].copy()
            wmap["playerId"] = pd.to_numeric(wmap["playerId"], errors="coerce").fillna(0).astype(int)
            wmap["teamId"] = pd.to_numeric(wmap["teamId"], errors="coerce").fillna(0).astype(int)
            # Pre-group for quick lookup
            w_by_player: Dict[int, pd.DataFrame] = {}
            for pid, grp in wmap.groupby("playerId", sort=False):
                if int(pid) == 0:
                    continue
                w_by_player[int(pid)] = grp[["start_sec", "end_sec", "teamId"]].sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

            # Fill missing event teamIds
            zero_mask = ev["teamId"] == 0
            for i in ev[zero_mask].index:
                pid = int(ev.at[i, "shooterId"]) if not pd.isna(ev.at[i, "shooterId"]) else 0
                if pid == 0 or pid not in w_by_player:
                    continue
                sec = int(ev.at[i, "sec_game"]) if not pd.isna(ev.at[i, "sec_game"]) else 0
                rows = w_by_player[pid]
                # find window containing sec: start <= sec < end
                hit = rows[(rows["start_sec"] <= sec) & (sec < rows["end_sec"])].head(1)
                if not hit.empty:
                    ev.at[i, "teamId"] = int(hit.iloc[0]["teamId"]) or 0

        # Build per-player row index with intervals
        rows_by_player: Dict[int, List[Tuple[int,int,int]]] = {}
        for i, w in windows_df.iterrows():
            pid = int(w.get("playerId", 0)) if "playerId" in windows_df.columns else 0
            if pid == 0:
                continue
            s0 = int(w.get("start_sec", 0)); s1 = int(w.get("end_sec", 0))
            if s1 <= s0:
                continue
            rows_by_player.setdefault(pid, []).append((i, s0, s1))

        xgf_arr = np.zeros(len(windows_df), dtype=float)
        xga_arr = np.zeros(len(windows_df), dtype=float)

        # Precompute window boundary seconds for boundary-aware routing
        try:
            start_secs_set = set(int(s) for s in pd.to_numeric(windows_df["start_sec"], errors="coerce").dropna().astype(int).unique().tolist())
            end_secs_set = set(int(s) for s in pd.to_numeric(windows_df["end_sec"], errors="coerce").dropna().astype(int).unique().tolist())
        except Exception:
            start_secs_set = set(); end_secs_set = set()

        # Attribute only to players actually on the ice at that second from pbp_onice
        # Also collect per-(player,sec) routed own_* increments to ensure alignment with GF/SF/AF/AA
        own_routed_xg: Dict[Tuple[int,int], float] = {}
        own_routed_gf: Dict[Tuple[int,int], int] = {}
        own_routed_sf: Dict[Tuple[int,int], int] = {}
        own_routed_af: Dict[Tuple[int,int], int] = {}
        routed_goals: List[Tuple[int,int,int]] = []  # (scorerId, routed_sec, primary_sec)
        for _j, r in ev.iterrows():
            s_primary = int(r["sec_game"]) if not pd.isna(r["sec_game"]) else 0
            tid = int(r["teamId"]) if not pd.isna(r["teamId"]) else 0
            shooter_pid = int(r["shooterId"]) if not pd.isna(r["shooterId"]) else 0
            xg = float(r["xg_attempt"]) if not pd.isna(r["xg_attempt"]) else 0.0
            if xg <= 0:
                continue
            is_goal_evt = False
            try:
                is_goal_evt = bool(int(r.get("is_goal", 0)))
            except Exception:
                is_goal_evt = False

            credited = False
            # Candidate routing seconds:
            cands: List[int] = []
            if is_goal_evt:
                # Build candidates with strict rule: if shift at s-1, first try prev(s-1), then s-1; else try s-1, then prev(s)
                seen: set = set()
                shift_map = prev_sec_map.get("__shift_change_secs__", {}) if isinstance(prev_sec_map, dict) else {}
                has_shift_s1 = False
                try:
                    has_shift_s1 = bool(shift_map.get(s_primary - 1, 0)) if s_primary > 0 else False
                except Exception:
                    has_shift_s1 = False
                # 1) prev(s-1) if shift at s-1
                if has_shift_s1 and prev_sec_map is not None:
                    ps1 = prev_sec_map.get(s_primary - 1)
                    if isinstance(ps1, int) and ps1 >= 0:
                        cands.append(ps1); seen.add(ps1)
                # 2) s-1
                if s_primary > 0 and (s_primary - 1) not in seen:
                    cands.append(s_primary - 1); seen.add(s_primary - 1)
                # 3) prev(s)
                if prev_sec_map is not None:
                    ps = prev_sec_map.get(s_primary)
                    if isinstance(ps, int) and ps >= 0 and ps not in seen:
                        cands.append(ps); seen.add(ps)
                # 4) s last
                if s_primary not in seen:
                    cands.append(s_primary)
            else:
                # non-goal: try s then s-1
                cands = [s_primary, (s_primary - 1) if s_primary > 0 else -1]

            # HARD RULE: at a window boundary (sec is both an end and the next start),
            # force a prior second to the front if present
            if (s_primary in start_secs_set) and (s_primary in end_secs_set):
                prior = None
                # If a shift occurred at s-1, prefer the previous DISTINCT second before s-1
                chose_prev_before_s1 = False
                if prev_sec_map is not None:
                    try:
                        if s_primary > 0 and prev_sec_map.get("__shift_change_secs__", {}).get(s_primary - 1, 0):
                            ps1 = prev_sec_map.get(s_primary - 1, -1)
                            if isinstance(ps1, int) and ps1 >= 0:
                                prior = ps1
                                chose_prev_before_s1 = True
                    except Exception:
                        pass
                if not chose_prev_before_s1:
                    if s_primary > 0 and onice_idx and ((s_primary - 1) in onice_idx):
                        prior = s_primary - 1
                    elif prev_sec_map is not None:
                        try:
                            prior = int(prev_sec_map.get(s_primary, -1))
                        except Exception:
                            prior = -1
                if isinstance(prior, int) and prior >= 0:
                    cands = [prior] + [x for x in cands if x != prior]

            # Iterate candidates in order; assign at first second that yields any hits (matches prior behavior)
            if debug:
                print({"debug":"xg_candidates","sec_primary": s_primary, "cands": cands})
            for s in cands:
                if s < 0:
                    continue
                on_for = on_against = None
                if onice_idx and s in onice_idx:
                    s_idx = onice_idx[s]
                    # s-1 with a shift: use pre-change snapshot
                    if is_goal_evt and prev_sec_map is not None and (s == (s_primary - 1)):
                        try:
                            had_shift = bool(prev_sec_map.get("__shift_change_secs__", {}).get(s, 0))
                        except Exception:
                            had_shift = False
                        if had_shift and isinstance(prev_sec_map.get("__onice_pre__"), dict) and s in prev_sec_map.get("__onice_pre__"):
                            pre_snapshot = prev_sec_map["__onice_pre__"][s]
                            s_idx = {"home": set(pre_snapshot.get("home", set())), "away": set(pre_snapshot.get("away", set()))}
                    # shift earlier than goal in same second → pre snapshot
                    if is_goal_evt and prev_sec_map is not None:
                        try:
                            goal_so = float(prev_sec_map.get("__goal_so_min__", {}).get(s, 1e18))
                            shift_so = float(prev_sec_map.get("__shift_so_min__", {}).get(s, -1e18))
                            if shift_so < goal_so and isinstance(prev_sec_map.get("__onice_pre__"), dict) and s in prev_sec_map.get("__onice_pre__"):
                                pre_snapshot = prev_sec_map["__onice_pre__"][s]
                                s_idx = {"home": set(pre_snapshot.get("home", set())), "away": set(pre_snapshot.get("away", set()))}
                        except Exception:
                            pass
                    if shooter_pid and shooter_pid in s_idx.get("home", set()):
                        on_for = set(s_idx.get("home", set())); on_against = set(s_idx.get("away", set()))
                    elif shooter_pid and shooter_pid in s_idx.get("away", set()):
                        on_for = set(s_idx.get("away", set())); on_against = set(s_idx.get("home", set()))
                    # Fallback by teamId majority
                    if (on_for is None or on_against is None) and tid != 0:
                        home_set = s_idx.get("home", set()); away_set = s_idx.get("away", set())
                        def _count_side(side_set: set) -> int:
                            cnt = 0
                            for pid in side_set:
                                for tup in rows_by_player.get(pid, []):
                                    _i, s0, s1 = tup
                                    if s0 <= s < s1 and int(windows_df.loc[_i, "teamId"]) == tid:
                                        cnt += 1; break
                            return cnt
                        c_home = _count_side(home_set); c_away = _count_side(away_set)
                        if c_home > c_away: on_for = set(home_set); on_against = set(away_set)
                        elif c_away > c_home: on_for = set(away_set); on_against = set(home_set)
                if on_for is None or on_against is None:
                    continue
                # Ensure scorer/assists are included on for-side if they have active rows
                if is_goal_evt:
                    part_ids: List[int] = []
                    if shooter_pid: part_ids.append(int(shooter_pid))
                    for a in ("assist1PlayerId", "assist2PlayerId"):
                        try:
                            aid = int(r.get(a, 0));
                            if aid: part_ids.append(aid)
                        except Exception:
                            pass
                    for need in part_ids:
                        if need in on_for: continue
                        for _t in rows_by_player.get(need, []):
                            _i2, _s0, _s1 = _t
                            if _s0 <= s < _s1:
                                on_for.add(need); break
                # Assign at this candidate second
                any_hit = False
                credited_rows: List[int] = []
                for pid in on_for:
                    for tup in rows_by_player.get(pid, []):
                        i, s0, s1 = tup
                        if s0 <= s < s1:
                            xgf_arr[i] += xg
                            any_hit = True
                            credited_rows.append(i)
                            # own_* routed alignment (shooter contributes only on for-side)
                            # GF credit verification: only the actual scorer gets own_gf
                            scorer_pid = 0
                            # Prefer scorer from pbp goal details at the goal second
                            try:
                                gmap = prev_sec_map.get("__goal_scorer_by_sec__", {}) if isinstance(prev_sec_map, dict) else {}
                                cand_s = s_primary  # base on event primary second
                                if isinstance(gmap, dict) and isinstance(gmap.get(cand_s), list) and gmap.get(cand_s):
                                    scorer_pid = int(gmap[cand_s][0])
                            except Exception:
                                pass
                            if not scorer_pid:
                                try:
                                    scorer_pid = int(ev.at[_j, "scorerId"]) if "scorerId" in ev.columns else shooter_pid
                                except Exception:
                                    scorer_pid = shooter_pid
                            if pid == shooter_pid or pid == scorer_pid:
                                k = (pid, s)
                                # Flags
                                try:
                                    is_sog_flag = int(ev.at[_j, "is_sog"]) == 1
                                except Exception:
                                    is_sog_flag = False
                                try:
                                    is_attempt_flag = int(ev.at[_j, "is_attempt"]) == 1
                                except Exception:
                                    is_attempt_flag = is_sog_flag
                                # Gate by window ground-truth tallies
                                gf_ok = bool(windows_df.loc[i, "GF"]) if "GF" in windows_df.columns else True
                                sf_ok = bool(windows_df.loc[i, "SF"]) if "SF" in windows_df.columns else True
                                af_ok = bool(windows_df.loc[i, "AF"]) if "AF" in windows_df.columns else True
                                # own_xg uses xG from all attempts; gate by AF
                                if is_attempt_flag and af_ok:
                                    own_routed_xg[k] = own_routed_xg.get(k, 0.0) + float(xg)
                                # Only scorer gets own_gf
                                if is_goal_evt and gf_ok and (pid == scorer_pid):
                                    own_routed_gf[k] = own_routed_gf.get(k, 0) + 1
                                if is_sog_flag and sf_ok:
                                    own_routed_sf[k] = own_routed_sf.get(k, 0) + 1
                                if is_attempt_flag and af_ok:
                                    own_routed_af[k] = own_routed_af.get(k, 0) + 1
                for pid in on_against:
                    for tup in rows_by_player.get(pid, []):
                        i, s0, s1 = tup
                        if s0 <= s < s1:
                            xga_arr[i] += xg
                            any_hit = True
                            credited_rows.append(i)
                if debug:
                    print({
                        "debug":"xg_assign",
                        "sec": s,
                        "sec_primary": s_primary,
                        "xg": xg,
                        "shooterId": shooter_pid,
                        "on_for_ct": len(on_for),
                        "on_against_ct": len(on_against),
                        "hit": any_hit,
                        "credited_rows": credited_rows[:6],
                    })
                if any_hit:
                    credited = True
                    if is_goal_evt:
                        # remember which player actually scored for this routed second
                        try:
                            gmap = prev_sec_map.get("__goal_scorer_by_sec__", {}) if isinstance(prev_sec_map, dict) else {}
                            scorer_list = gmap.get(s_primary) if isinstance(gmap, dict) else None
                            if isinstance(scorer_list, list) and scorer_list:
                                routed_goals.append((int(scorer_list[0]), int(s), int(s_primary)))
                            else:
                                routed_goals.append((int(shooter_pid), int(s), int(s_primary)))
                        except Exception:
                            routed_goals.append((int(shooter_pid), int(s), int(s_primary)))
                    break
            if debug and (not credited):
                print({
                    "debug":"xg_assign_none",
                    "sec_primary": s_primary,
                    "xg": xg,
                    "shooterId": shooter_pid,
                    "reason": "no_rows_matched_at_s_or_sminus1",
                })

        # Write into columns, creating if needed
        if "xGF" not in windows_df.columns:
            windows_df["xGF"] = 0.0
        if "xGA" not in windows_df.columns:
            windows_df["xGA"] = 0.0
        windows_df["xGF"] = pd.to_numeric(windows_df["xGF"], errors="coerce").fillna(0.0) + xgf_arr
        windows_df["xGA"] = pd.to_numeric(windows_df["xGA"], errors="coerce").fillna(0.0) + xga_arr

        # Override own_* windows using routed seconds so they align to GF/SF/AF ground truth
        if own_routed_xg or own_routed_gf or own_routed_sf or own_routed_af:
            windows_df["own_xg_window"] = 0.0
            windows_df["own_gf_window"] = 0
            windows_df["own_sf_window"] = 0
            windows_df["own_af_window"] = 0
            for (pid, s), vxg in own_routed_xg.items():
                for tup in rows_by_player.get(pid, []):
                    i, s0, s1 = tup
                    if s0 <= s < s1:
                        windows_df.at[i, "own_xg_window"] += float(vxg)
            # Enforce: at most one individual GF per window. Use PBP scorer and the routed second to select row
            if routed_goals:
                for (sid, s, _sp) in routed_goals:
                    # find scorer's row for this window
                    sc_rows: List[int] = []
                    win_key = None
                    for (i, s0, s1) in rows_by_player.get(int(sid), []):
                        if s0 <= s < s1:
                            sc_rows.append(i)
                            # build window identity
                            team_val = windows_df.loc[i, "teamId"] if "teamId" in windows_df.columns else None
                            gpk_val = windows_df.loc[i, "gamePk"] if "gamePk" in windows_df.columns else None
                            win_key = (int(s0), int(s1), int(team_val) if team_val is not None else -1, int(gpk_val) if gpk_val is not None else -1)
                            break
                    if not sc_rows or win_key is None:
                        continue
                    i_sc = sc_rows[0]
                    s0, s1, team_v, gpk_v = win_key
                    # zero everyone in this window, then set scorer to 1
                    mask_win = (
                        (windows_df.get("start_sec") == s0) &
                        (windows_df.get("end_sec") == s1)
                    )
                    if "teamId" in windows_df.columns:
                        mask_win = mask_win & (windows_df["teamId"] == team_v)
                    if "gamePk" in windows_df.columns:
                        mask_win = mask_win & (pd.to_numeric(windows_df["gamePk"], errors="coerce") == gpk_v)
                    # Only operate within rows that actually have GF>0 (ground truth)
                    if "GF" in windows_df.columns:
                        gf_mask = mask_win & (pd.to_numeric(windows_df["GF"], errors="coerce").fillna(0).astype(int) > 0)
                    else:
                        gf_mask = mask_win
                    windows_df.loc[gf_mask, "own_gf_window"] = 0
                    # Set scorer only if his row has GF>0
                    if ("GF" not in windows_df.columns) or (pd.to_numeric(windows_df.loc[i_sc, "GF"], errors="coerce") > 0):
                        windows_df.at[i_sc, "own_gf_window"] = 1
            for (pid, s), v in own_routed_sf.items():
                for tup in rows_by_player.get(pid, []):
                    i, s0, s1 = tup
                    if s0 <= s < s1:
                        windows_df.at[i, "own_sf_window"] += int(v)
            for (pid, s), v in own_routed_af.items():
                for tup in rows_by_player.get(pid, []):
                    i, s0, s1 = tup
                    if s0 <= s < s1:
                        windows_df.at[i, "own_af_window"] += int(v)
            # Fill primary/secondary assists directly from PBP details at the goal's primary second, mapped to routed window
            try:
                ga_map = prev_sec_map.get("__goal_assists_by_sec__", {}) if isinstance(prev_sec_map, dict) else {}
                if isinstance(ga_map, dict):
                    if "own_primary_assists_window" not in windows_df.columns:
                        windows_df["own_primary_assists_window"] = 0
                    if "own_secondary_assists_window" not in windows_df.columns:
                        windows_df["own_secondary_assists_window"] = 0
                    # Reset to 0 so we only count from routed PBP assists
                    windows_df["own_primary_assists_window"] = pd.to_numeric(windows_df["own_primary_assists_window"], errors="coerce").fillna(0).astype(int)
                    windows_df["own_secondary_assists_window"] = pd.to_numeric(windows_df["own_secondary_assists_window"], errors="coerce").fillna(0).astype(int)
                    for (sid, s, sp) in routed_goals:
                        a1, a2 = (0, 0)
                        try:
                            pair = ga_map.get(int(sp))
                            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                                a1 = int(pair[0] or 0)
                                a2 = int(pair[1] or 0)
                        except Exception:
                            a1, a2 = (0, 0)
                        for aid, col in [(a1, "own_primary_assists_window"), (a2, "own_secondary_assists_window")]:
                            if aid:
                                for tup in rows_by_player.get(int(aid), []):
                                    i, s0, s1 = tup
                                    if s0 <= s < s1:
                                        windows_df.at[i, col] += 1
            except Exception:
                pass

    return windows_df


# -------------------------------
# CLI
# -------------------------------

def _infer_game_id_from_path(path: str) -> Optional[str]:
    m = re.search(r"(\d{10})", os.path.basename(path))
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill player window own-xG and assist metrics")
    ap.add_argument("--windows", required=True, help="Path to player_windows_train_*.csv or player_rollup_*.csv")
    ap.add_argument("--shots", required=True, help="Path to shots_train_*_scored.csv")
    ap.add_argument("--onice", required=True, help="Path to pbp_onice_*.json (not used for own stats, kept for compatibility)")
    ap.add_argument("--ages_json", required=False, default=None, help="Optional ages json")
    ap.add_argument("--out", required=True, help="Output csv path")
    ap.add_argument("--debug", action="store_true", help="Debug logging")
    args = ap.parse_args()

    windows_path = os.path.abspath(args.windows)
    shots_path = os.path.abspath(args.shots)
    onice_path = os.path.abspath(args.onice)

    # Read inputs
    df_win = _read_windows_csv(windows_path)
    df_shots = _read_shots_csv(shots_path)
    ages = _read_ages_json(args.ages_json)
    bdates, ages_asof = _read_birthdates_json(args.ages_json)

    # Warn if windows has no game IDs (common for rollups)
    if "gamePk" not in df_win.columns or df_win["gamePk"].isna().all():
        print("[warn] windows has 0 game IDs; proceeding, but shots/on-ice should match.")

    # Build contributions (we only use own_contrib)
    contrib, own_contrib = _build_contributions(df_shots, None, debug=args.debug, onice_json_path=onice_path)
    onice_idx, prev_sec_map = _load_onice_index_from_pbp(onice_path)

    # Aggregate into windows
    try:
        df_out = _aggregate_own_into_windows(df_win, own_contrib, contrib, onice_idx, prev_sec_map, debug=args.debug)
    except KeyError as e:
        # Provide clearer guidance when windows are not suitable (e.g., team_rollup without start/end)
        raise SystemExit(f"windows file missing required columns for windowing: {e}. Use player_windows_train_* or player_rollup_*.")

    # Optionally attach ages/birthdates (if ages_json provided)
    if "playerId" in df_out.columns:
        pid_str = df_out["playerId"].astype(int).astype(str)
        # birthDate
        if bdates:
            # fillna requires a scalar or Series; ensure Series fallback exists
            existing_bd = df_out["birthDate"] if "birthDate" in df_out.columns else pd.Series([None]*len(df_out))
            df_out["birthDate"] = pid_str.map(bdates)
            df_out["birthDate"] = df_out["birthDate"].where(df_out["birthDate"].notna(), existing_bd)
        # age_on_asof if available in JSON
        if ages_asof:
            df_out["age_on_asof"] = pd.to_numeric(pid_str.map(ages_asof), errors="coerce")
        # age on game date if we have date
        if "date" in df_out.columns:
            try:
                df_out["age_on_game"] = [
                    _compute_age_years(bdates.get(str(int(pid)), None), dt)
                    for pid, dt in zip(df_out["playerId"], df_out["date"])
                ] if bdates else df_out.get("age_on_game")
            except Exception:
                pass
        # attach age (flat mapping) if provided
        if ages:
            df_out["age"] = pd.to_numeric(pid_str.map(ages), errors="coerce").fillna(df_out.get("age", np.nan))
        # Derived centered age columns if we got age on game
        if "age_on_game" in df_out.columns:
            mu = 27.0
            try:
                df_out["age_c"] = pd.to_numeric(df_out["age_on_game"], errors="coerce") - mu
                df_out["age_c_sq"] = df_out["age_c"] * df_out["age_c"]
            except Exception:
                pass

    # Derive ai_OZ_start / ai_DZ_start from after_icing and zone_start (fallback fo_zone)
    try:
        if "after_icing" in df_out.columns:
            if "ai_OZ_start" not in df_out.columns:
                df_out["ai_OZ_start"] = 0
            if "ai_DZ_start" not in df_out.columns:
                df_out["ai_DZ_start"] = 0
            ai_mask = pd.to_numeric(df_out["after_icing"], errors="coerce").fillna(0).astype(int) == 1
            # pick zone source
            if "zone_start" in df_out.columns:
                zsrc = df_out["zone_start"].astype(str).str.upper()
            elif "fo_zone" in df_out.columns:
                zsrc = df_out["fo_zone"].astype(str).str.upper()
            else:
                zsrc = pd.Series([""] * len(df_out))
            oz = ai_mask & (zsrc.str.startswith("O") | zsrc.str.startswith("OFF"))
            dz = ai_mask & (zsrc.str.startswith("D") | zsrc.str.startswith("DEF"))
            df_out.loc[:, "ai_OZ_start"] = 0
            df_out.loc[:, "ai_DZ_start"] = 0
            df_out.loc[oz, "ai_OZ_start"] = 1
            df_out.loc[dz, "ai_DZ_start"] = 1
            # Ensure ints
            df_out["ai_OZ_start"] = pd.to_numeric(df_out["ai_OZ_start"], errors="coerce").fillna(0).astype(int)
            df_out["ai_DZ_start"] = pd.to_numeric(df_out["ai_DZ_start"], errors="coerce").fillna(0).astype(int)
    except Exception:
        pass

    # Derive long_change by period: 1 for even-numbered periods (2,4,...) else 0
    try:
        per = None
        if "period" in df_out.columns:
            per = pd.to_numeric(df_out["period"], errors="coerce")
        elif "periodNumber" in df_out.columns:
            per = pd.to_numeric(df_out["periodNumber"], errors="coerce")
        elif "start_sec" in df_out.columns:
            per = (pd.to_numeric(df_out["start_sec"], errors="coerce").fillna(0) // 1200) + 1
        if per is not None:
            per = per.fillna(0).astype(int)
            df_out["long_change"] = (per % 2 == 0).astype(int)
            # Also populate long_change_start to match naming expected by some rollups
            df_out["long_change_start"] = (per % 2 == 0).astype(int)
    except Exception:
        pass

    # Write output
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Drop only the specifically requested columns if present
    try:
        for col in ["early_mass", "long_change_start", "fo_O_start", "fo_D_start", "last_change", "last_change_start"]:
            if col in df_out.columns:
                df_out.drop(columns=[col], inplace=True)
    except Exception:
        pass
    # Sort: home players first, then away players, within each window_id
    if "window_id" in df_out.columns and "team_side" in df_out.columns:
        def sort_key(row):
            window_id = row.get("window_id", 0) if isinstance(row, dict) else (row["window_id"] if "window_id" in row.index else 0)
            team_side = row.get("team_side", "") if isinstance(row, dict) else (row["team_side"] if "team_side" in row.index else "")
            side_order = 0 if team_side == "home" else 1
            return (window_id, side_order)
        # Use pandas sorting
        if "window_id" in df_out.columns and "team_side" in df_out.columns:
            df_out["_sort_side"] = df_out["team_side"].map(lambda x: 0 if x == "home" else 1)
            df_out = df_out.sort_values(by=["window_id", "_sort_side"], kind="stable")
            df_out = df_out.drop(columns=["_sort_side"])
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df_out)} rows)")


if __name__ == "__main__":
    main()


