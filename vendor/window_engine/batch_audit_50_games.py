import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import perfect_windows as pw


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def strength_tuple(team_onice_by_sec, goalie_ids_by_sec, sec: int) -> Tuple[int, int, int, int]:
    hh, aa = team_onice_by_sec[sec]
    gh = _safe_int(goalie_ids_by_sec[sec].get("home", 0))
    ga = _safe_int(goalie_ids_by_sec[sec].get("away", 0))
    return (len(hh), len(aa), 1 if gh != 0 else 0, 1 if ga != 0 else 0)


def has_weird_total_players(team_onice_by_sec, goalie_ids_by_sec, horizon: int) -> int:
    # with the goalie-vs-6-skaters guard, this should be 0
    weird = 0
    for s in range(0, horizon + 1):
        hs, as_, ghp, gap = strength_tuple(team_onice_by_sec, goalie_ids_by_sec, s)
        if hs + ghp > 6 or as_ + gap > 6:
            weird += 1
    return weird


def window_quality(game: pw.ParsedGame, windows: List[Dict[str, Any]], horizon: int, fo_meta: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
    bad_bounds = 0
    faceoff_mismatch = 0
    missing_reason = 0
    dup_ids = len(windows) - len({str(w.get("window_id") or "") for w in windows})
    le1 = le2 = le3 = eq0 = 0
    for w in windows:
        s = _safe_int(w.get("start_sec"), 0)
        e = _safe_int(w.get("end_sec"), 0)
        d = _safe_int(w.get("duration"), e - s)
        if not (0 <= s <= e <= horizon):
            bad_bounds += 1
        if d == 0:
            eq0 += 1
        if d <= 1:
            le1 += 1
        if d <= 2:
            le2 += 1
        if d <= 3:
            le3 += 1
        reason = str(w.get("end_event_type") or "")
        if reason == "":
            missing_reason += 1
        if reason.lower() == "faceoff" and e not in fo_meta:
            faceoff_mismatch += 1
    return {
        "windows_total": len(windows),
        "windows_eq0": eq0,
        "windows_le1": le1,
        "windows_le2": le2,
        "windows_le3": le3,
        "windows_bad_bounds": bad_bounds,
        "windows_faceoff_end_mismatch": faceoff_mismatch,
        "windows_missing_end_reason": missing_reason,
        "windows_duplicate_ids": dup_ids,
    }


def event_counts(events_by_sec: Dict[int, List[Dict[str, Any]]]) -> Dict[str, int]:
    c = Counter()
    for evs in events_by_sec.values():
        for ev in evs:
            c[str(ev.get("type") or "").lower()] += 1
    return {
        "events_goal": int(c.get("goal", 0)),
        "events_shot_on_goal": int(c.get("shot-on-goal", 0)),
        "events_missed_shot": int(c.get("missed-shot", 0)),
        "events_blocked_shot": int(c.get("blocked-shot", 0)),
        "events_penalty": int(c.get("penalty", 0)),
        "events_faceoff": int(c.get("faceoff", 0)),
        "events_shift_change": int(c.get("shift_change", 0)) + int(c.get("shift-change", 0)),
        "events_delayed_penalty": int(c.get("delayed-penalty", 0)),
    }


def validate_token_outcomes(
    game: pw.ParsedGame,
    windows: List[Dict[str, Any]],
    team_onice_by_sec,
    goalie_ids_by_sec,
    events_by_sec,
    orders_by_sec,
    shift_changes_by_sec,
    credit_by_sec,
    token_rows: List[Dict[str, Any]],
    max_examples: int = 20,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    win_by_id = {str(w["window_id"]): w for w in windows}
    by_sec_home = {s: set(team_onice_by_sec[s][0]) for s in range(len(team_onice_by_sec))}
    by_sec_away = {s: set(team_onice_by_sec[s][1]) for s in range(len(team_onice_by_sec))}

    cols_int = ["GF", "GA", "SF", "SA", "AF", "AA", "BF", "BA"]
    mismatch_counts = Counter()
    examples: List[Dict[str, Any]] = []

    for r in token_rows:
        win_id = str(r.get("window_id") or "")
        w = win_by_id.get(win_id)
        if w is None:
            mismatch_counts["missing_window"] += 1
            if len(examples) < max_examples:
                examples.append({"reason": "missing_window", "window_id": win_id})
            continue
        p = int(r.get("playerId"))
        side = str(r.get("team_side"))
        t_tok = int(r.get("t_token"))
        t_end = int(r.get("t_end"))
        rec = pw.aggregate_features_for_token(
            game=game,
            by_sec_home=by_sec_home,
            by_sec_away=by_sec_away,
            events_by_sec=events_by_sec,
            credit_by_sec=credit_by_sec,
            team_onice_by_sec=team_onice_by_sec,
            orders_by_sec=orders_by_sec,
            shift_changes_by_sec=shift_changes_by_sec,
            side=side,
            p=p,
            win_start=int(w.get("start_sec")),
            win_end=int(w.get("end_sec")),
            t_token=t_tok,
            t_end=t_end,
        )
        bad = []
        for c in cols_int:
            got = _safe_int(r.get(c), 0)
            exp = _safe_int(rec.get(c), 0)
            if got != exp:
                mismatch_counts[f"col_{c}"] += 1
                bad.append((c, got, exp))
        if bad:
            mismatch_counts["rows_with_mismatch"] += 1
            if len(examples) < max_examples:
                examples.append(
                    {
                        "window_id": win_id,
                        "team_side": side,
                        "playerId": p,
                        "token_type": r.get("token_type"),
                        "t_token": t_tok,
                        "t_end": t_end,
                        "bad": bad,
                    }
                )

    return dict(mismatch_counts), examples


@dataclass
class GameAuditSummary:
    gamePk: int
    file: str
    horizon_sec: int
    ok: int
    # windows
    windows_total: int
    windows_eq0: int
    windows_le1: int
    windows_le2: int
    windows_le3: int
    windows_bad_bounds: int
    windows_faceoff_end_mismatch: int
    windows_missing_end_reason: int
    windows_duplicate_ids: int
    # strength/weird
    weird_total_players_gt6: int
    # events
    events_goal: int
    events_shot_on_goal: int
    events_missed_shot: int
    events_blocked_shot: int
    events_penalty: int
    events_faceoff: int
    events_shift_change: int
    events_delayed_penalty: int
    # token outcomes
    token_rows: int
    token_outcome_mismatch_rows: int


def main():
    ap = argparse.ArgumentParser(description="Batch audit N games for window+token outcome correctness.")
    ap.add_argument("--pbp_dir", default="API/Final/2025/raw/pbpice", help="Directory with pbp_onice_*.json")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out_dir", default="artifacts/audits/batch50")
    ap.add_argument("--max_issue_examples", type=int, default=30)
    args = ap.parse_args()

    pbp_files = sorted(glob.glob(os.path.join(args.pbp_dir, "pbp_onice_*.json")))[: int(args.n)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[GameAuditSummary] = []
    issues: Dict[int, Dict[str, Any]] = {}

    for fp in pbp_files:
        pbp_onice = pw.load_json(fp)
        game = pw.parse_game(pbp_onice)
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
        ) = pw.build_second_index(game)
        credit_by_sec, cum_home_goals, cum_away_goals = pw.build_credit_by_sec(
            game, team_onice_by_sec, goalie_ids_by_sec, events_by_sec, orders_by_sec, shift_changes_by_sec
        )
        windows = pw.build_windows(
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
            hard_cap_sec=0,
        )

        # metadata for names/positions/rink isn't needed for correctness checks
        token_rows = pw.generate_tokens_and_rows(
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
            horizon_sec=20,
            min_swap=2,
            min_gap_sec=4,
            stable_sec=2,
            rc_require_stable=1,
            post_dwell_sec=2,
            mass_swap_suppress_threshold=3,
            max_tokens=10,
            mode="both",
            player_name_map={},
            player_pos_map={},
            season="",
            date="",
            rinkid="",
        )

        wq = window_quality(game, windows, horizon, fo_meta)
        ec = event_counts(events_by_sec)
        weird = has_weird_total_players(team_onice_by_sec, goalie_ids_by_sec, horizon)
        mism, ex = validate_token_outcomes(
            game,
            windows,
            team_onice_by_sec,
            goalie_ids_by_sec,
            events_by_sec,
            orders_by_sec,
            shift_changes_by_sec,
            credit_by_sec,
            token_rows,
            max_examples=int(args.max_issue_examples),
        )

        ok = int(
            wq["windows_bad_bounds"] == 0
            and wq["windows_faceoff_end_mismatch"] == 0
            and wq["windows_missing_end_reason"] == 0
            and wq["windows_duplicate_ids"] == 0
            and wq["windows_eq0"] == 0
            and weird == 0
            and int(mism.get("rows_with_mismatch", 0)) == 0
        )

        summ = GameAuditSummary(
            gamePk=int(game.gamePk),
            file=str(fp),
            horizon_sec=int(horizon),
            ok=int(ok),
            weird_total_players_gt6=int(weird),
            token_rows=int(len(token_rows)),
            token_outcome_mismatch_rows=int(mism.get("rows_with_mismatch", 0)),
            **wq,
            **ec,
        )
        summaries.append(summ)

        if not ok:
            issues[int(game.gamePk)] = {
                "summary": asdict(summ),
                "token_outcome_mismatch_counts": mism,
                "token_outcome_mismatch_examples": ex,
            }

    # write summary csv/json
    summ_rows = [asdict(s) for s in summaries]
    (out_dir / "batch_summary.json").write_text(json.dumps(summ_rows, indent=2), encoding="utf-8")

    with (out_dir / "batch_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(GameAuditSummary.__annotations__.keys()))
        w.writeheader()
        for r in summ_rows:
            w.writerow(r)

    (out_dir / "issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")

    ok_n = sum(1 for s in summaries if s.ok == 1)
    print(f"audited={len(summaries)} ok={ok_n} bad={len(summaries)-ok_n} out_dir={out_dir}")


if __name__ == "__main__":
    main()

