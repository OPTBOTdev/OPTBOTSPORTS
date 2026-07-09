import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Reuse parsing/indexing + correction logic from perfect_windows
import perfect_windows as pw


GOAL_TYPES = {"goal"}


@dataclass
class GoalAuditRow:
    gamePk: int
    sec_game: int
    period: int
    clock: str
    sortOrder: int
    same_sec_order: int
    eventOwnerTeamId: Optional[int]
    side: Optional[str]
    scoringPlayerId: Optional[int]
    assist1PlayerId: Optional[int]
    assist2PlayerId: Optional[int]
    # pre snapshots
    pre_home_has_scorer: int
    pre_away_has_scorer: int
    pre_home_n: int
    pre_away_n: int
    # corrected snapshots
    corr_home_has_scorer: int
    corr_away_has_scorer: int
    corr_home_n: int
    corr_away_n: int
    # flags
    shift_change_same_sec_before: int
    shift_change_prev_sec: int
    period_boundary_sec: int
    corrected_changed_sets: int
    still_missing_scorer_after_corr: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def iter_pbp_onice_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in root.rglob("pbp_onice_*.json"):
        yield p


def parse_gamePk_from_path(p: Path) -> Optional[int]:
    # pbp_onice_2025020001.json
    name = p.name
    try:
        s = name.split("pbp_onice_")[1].split(".json")[0]
        return int(s)
    except Exception:
        return None


def audit_one_file(path: Path) -> Tuple[Dict[str, Any], List[GoalAuditRow]]:
    pbp_onice = pw.load_json(str(path))
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

    # Count raw goal events
    raw_goals = []
    for sec, evs in events_by_sec.items():
        for ev in evs:
            if str(ev.get("type") or "").lower() in GOAL_TYPES:
                raw_goals.append(ev)

    # Count credited goals from our per-second credit
    credited_home = sum(int((credit_by_sec.get(s) or {}).get("home", {}).get("GF", 0)) for s in range(horizon + 1))
    credited_away = sum(int((credit_by_sec.get(s) or {}).get("away", {}).get("GF", 0)) for s in range(horizon + 1))

    summary = {
        "file": str(path),
        "gamePk": int(game.gamePk),
        "raw_goal_events": int(len(raw_goals)),
        "credited_GF_home": int(credited_home),
        "credited_GF_away": int(credited_away),
        "credited_GF_total": int(credited_home + credited_away),
        "horizon_sec": int(horizon),
    }

    rows: List[GoalAuditRow] = []

    for ev in raw_goals:
        det = ev.get("details") or {}
        sec = int(ev.get("sec_game") or 0)
        sortOrder = int(float(ev.get("sortOrder", 0) or 0))
        sso = int(float(ev.get("same_sec_order", 0) or 0))
        owner = det.get("eventOwnerTeamId")
        try:
            owner_i = int(owner) if owner is not None else None
        except Exception:
            owner_i = None
        side = pw._owner_side_for_event(det, game.home_team_id, game.away_team_id)

        scorer = det.get("scoringPlayerId") or det.get("shootingPlayerId")
        a1 = det.get("assist1PlayerId")
        a2 = det.get("assist2PlayerId")
        try:
            scorer_i = int(scorer) if scorer is not None else None
        except Exception:
            scorer_i = None
        try:
            a1_i = int(a1) if a1 is not None else None
        except Exception:
            a1_i = None
        try:
            a2_i = int(a2) if a2 is not None else None
        except Exception:
            a2_i = None

        cb = credit_by_sec.get(sec) or {}
        pre_home = set(cb.get("onice_home", []) or [])
        pre_away = set(cb.get("onice_away", []) or [])

        # flags: shift change same second before goal
        evs_this = orders_by_sec.get(sec, [])
        same_sec_shift_before = any((t == "shift-change" and int(so) < int(sortOrder)) for so, t in evs_this)
        prev_sec_shift = bool(shift_changes_by_sec.get(sec - 1)) if sec - 1 >= 0 else False
        period_boundary = int(sec in (1200, 2400, 3600, 4800))

        corr_home, corr_away = pw._maybe_correct_onice_for_goal(
            sec,
            ev,
            set(pre_home),
            set(pre_away),
            game.home_team_id,
            game.away_team_id,
            team_onice_by_sec,
            orders_by_sec,
            shift_changes_by_sec,
        )

        changed = int((set(pre_home) != set(corr_home)) or (set(pre_away) != set(corr_away)))
        still_missing = 0
        if scorer_i is not None:
            if side == "home":
                still_missing = int(scorer_i not in set(corr_home))
            elif side == "away":
                still_missing = int(scorer_i not in set(corr_away))
            else:
                still_missing = 0

        rows.append(
            GoalAuditRow(
                gamePk=int(game.gamePk),
                sec_game=int(sec),
                period=int(pw.period_of(sec)),
                clock=str(pw.clock_str(sec)),
                sortOrder=int(sortOrder),
                same_sec_order=int(sso),
                eventOwnerTeamId=owner_i,
                side=side,
                scoringPlayerId=scorer_i,
                assist1PlayerId=a1_i,
                assist2PlayerId=a2_i,
                pre_home_has_scorer=int(1 if (scorer_i is not None and scorer_i in pre_home) else 0),
                pre_away_has_scorer=int(1 if (scorer_i is not None and scorer_i in pre_away) else 0),
                pre_home_n=int(len(pre_home)),
                pre_away_n=int(len(pre_away)),
                corr_home_has_scorer=int(1 if (scorer_i is not None and scorer_i in set(corr_home)) else 0),
                corr_away_has_scorer=int(1 if (scorer_i is not None and scorer_i in set(corr_away)) else 0),
                corr_home_n=int(len(set(corr_home))),
                corr_away_n=int(len(set(corr_away))),
                shift_change_same_sec_before=int(1 if same_sec_shift_before else 0),
                shift_change_prev_sec=int(1 if prev_sec_shift else 0),
                period_boundary_sec=int(period_boundary),
                corrected_changed_sets=int(changed),
                still_missing_scorer_after_corr=int(still_missing),
            )
        )

    return summary, rows


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Audit goal crediting + on-ice correction vs pbp_onice feed")
    ap.add_argument("--in", dest="in_path", required=True, help="pbp_onice_<gamePk>.json OR directory containing them")
    ap.add_argument("--out_dir", default="artifacts/audits", help="Output directory")
    ap.add_argument("--min-hard", type=int, default=50, help="Ensure at least this many hard edge-case goal rows are sampled into hard_cases.csv")
    args = ap.parse_args()

    root = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    all_goal_rows: List[Dict[str, Any]] = []
    hard_rows: List[Dict[str, Any]] = []

    files = list(iter_pbp_onice_files(root))
    if not files:
        raise SystemExit(f"No pbp_onice_*.json files found under {root}")

    for fp in files:
        try:
            summary, rows = audit_one_file(fp)
        except Exception as e:
            summaries.append({"file": str(fp), "error": str(e)})
            continue
        summaries.append(summary)
        for r in rows:
            d = r.to_dict()
            all_goal_rows.append(d)
            is_hard = (
                d["shift_change_same_sec_before"] == 1
                or d["shift_change_prev_sec"] == 1
                or d["period_boundary_sec"] == 1
                or d["pre_home_has_scorer"] == 0 and d["pre_away_has_scorer"] == 0
                or d["corrected_changed_sets"] == 1
                or d["still_missing_scorer_after_corr"] == 1
            )
            if is_hard:
                hard_rows.append(d)

    # Sort hard rows to prioritize most suspicious cases
    def hard_key(d):
        return (
            -int(d.get("still_missing_scorer_after_corr", 0)),
            -int(d.get("corrected_changed_sets", 0)),
            -int(d.get("shift_change_same_sec_before", 0)),
            -int(d.get("shift_change_prev_sec", 0)),
            -int(d.get("period_boundary_sec", 0)),
            int(d.get("gamePk", 0)),
            int(d.get("sec_game", 0)),
            int(d.get("sortOrder", 0)),
        )

    hard_rows_sorted = sorted(hard_rows, key=hard_key)
    # Ensure at least N rows; if not enough hard cases exist, top up with earliest goals
    min_hard = int(args.min_hard)
    if len(hard_rows_sorted) < min_hard:
        extra = [d for d in all_goal_rows if d not in hard_rows_sorted]
        hard_rows_sorted = hard_rows_sorted + extra[: max(0, min_hard - len(hard_rows_sorted))]

    # Write outputs
    with (out_dir / "goal_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    fieldnames = list(GoalAuditRow.__annotations__.keys())
    write_csv_rows(out_dir / "all_goals.csv", all_goal_rows, fieldnames)
    write_csv_rows(out_dir / "hard_cases.csv", hard_rows_sorted[: max(min_hard, 50)], fieldnames)

    print(f"Wrote {out_dir/'goal_audit_summary.json'} ({len(summaries)} files)")
    print(f"Wrote {out_dir/'all_goals.csv'} ({len(all_goal_rows)} goal events)")
    print(f"Wrote {out_dir/'hard_cases.csv'} ({min(len(hard_rows_sorted), max(min_hard, 50))} rows)")


if __name__ == "__main__":
    main()

