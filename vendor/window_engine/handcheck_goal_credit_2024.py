import argparse
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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


def _etype(ev: Dict[str, Any]) -> str:
    return str(ev.get("type") or "").strip().lower()


def _details(ev: Dict[str, Any]) -> Dict[str, Any]:
    return ev.get("details") or {}


def _owner_side(det: Dict[str, Any], home_tid: int, away_tid: int) -> str:
    return pw._owner_side_for_event(det, home_tid, away_tid)


def _goal_label(ev: Dict[str, Any], home_tid: int, away_tid: int) -> str:
    det = _details(ev)
    owner = _owner_side(det, home_tid, away_tid)
    scorer = det.get("scoringPlayerId") or det.get("shootingPlayerId") or det.get("shooterPlayerId")
    as1 = det.get("assist1PlayerId") or det.get("assistOnePlayerId")
    as2 = det.get("assist2PlayerId") or det.get("assistTwoPlayerId")
    return f"goal owner={owner} scorer={scorer} a1={as1} a2={as2}"


def _find_window_for_goal_second(windows: List[Dict[str, Any]], sec: int) -> Optional[Dict[str, Any]]:
    sec_i = int(sec)
    for w in windows:
        if int(w.get("end_sec", -1)) == sec_i and int(w.get("start_sec", 10**9)) < sec_i:
            return w
    for w in windows:
        s = int(w.get("start_sec", 0))
        e = int(w.get("end_sec", 0))
        if s < sec_i <= e:
            return w
    return None


def _per_second_delta(
    *,
    game: pw.ParsedGame,
    team_onice_by_sec,
    events_by_sec,
    credit_by_sec,
    orders_by_sec,
    shift_changes_by_sec,
    side: str,
    player_id: int,
    win_start: int,
    win_end: int,
    sec: int,
) -> Dict[str, int]:
    """
    Compute the per-second (sec only) delta for GF/GA/SF/SA/AF/AA/BF/BA by running
    aggregate_features_for_token over (sec-1, sec].
    """
    t_token = int(sec) - 1
    t_end = int(sec)
    if t_token < int(win_start):
        t_token = int(win_start) - 1
    if t_token < 0:
        t_token = -1
    out = pw.aggregate_features_for_token(
        game=game,
        by_sec_home={s: set(team_onice_by_sec[s][0]) for s in range(len(team_onice_by_sec))},
        by_sec_away={s: set(team_onice_by_sec[s][1]) for s in range(len(team_onice_by_sec))},
        events_by_sec=events_by_sec,
        credit_by_sec=credit_by_sec,
        team_onice_by_sec=team_onice_by_sec,
        orders_by_sec=orders_by_sec,
        shift_changes_by_sec=shift_changes_by_sec,
        side=str(side),
        p=int(player_id),
        win_start=int(win_start),
        win_end=int(win_end),
        t_token=int(t_token),
        t_end=int(t_end),
    )
    return {k: int(out.get(k, 0) or 0) for k in ("GF", "GA", "SF", "SA", "AF", "AA", "BF", "BA")}


@dataclass
class GoalCaseSummary:
    case_id: int
    gamePk: int
    sec: int
    window_id: str
    window_start: int
    window_end: int
    goal_label: str
    owner_side: str
    pre_onice_home: str
    pre_onice_away: str
    corr_onice_home: str
    corr_onice_away: str
    per_second_ok: int
    failures: str
    missing_outcome_rows: int  # diagnostic only (token horizon coverage), NOT a correctness failure


def main() -> None:
    ap = argparse.ArgumentParser(description="Tough goal-credit handcheck: verify per-second GF/GA deltas for all on-ice players (2024).")
    ap.add_argument("--pbp_dir", default="API/Final/2024/raw/pbp_built_20242025")
    ap.add_argument("--out_dir", default="artifacts/audits/handcheck_2024_goals_50cases")
    ap.add_argument("--sample_games", type=int, default=30)
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    # token settings to mirror generator
    ap.add_argument("--horizon_sec", type=int, default=20)
    ap.add_argument("--min_exposure_sec", type=int, default=1)
    ap.add_argument("--state_stable_sec", type=int, default=1)
    ap.add_argument("--state_chunk_by_h", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    pbp_dir = str(args.pbp_dir)
    files = sorted([os.path.join(pbp_dir, f) for f in os.listdir(pbp_dir) if f.startswith("pbp_onice_") and f.endswith(".json")])
    if not files:
        raise SystemExit(f"No pbp_onice_*.json found in {pbp_dir}")

    n_games = min(int(args.sample_games), len(files))
    game_files = sorted(rng.sample(files, n_games))

    out_dir = Path(str(args.out_dir))
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    picked: List[Tuple[str, int, Dict[str, Any]]] = []
    # collect goals from sampled games
    goal_pool: List[Tuple[str, int, Dict[str, Any]]] = []
    for fp in game_files:
        pbp_onice = pw.load_json(fp)
        game = pw.parse_game(pbp_onice)
        _, _, events_by_sec, _, _, _, _, _, _, _ = pw.build_second_index(game)
        for sec, evs in (events_by_sec or {}).items():
            for ev in evs or []:
                if _etype(ev) == "goal":
                    goal_pool.append((fp, int(sec), ev))
    if not goal_pool:
        raise SystemExit("No goals found in sampled games.")

    need = min(int(args.cases), len(goal_pool))
    picked = rng.sample(goal_pool, need)

    rows: List[GoalCaseSummary] = []
    total_missing_outcome = 0

    for idx, (fp, sec, goal_ev) in enumerate(picked, start=1):
        pbp_onice = pw.load_json(fp)
        game = pw.parse_game(pbp_onice)
        gamePk = int(game.gamePk)
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
        win = _find_window_for_goal_second(windows, int(sec))
        win_id = str((win or {}).get("window_id") or "")
        win_s = int((win or {}).get("start_sec") or 0)
        win_e = int((win or {}).get("end_sec") or 0)
        windows_by_id = {str(w["window_id"]): w for w in windows}

        # compute corrected on-ice sets at goal second
        sec_i = int(sec)
        cb = credit_by_sec.get(int(sec_i)) or {}
        pre_home = sorted(int(x) for x in (cb.get("onice_home") or team_onice_by_sec[sec_i][0] or []))
        pre_away = sorted(int(x) for x in (cb.get("onice_away") or team_onice_by_sec[sec_i][1] or []))
        det = _details(goal_ev)
        owner_side = _owner_side(det, game.home_team_id, game.away_team_id)
        corr_home, corr_away = pw._maybe_correct_onice_for_goal(
            int(sec_i),
            goal_ev,
            set(pre_home),
            set(pre_away),
            game.home_team_id,
            game.away_team_id,
            team_onice_by_sec,
            orders_by_sec,
            shift_changes_by_sec,
        )
        corr_home = sorted(int(x) for x in corr_home)
        corr_away = sorted(int(x) for x in corr_away)
        on_for = corr_home if owner_side == "home" else corr_away
        on_opp = corr_away if owner_side == "home" else corr_home

        # generate token rows (state engine) to check coverage
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
            horizon_sec=int(args.horizon_sec),
            min_swap=2,
            min_gap_sec=8,
            stable_sec=2,
            rc_require_stable=1,
            post_dwell_sec=2,
            mass_swap_suppress_threshold=3,
            max_tokens=10,
            mode="sim",
            player_name_map={},
            player_pos_map={},
            season="",
            date="",
            rinkid="",
            min_exposure_sec=int(args.min_exposure_sec),
            state_stable_sec=int(args.state_stable_sec),
            state_chunk_by_h=int(args.state_chunk_by_h),
        )

        # Build quick lookup for outcome row that covers sec for (side,player)
        cover: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for r in token_rows:
            if str(r.get("window_id") or "") != win_id:
                continue
            if int(r.get("is_outcome_token") or 0) != 1:
                continue
            t0 = int(r.get("t_token") or 0)
            t1 = int(r.get("t_end") or 0)
            if int(t0) < sec_i <= int(t1):
                cover[(str(r.get("team_side") or ""), int(r.get("playerId") or 0))] = r

        failures: List[str] = []
        missing_outcome = 0

        # Check per-player per-second deltas for every on-ice player (using single-second recompute).
        for pid in on_for:
            side = owner_side
            d = _per_second_delta(
                game=game,
                team_onice_by_sec=team_onice_by_sec,
                events_by_sec=events_by_sec,
                credit_by_sec=credit_by_sec,
                orders_by_sec=orders_by_sec,
                shift_changes_by_sec=shift_changes_by_sec,
                side=side,
                player_id=int(pid),
                win_start=win_s,
                win_end=win_e,
                sec=sec_i,
            )
            if int(d.get("GF", 0)) != 1:
                failures.append(f"on_for pid={pid} expected GF=1 got {d.get('GF')}")
            if int(d.get("GA", 0)) != 0:
                failures.append(f"on_for pid={pid} expected GA=0 got {d.get('GA')}")
            if (side, int(pid)) not in cover:
                missing_outcome += 1

        opp_side = "away" if owner_side == "home" else "home"
        for pid in on_opp:
            d = _per_second_delta(
                game=game,
                team_onice_by_sec=team_onice_by_sec,
                events_by_sec=events_by_sec,
                credit_by_sec=credit_by_sec,
                orders_by_sec=orders_by_sec,
                shift_changes_by_sec=shift_changes_by_sec,
                side=opp_side,
                player_id=int(pid),
                win_start=win_s,
                win_end=win_e,
                sec=sec_i,
            )
            if int(d.get("GA", 0)) != 1:
                failures.append(f"on_opp pid={pid} expected GA=1 got {d.get('GA')}")
            if int(d.get("GF", 0)) != 0:
                failures.append(f"on_opp pid={pid} expected GF=0 got {d.get('GF')}")
            if (opp_side, int(pid)) not in cover:
                missing_outcome += 1

        per_second_ok = 1 if (not failures) else 0
        total_missing_outcome += int(missing_outcome)

        # Write markdown for manual inspection
        md = []
        md.append(f"# Goal case {idx:04d} — game {gamePk} sec={sec_i}")
        md.append("")
        md.append(f"- goal: `{_goal_label(goal_ev, game.home_team_id, game.away_team_id)}`")
        md.append(f"- window: `{win_id}` start={win_s} end={win_e}")
        md.append(f"- pre_onice_home: `{','.join(str(x) for x in pre_home)}`")
        md.append(f"- pre_onice_away: `{','.join(str(x) for x in pre_away)}`")
        md.append(f"- corrected_onice_home: `{','.join(str(x) for x in corr_home)}`")
        md.append(f"- corrected_onice_away: `{','.join(str(x) for x in corr_away)}`")
        md.append("")
        md.append("## Per-player checks (per-second delta at goal second)")
        md.append(f"- missing_outcome_rows_for_onice_players: {missing_outcome}")
        if failures:
            md.append("### FAILURES")
            for f in failures[:50]:
                md.append(f"- {f}")
        else:
            md.append("- PASS (all on-ice players get correct GF/GA delta)")
        md.append("")
        md.append("## Note on 'missing_outcome_rows_for_onice_players'")
        md.append(
            "- This is **token-horizon coverage**, not credit correctness. With `STATE` tokens emitted only when the matchup state changes (no chunking), "
            "a goal can happen far (>H seconds) after the last token, so **no token row will include that goal in (t_token, t_end]**. "
            "That is expected behavior unless you re-enable chunking (emit periodic STATE tokens every H seconds)."
        )
        md.append("")
        md.append("## Token coverage examples (first 5 on-for, first 5 on-opp)")
        md.append("### on-for")
        for pid in on_for[:5]:
            rr = cover.get((owner_side, int(pid)))
            md.append(f"- pid={pid} covered_row={('YES' if rr else 'NO')} t={rr.get('t_token') if rr else ''}..{rr.get('t_end') if rr else ''} type={rr.get('token_type') if rr else ''}")
        md.append("### on-opp")
        for pid in on_opp[:5]:
            rr = cover.get((opp_side, int(pid)))
            md.append(f"- pid={pid} covered_row={('YES' if rr else 'NO')} t={rr.get('t_token') if rr else ''}..{rr.get('t_end') if rr else ''} type={rr.get('token_type') if rr else ''}")
        (cases_dir / f"goal_case_{idx:04d}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

        rows.append(
            GoalCaseSummary(
                case_id=int(idx),
                gamePk=int(gamePk),
                sec=int(sec_i),
                window_id=str(win_id),
                window_start=int(win_s),
                window_end=int(win_e),
                goal_label=str(_goal_label(goal_ev, game.home_team_id, game.away_team_id)),
                owner_side=str(owner_side),
                pre_onice_home="|".join(str(x) for x in pre_home),
                pre_onice_away="|".join(str(x) for x in pre_away),
                corr_onice_home="|".join(str(x) for x in corr_home),
                corr_onice_away="|".join(str(x) for x in corr_away),
                per_second_ok=int(per_second_ok),
                failures="; ".join(failures[:5]),
                missing_outcome_rows=int(missing_outcome),
            )
        )

    df = pd.DataFrame([asdict(r) for r in rows])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "goal_cases.csv", index=False)
    (out_dir / "goal_cases.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    ok_n = int((df.per_second_ok == 1).sum())
    print(
        f"wrote {len(df)} goal cases to {out_dir} per_second_ok={ok_n} bad={len(df)-ok_n} "
        f"avg_missing_outcome_rows_per_goal={total_missing_outcome/max(1,len(df)):.2f}"
    )


if __name__ == "__main__":
    main()

