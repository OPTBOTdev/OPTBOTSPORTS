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


MICRO_COLS = [
    "giveaways_committed",
    "takeaways_forced",
    "hits_personal",
    "blocks_personal",
    "shots_blocked_personal",
    "giveaways_committed_oz",
    "giveaways_committed_nz",
    "giveaways_committed_dz",
    "takeaways_forced_oz",
    "takeaways_forced_nz",
    "takeaways_forced_dz",
    "hits_personal_oz",
    "hits_personal_nz",
    "hits_personal_dz",
    "blocks_personal_oz",
    "blocks_personal_nz",
    "blocks_personal_dz",
    "shots_blocked_personal_oz",
    "shots_blocked_personal_nz",
    "shots_blocked_personal_dz",
]


def fast_aggregate_token(
    *,
    game: pw.ParsedGame,
    home_sets: List[set],
    away_sets: List[set],
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
    # Mirror aggregate_features_for_token core logic, but omit post overlap evidence.
    start_target = max(int(win_start), int(t_token) + 1)
    end_target = min(int(t_end), int(win_end))
    if end_target < start_target:
        out = {k: 0 for k in ("xGF", "xGA", "GF", "GA", "SF", "SA", "AF", "AA", "BF", "BA")}
        for k in MICRO_COLS:
            out[k] = 0
        return out

    onice_map = home_sets if side == "home" else away_sets
    opp_map = away_sets if side == "home" else home_sets

    Y = Counter()
    xgF = 0.0
    xgA = 0.0

    gv_p = tk_p = hp_p = bp_p = sbp_p = 0
    gv_zone = {zb: 0 for zb in pw.ZONE_BUCKETS}
    tk_zone = {zb: 0 for zb in pw.ZONE_BUCKETS}
    hp_zone = {zb: 0 for zb in pw.ZONE_BUCKETS}
    bp_zone = {zb: 0 for zb in pw.ZONE_BUCKETS}
    sbp_zone = {zb: 0 for zb in pw.ZONE_BUCKETS}

    for s in range(int(start_target), int(end_target) + 1):
        if p not in onice_map[s]:
            continue

        # micro events by actor at second s
        for ev in events_by_sec.get(int(s), []):
            et = str(ev.get("type") or "").lower()
            if et not in pw.MICRO_TYPES:
                continue
            det = ev.get("details") or {}
            if et in ("giveaway", "takeaway"):
                pid = _safe_int(det.get("playerId") or det.get("player_id") or det.get("actorPlayerId"), 0)
                if pid == p:
                    if et == "giveaway":
                        gv_p += 1
                        pw._bump_zone_counter(gv_zone, det)
                    else:
                        tk_p += 1
                        pw._bump_zone_counter(tk_zone, det)
            elif et == "hit":
                pid = _safe_int(det.get("hittingPlayerId") or det.get("hitterPlayerId") or det.get("hitterId") or det.get("playerId"), 0)
                if pid == p:
                    hp_p += 1
                    pw._bump_zone_counter(hp_zone, det)
            elif et == "blocked-shot":
                if str(det.get("reason", "")).lower() == "teammate-blocked":
                    continue
                blk_pid = _safe_int(det.get("blockingPlayerId") or det.get("blockedByPlayerId") or det.get("blockerPlayerId") or det.get("blockerId") or det.get("playerId"), 0)
                if blk_pid == p:
                    bp_p += 1
                    pw._bump_zone_counter(bp_zone, det)
                shot_pid = _safe_int(det.get("shootingPlayerId") or det.get("shooterId") or det.get("shooter"), 0)
                if shot_pid == p:
                    sbp_p += 1
                    pw._bump_zone_counter(sbp_zone, det)

        # shots/goals crediting (pre-change snapshots)
        cb = credit_by_sec.get(int(s)) or {}
        pre_home = list(cb.get("onice_home", ()))
        pre_away = list(cb.get("onice_away", ()))
        for ev in events_by_sec.get(int(s), []):
            et = str(ev.get("type") or "").lower()
            if et not in pw.SHOT_TYPES and et not in pw.GOAL_TYPES:
                continue
            det = ev.get("details") or {}
            if et in pw.BLOCK_TYPES and str(det.get("reason", "")).lower() == "teammate-blocked":
                continue
            ev_side = pw._owner_side_for_event(det, game.home_team_id, game.away_team_id)
            if ev_side not in ("home", "away"):
                continue

            if et in pw.GOAL_TYPES:
                corr_home, corr_away = pw._maybe_correct_onice_for_goal(
                    int(s),
                    ev,
                    set(pre_home),
                    set(pre_away),
                    game.home_team_id,
                    game.away_team_id,
                    team_onice_by_sec,
                    orders_by_sec,
                    shift_changes_by_sec,
                )
                on_for = list(corr_home) if ev_side == "home" else list(corr_away)
                on_opp = list(corr_away) if ev_side == "home" else list(corr_home)
            else:
                on_for = pre_home if ev_side == "home" else pre_away
                on_opp = pre_away if ev_side == "home" else pre_home

            xg = float(det.get("xg", 0.0)) if det.get("xg") is not None else 0.0
            if p in on_for:
                Y["AF"] += 1
                if et in pw.SHOT_ON_GOAL_TYPES or et in pw.GOAL_TYPES:
                    Y["SF"] += 1
                    xgF += xg
            elif p in on_opp:
                Y["AA"] += 1
                if et in pw.SHOT_ON_GOAL_TYPES or et in pw.GOAL_TYPES:
                    Y["SA"] += 1
                    xgA += xg

            if et in pw.BLOCK_TYPES:
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

            if et in pw.GOAL_TYPES:
                if p in on_for:
                    Y["GF"] += 1
                elif p in on_opp:
                    Y["GA"] += 1

    out = {
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
    }
    return out


def stable_team_at(onice_map: List[set], p: int, ss: int, seg_last_on: int, stable_sec: int) -> Optional[set]:
    if stable_sec <= 1:
        return set(onice_map[int(ss)]) - {p}
    last = None
    for tt in range(int(ss), int(ss) + int(stable_sec)):
        if tt > int(seg_last_on):
            return None
        if p not in onice_map[int(tt)]:
            return None
        cur = set(onice_map[int(tt)]) - {p}
        if last is None:
            last = cur
        elif cur != last:
            return None
    return last


def audit_roster_change_tokens(
    *,
    token_rows: List[Dict[str, Any]],
    home_sets: List[set],
    away_sets: List[set],
    shift_changes_by_sec: Dict[int, Dict[str, set]],
    min_swap: int,
    stable_sec: int,
    post_dwell_sec: int,
    max_examples: int = 50,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Validate kept ROSTER_CHANGE tokens satisfy the key invariants:
    - player is on-ice at t_token and remains on-ice through post_dwell_sec
    - lineup is stable for stable_sec seconds at t_token (if stable_sec>1)
    - material teammate change vs previous kept snapshot in same segment >= min_swap
    - there exists a shift/inferred-change second near the emitted time that could have triggered it
    """
    # group by (window_id, playerId, seg_idx, side)
    groups: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in token_rows:
        if str(r.get("token_type")).upper() != "ROSTER_CHANGE":
            continue
        groups[(str(r.get("window_id")), int(r.get("playerId")), int(r.get("seg_idx", 0)), str(r.get("team_side")))].append(r)

    counts = Counter()
    examples: List[Dict[str, Any]] = []

    for (win_id, p, seg_idx, side), rows in groups.items():
        onice_map = home_sets if side == "home" else away_sets
        # pick segment bounds from any row
        seg_s = _safe_int(rows[0].get("seg_start"), _safe_int(rows[0].get("seg_start_true"), 0))
        seg_last_on = _safe_int(rows[0].get("seg_last_on"), seg_s)
        if seg_s <= 0 and _safe_int(rows[0].get("seg_start_true"), 0) > 0:
            seg_s = _safe_int(rows[0].get("seg_start_true"), 0)

        rows_sorted = sorted(rows, key=lambda r: int(r.get("t_token")))
        prev_team = set(onice_map[int(seg_s)]) - {p} if int(seg_s) < len(onice_map) else set()
        prev_t = int(seg_s)

        for r in rows_sorted:
            t = int(r.get("t_token"))
            counts["rc_total"] += 1

            # on-ice at token
            if t >= len(onice_map) or p not in onice_map[t]:
                counts["rc_fail_office_at_t"] += 1
                if len(examples) < max_examples:
                    examples.append({"reason": "office_at_t", "window_id": win_id, "playerId": p, "t": t, "seg_idx": seg_idx, "side": side})
                continue

            # post-dwell
            ok_pd = True
            if int(post_dwell_sec) > 0:
                for tt in range(int(t), int(t) + int(post_dwell_sec) + 1):
                    if tt > int(seg_last_on) or tt >= len(onice_map) or p not in onice_map[int(tt)]:
                        ok_pd = False
                        break
            if not ok_pd:
                counts["rc_fail_post_dwell"] += 1
                if len(examples) < max_examples:
                    examples.append({"reason": "post_dwell", "window_id": win_id, "playerId": p, "t": t, "seg_idx": seg_idx, "side": side})
                continue

            # stability at emitted time
            if int(stable_sec) > 1:
                st = stable_team_at(onice_map, p, t, seg_last_on, stable_sec)
                if st is None:
                    counts["rc_fail_not_stable_at_emit"] += 1
                    if len(examples) < max_examples:
                        examples.append({"reason": "not_stable_at_emit", "window_id": win_id, "playerId": p, "t": t, "seg_idx": seg_idx, "side": side})
                    continue

            # material diff vs prev snapshot
            team_now = set(onice_map[t]) - {p}
            if len(team_now.symmetric_difference(prev_team)) < int(min_swap):
                counts["rc_fail_min_swap_vs_prev"] += 1
                if len(examples) < max_examples:
                    examples.append({"reason": "min_swap_vs_prev", "window_id": win_id, "playerId": p, "t": t, "seg_idx": seg_idx, "side": side})
                continue

            # plausible trigger second: find any second near emitted time where explicit shift or inferred change occurred.
            plausible = False
            start_scan = max(int(prev_t) + 1, int(t) - max(1, int(stable_sec)))
            for s in range(int(start_scan), int(t) + 1):
                if s <= 0 or s >= len(onice_map):
                    continue
                sc = shift_changes_by_sec.get(int(s)) or {}
                explicit_shift = bool(sc.get(f"{side}_in") or sc.get(f"{side}_out"))
                inferred_shift = bool(onice_map[int(s)] != onice_map[int(s) - 1])
                if explicit_shift or inferred_shift:
                    plausible = True
                    break
            if not plausible:
                counts["rc_warn_no_shift_near_emit"] += 1
                if len(examples) < max_examples:
                    examples.append({"reason": "no_shift_near_emit", "window_id": win_id, "playerId": p, "t": t, "seg_idx": seg_idx, "side": side})

            # update prev snapshot
            prev_team = set(team_now)
            prev_t = int(t)

    return dict(counts), examples


@dataclass
class GameTokenAuditSummary:
    gamePk: int
    file: str
    horizon_sec: int
    token_rows: int
    outcome_mismatch_rows: int
    outcome_mismatch_cells: int
    rc_tokens: int
    rc_failures: int
    rc_warnings: int


def main():
    ap = argparse.ArgumentParser(description="Full-season 2024 token outcome + roster-change audit (slow, thorough).")
    ap.add_argument("--pbp_dir", default="API/Final/2024/raw/pbp_built_20242025")
    ap.add_argument("--out_dir", default="artifacts/audits/season2024_full_token_audit")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for debugging (0=all)")
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--max_issue_examples", type=int, default=30)
    # audit params should match your generator defaults
    ap.add_argument("--horizon_sec", type=int, default=20)
    ap.add_argument("--min_swap", type=int, default=2)
    ap.add_argument("--min_gap_sec", type=int, default=4)
    ap.add_argument("--stable_sec", type=int, default=2)
    ap.add_argument("--post_dwell_sec", type=int, default=2)
    ap.add_argument("--rc_require_stable", type=int, default=1)
    ap.add_argument("--mass_swap_suppress_threshold", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=10)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.pbp_dir, "pbp_onice_*.json")))
    if int(args.limit) > 0:
        files = files[: int(args.limit)]
    if not files:
        raise SystemExit(f"No pbp_onice_*.json found in {args.pbp_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[GameTokenAuditSummary] = []
    issues: Dict[int, Dict[str, Any]] = {}

    for i, fp in enumerate(files, start=1):
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
            min_swap=int(args.min_swap),
            min_gap_sec=int(args.min_gap_sec),
            stable_sec=int(args.stable_sec),
            rc_require_stable=int(args.rc_require_stable),
            post_dwell_sec=int(args.post_dwell_sec),
            mass_swap_suppress_threshold=int(args.mass_swap_suppress_threshold),
            max_tokens=int(args.max_tokens),
            mode="both",
            player_name_map={},
            player_pos_map={},
            season="",
            date="",
            rinkid="",
        )

        # per-second sets for gating
        H = len(team_onice_by_sec) - 2
        home_sets = [set() for _ in range(H + 2)]
        away_sets = [set() for _ in range(H + 2)]
        for s in range(H + 2):
            hh, aa = team_onice_by_sec[s]
            home_sets[s] = set(int(x) for x in (hh or []))
            away_sets[s] = set(int(x) for x in (aa or []))

        win_by_id = {str(w["window_id"]): w for w in windows}

        mismatch_rows = 0
        mismatch_cells = 0
        ex_rows: List[Dict[str, Any]] = []

        # columns to compare (high-value + micro)
        compare_cols = ["xGF", "xGA", "GF", "GA", "SF", "SA", "AF", "AA", "BF", "BA"] + MICRO_COLS

        for r in token_rows:
            win_id = str(r.get("window_id") or "")
            w = win_by_id.get(win_id)
            if w is None:
                mismatch_rows += 1
                mismatch_cells += 1
                if len(ex_rows) < int(args.max_issue_examples):
                    ex_rows.append({"reason": "missing_window", "window_id": win_id})
                continue
            side = str(r.get("team_side"))
            p = int(r.get("playerId"))
            t_tok = int(r.get("t_token"))
            t_end = int(r.get("t_end"))

            exp = fast_aggregate_token(
                game=game,
                home_sets=home_sets,
                away_sets=away_sets,
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
            for c in compare_cols:
                got = r.get(c, 0)
                if c in ("xGF", "xGA"):
                    try:
                        got_f = float(got)
                    except Exception:
                        got_f = 0.0
                    if abs(float(exp.get(c, 0.0)) - got_f) > 1e-6:
                        bad.append((c, got, exp.get(c)))
                else:
                    if _safe_int(got, 0) != _safe_int(exp.get(c, 0), 0):
                        bad.append((c, got, exp.get(c)))
            if bad:
                mismatch_rows += 1
                mismatch_cells += len(bad)
                if len(ex_rows) < int(args.max_issue_examples):
                    ex_rows.append(
                        {
                            "window_id": win_id,
                            "team_side": side,
                            "playerId": p,
                            "token_type": r.get("token_type"),
                            "t_token": t_tok,
                            "t_end": t_end,
                            "bad": bad[:20],
                        }
                    )

        rc_counts, rc_examples = audit_roster_change_tokens(
            token_rows=token_rows,
            home_sets=home_sets,
            away_sets=away_sets,
            shift_changes_by_sec=shift_changes_by_sec,
            min_swap=int(args.min_swap),
            stable_sec=int(args.stable_sec),
            post_dwell_sec=int(args.post_dwell_sec),
            max_examples=int(args.max_issue_examples),
        )
        rc_total = int(rc_counts.get("rc_total", 0))
        rc_fail = sum(v for k, v in rc_counts.items() if k.startswith("rc_fail_"))
        rc_warn = sum(v for k, v in rc_counts.items() if k.startswith("rc_warn_"))

        summ = GameTokenAuditSummary(
            gamePk=int(game.gamePk),
            file=str(fp),
            horizon_sec=int(horizon),
            token_rows=int(len(token_rows)),
            outcome_mismatch_rows=int(mismatch_rows),
            outcome_mismatch_cells=int(mismatch_cells),
            rc_tokens=int(rc_total),
            rc_failures=int(rc_fail),
            rc_warnings=int(rc_warn),
        )
        summaries.append(summ)

        if mismatch_rows > 0 or rc_fail > 0:
            issues[int(game.gamePk)] = {
                "summary": asdict(summ),
                "outcome_examples": ex_rows,
                "rc_counts": rc_counts,
                "rc_examples": rc_examples,
            }

        if int(args.progress_every) > 0 and (i % int(args.progress_every) == 0 or i == len(files)):
            ok_n = sum(1 for s in summaries if s.outcome_mismatch_rows == 0 and s.rc_failures == 0)
            print(f"progress {i}/{len(files)} ok={ok_n} bad={i-ok_n}")

    # write
    rows = [asdict(s) for s in summaries]
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(GameTokenAuditSummary.__annotations__.keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "issues.json").write_text(json.dumps(issues, indent=2), encoding="utf-8")

    ok_n = sum(1 for s in summaries if s.outcome_mismatch_rows == 0 and s.rc_failures == 0)
    print(f"done audited={len(summaries)} ok={ok_n} bad={len(summaries)-ok_n} out_dir={out_dir}")


if __name__ == "__main__":
    main()

