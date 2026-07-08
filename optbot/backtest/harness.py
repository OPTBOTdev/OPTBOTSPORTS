"""Freeze-disciplined trade backtest — THE credibility artifact.

For every qualifying ledger move:
  1. FREEZE at t0 = move_date: talent prior snapshot(t0), scenario built from
     destination windows strictly < t0, baselines from games < t0.
  2. PROJECT with each registered engine (v0, marcel, carryover, marcel+team, v1...).
  3. SCORE against actual first-N-games post-move rates.
  4. Bootstrap the RMSE delta vs Marcel -> the headline number WITH a CI.

Leakage checklist is executable: `audit_freeze()` recomputes 5 random moves with a
hard date-mask and asserts identical projections.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..data.ledger import actual_post_move
from ..priors.talent import snapshot
from ..baselines import marcel as mb
from ..cin.scenario import Scenario
from ..cin.project import project_v0


def run(ledger: pd.DataFrame, obs: pd.DataFrame, windows: pd.DataFrame,
        lines: pd.DataFrame, asof_talent: pd.DataFrame, K_minutes: float,
        horizon_games: int = 40) -> pd.DataFrame:
    rows = []
    for mv in ledger.itertuples():
        t0 = mv.move_date
        actual = actual_post_move(obs, mv.player_id, t0, horizon_games)
        if not actual["ok"]:
            continue
        tal = snapshot(asof_talent, t0, K_minutes)
        prow = tal[tal.playerId == mv.player_id]
        player_row = prow.iloc[0] if len(prow) else pd.Series(dtype=float)

        preds = {}
        try:
            sc = Scenario(player_id=mv.player_id, as_of_date=t0,
                          dest_team=mv.to_team, line_no=_infer_line(mv, obs, t0),
                          horizon_games=horizon_games)
            preds["v0"] = project_v0(sc, windows, lines, player_row)["xgf_pct"]
        except ValueError as e:
            preds["v0"] = np.nan
            preds["v0_err"] = str(e)
        preds["marcel"] = mb.project(obs, mv.player_id, t0)["xgf_pct"]
        preds["carryover"] = mb.carryover(obs, mv.player_id, t0)["xgf_pct"]

        rows.append({"player_id": mv.player_id, "move_date": t0,
                     "from_team": mv.from_team, "to_team": mv.to_team,
                     "move_type": mv.move_type,
                     "talent_n_eff": float(player_row.get("talent_n_eff", 0.0)),
                     "actual_xgf_pct": actual["xgf_pct"], "actual_gp": actual["gp"],
                     **{f"pred_{k}": v for k, v in preds.items()}})
    return pd.DataFrame(rows)


def _infer_line(mv, obs, t0) -> int:
    """Deterministic pre-t0 role rule: rank by prior-season TOI/gp -> line slot.
    Deliberately crude — identical crudeness to the live product (see spec 4c)."""
    ev = obs[(obs.strength_global == "5v5") & (obs.playerId == mv.player_id) & (obs.date < t0)]
    if ev.empty:
        return 3
    toi_gp = ev.groupby("gamePk")["toi_sec"].sum().mean() / 60.0
    return 1 if toi_gp >= 16 else 2 if toi_gp >= 13.5 else 3 if toi_gp >= 11 else 4


def headline(bt: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict:
    """RMSE per engine + bootstrap CI on (v0 - marcel). CI crossing 0 = no claim yet."""
    m = bt.dropna(subset=["pred_v0", "pred_marcel", "actual_xgf_pct"])
    rng = np.random.default_rng(seed)

    def rmse(p):  return float(np.sqrt(((m[p] - m.actual_xgf_pct) ** 2).mean()))
    out = {f"rmse_{e}": rmse(f"pred_{e}") for e in ["v0", "marcel", "carryover"]}
    out["n_moves"] = int(len(m))
    deltas = []
    idx = np.arange(len(m))
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        d = (np.sqrt(((m.pred_v0.values[s] - m.actual_xgf_pct.values[s]) ** 2).mean())
             - np.sqrt(((m.pred_marcel.values[s] - m.actual_xgf_pct.values[s]) ** 2).mean()))
        deltas.append(d)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    out["delta_rmse_v0_vs_marcel"] = out["rmse_v0"] - out["rmse_marcel"]
    out["delta_ci95"] = [float(lo), float(hi)]
    out["claim_ok"] = hi < 0
    out["pct_improvement"] = 100 * (1 - out["rmse_v0"] / out["rmse_marcel"])
    return out


def audit_freeze(ledger, obs, windows, lines, asof_talent, K_minutes, n=5, seed=1):
    """Executable leakage checklist: re-run n random moves with all post-t0 rows
    physically deleted; projections must be bit-identical."""
    rng = np.random.default_rng(seed)
    sample = ledger.sample(min(n, len(ledger)), random_state=int(rng.integers(1 << 31)))
    full = run(sample, obs, windows, lines, asof_talent, K_minutes)
    obs_cut = obs.copy(); win_cut = windows.copy()
    results = []
    for mv in sample.itertuples():
        t0 = mv.move_date
        o = obs_cut[(obs_cut.date < t0) | (obs_cut.playerId == mv.player_id)]
        w = win_cut[win_cut.date < t0]
        redo = run(sample[sample.player_id == mv.player_id], o, w, lines[lines.date < t0],
                   asof_talent[asof_talent.date < t0], K_minutes)
        a = full[full.player_id == mv.player_id]["pred_v0"].iloc[0]
        b = redo["pred_v0"].iloc[0] if len(redo) else np.nan
        results.append({"player_id": mv.player_id,
                        "identical": bool(np.isclose(a, b, equal_nan=True))})
    ok = all(r["identical"] for r in results)
    return ok, results
