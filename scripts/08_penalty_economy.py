"""Penalty-economy transport backtest: does penalty skill survive a team change?

For each ledger move: frozen shrunk prior delta60 at t0  vs  actual first-40-GP
post-move delta60. Baselines: zero (league average) and raw last-prior (unshrunk).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.priors.penalty_economy import (NET_GOALS_PER_PENALTY, actual_post,
                                           load_all, snapshot)

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    pen = load_all()
    print(f"penalty per-game rows: {len(pen):,} "
          f"({pen.gameid.min()} .. {pen.gameid.max()})")
    ledger = pd.read_parquet(ART / "ledger.parquet")
    # date -> gamePk mapping from the obs table: the player's first game ON/AFTER
    # the move date is his first game with the new team (that's how the ledger
    # detected the move in the first place).
    obs = pd.read_parquet(ART / "player_game_obs_rebuilt.parquet",
                          columns=["playerId", "gamePk", "date"]).drop_duplicates()

    rows = []
    for mv in ledger.itertuples():
        first = obs[(obs.playerId == mv.player_id) & (obs.date >= mv.move_date)]
        if first.empty:
            continue
        t0_game = int(first.sort_values("date").gamePk.iloc[0])
        snap = snapshot(pen, t0_game)
        prow = snap[snap.playerId == mv.player_id]
        if prow.empty:
            continue
        act = actual_post(pen, mv.player_id, t0_game)
        if not act["ok"]:
            continue
        rows.append({"player_id": mv.player_id, "move_date": str(mv.move_date),
                     "pred_delta60": float(prow.pen_delta60_shrunk.iloc[0]),
                     "pred_raw": float(prow.pen_taken60_prior_ev.iloc[0] * -1
                                       + prow.pen_drawn60_prior_ev.iloc[0]),
                     "n_eff_min": float(prow.pen_neff_minutes_prior.iloc[0]),
                     "actual_delta60": act["actual_delta60"], "gp": act["gp"]})
    bt = pd.DataFrame(rows)
    bt.to_parquet(ART / "penalty_backtest.parquet", index=False)
    print(f"scoreable moves: {len(bt)}")

    def rmse(pred):
        return float(np.sqrt(((pred - bt.actual_delta60) ** 2).mean()))
    r_shrunk = rmse(bt.pred_delta60)
    r_raw = rmse(bt.pred_raw.fillna(0))
    r_zero = rmse(pd.Series(0.0, index=bt.index))
    corr = float(bt.pred_delta60.corr(bt.actual_delta60))
    print(f"corr(shrunk prior, post-move actual): {corr:.3f}")
    print(f"RMSE  shrunk {r_shrunk:.3f} | raw {r_raw:.3f} | zero-baseline {r_zero:.3f}")
    print(f"transport verdict: {'SKILL TRANSPORTS' if r_shrunk < r_zero and corr > 0.2 else 'WEAK — investigate'}")
    hi = bt[bt.n_eff_min > 1500]
    if len(hi) > 50:
        print(f"veterans only (n_eff>1500min, n={len(hi)}): corr "
              f"{hi.pred_delta60.corr(hi.actual_delta60):.3f}")
    print(f"\neconomy scale: 1.0 delta60 = {NET_GOALS_PER_PENALTY:.2f} goals/60 "
          f"~ {NET_GOALS_PER_PENALTY * 60 / 60 * 82 * 15 / 60:.1f} goals/82GP at 15min/night")
