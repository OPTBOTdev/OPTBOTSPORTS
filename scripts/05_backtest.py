"""THE number: run the freeze-disciplined backtest, fit conformal bands, print headline.

  python scripts/05_backtest.py            # full run
  python scripts/05_backtest.py --audit    # executable leakage checklist (5 moves)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.backtest.harness import run, headline, audit_freeze  # noqa: E402
from optbot.cin.conformal import ConformalBands  # noqa: E402

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--horizon", type=int, default=40)
    args = ap.parse_args()

    ledger = pd.read_parquet(ART / "ledger.parquet")
    obs = pd.read_parquet(ART / "player_game_obs_rebuilt.parquet")
    windows = pd.read_parquet(ART / "perfect_windows.parquet")
    lines = pd.read_parquet(ART / "lines.parquet")
    asof = pd.read_parquet(ART / "talent_asof.parquet")
    K = json.loads((ART / "talent_K.json").read_text())["K_minutes"]

    if args.audit:
        ok, results = audit_freeze(ledger, obs, windows, lines, asof, K)
        print(json.dumps(results, indent=2))
        sys.exit(0 if ok else 1)

    bt = run(ledger, obs, windows, lines, asof, K, args.horizon)
    bt.to_parquet(ART / "backtest_results.parquet", index=False)

    h = headline(bt)
    print(json.dumps(h, indent=2))
    if not h["claim_ok"]:
        print("\n!! CI crosses zero — no product claim yet. See error slices below.")
    for col in ["move_type"]:
        print(f"\nRMSE by {col}:")
        print(bt.groupby(col).apply(
            lambda g: ((g.pred_v0 - g.actual_xgf_pct) ** 2).mean() ** 0.5).round(2))

    bands = ConformalBands(target=0.80).fit(bt.dropna(subset=["pred_v0"]),
                                            pred_col="pred_v0")
    bands.save(str(ART / "conformal_bands.json"))
    print("\nconformal bands per n_eff bin (achieved coverage):", bands.coverage_)
