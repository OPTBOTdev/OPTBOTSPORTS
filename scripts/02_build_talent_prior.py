"""Build the EB-shrunken league-relative talent prior from REBUILT residuals.

Outputs:
  artifacts/talent_asof.parquet   — per (player, game): lagged talent state
  artifacts/talent_K.json         — fitted shrinkage constant + fit table
  artifacts/talent_snapshot_<date>.parquet — frozen table for a given date

Prints the sniff test: top/bottom 15 by shrunken offensive talent, latest snapshot.
McDavid/MacKinnon/Pastrnak at the top or you don't ship.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.priors.talent import build_asof_table, fit_K, snapshot  # noqa: E402

OBS = r"D:\optbot\artifacts\player_game_obs_rebuilt.parquet"
ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-life", type=float, default=40.0)
    ap.add_argument("--as-of", default="2026-07-01")
    args = ap.parse_args()

    obs = pd.read_parquet(OBS)
    asof = build_asof_table(obs, half_life_games=args.half_life)
    asof.to_parquet(ART / "talent_asof.parquet", index=False)

    K, tab = fit_K(asof)
    (ART / "talent_K.json").write_text(json.dumps(
        {"K_minutes": K, "half_life_games": args.half_life,
         "fit_table": tab.to_dict("records")}, indent=2))
    print(f"fitted K = {K:.0f} effective minutes\n{tab.to_string(index=False)}\n")

    snap = snapshot(asof, args.as_of, K)
    snap.to_parquet(ART / f"talent_snapshot_{args.as_of}.parquet", index=False)

    vets = snap[snap.talent_n_eff > 15].sort_values("talent_off_shrunk")
    print("=== SNIFF TEST (talent_off_shrunk, n_eff>15h) ===")
    print("TOP 15:")
    print(vets.tail(15).iloc[::-1].to_string(index=False))
    print("BOTTOM 15:")
    print(vets.head(15).to_string(index=False))
