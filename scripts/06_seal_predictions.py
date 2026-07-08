"""Seal the 2026-27 mover projections: freeze -> hash -> print the tweetable digest.

  python scripts/06_seal_predictions.py --as-of 2026-09-15
Weekly during the season:
  python scripts/06_seal_predictions.py --score artifacts/sealed_2026.json
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.cin.scenario import Scenario  # noqa: E402
from optbot.cin.project import project_v0  # noqa: E402
from optbot.cin.conformal import ConformalBands  # noqa: E402
from optbot.priors.talent import snapshot  # noqa: E402
from optbot.seal.seal import seal, verify, score  # noqa: E402

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default="2026-09-15")
    ap.add_argument("--movers", default=str(ART / "movers_2026.csv"),
                    help="csv: player_id,dest_team,line_no,effective_date")
    ap.add_argument("--score", default=None, help="score an existing sealed file")
    args = ap.parse_args()

    if args.score:
        obs = pd.read_parquet(ART / "player_game_obs_rebuilt.parquet")
        board = score(args.score, obs)
        print(board.to_string(index=False))
        hit = board.inside_band.dropna()
        if len(hit):
            print(f"\ninside band: {hit.sum()}/{len(hit)} ({hit.mean():.0%})")
        sys.exit(0)

    movers = pd.read_csv(args.movers)
    windows = pd.read_parquet(ART / "perfect_windows.parquet")
    lines = pd.read_parquet(ART / "lines.parquet")
    asof = pd.read_parquet(ART / "talent_asof.parquet")
    K = json.loads((ART / "talent_K.json").read_text())["K_minutes"]
    tal = snapshot(asof, args.as_of, K)
    cb = ConformalBands(target=0.80)
    cb_state = json.loads((ART / "conformal_bands.json").read_text())
    cb.q_ = {int(k): tuple(v) for k, v in cb_state["q"].items()}
    cb.coverage_ = {int(k): v for k, v in cb_state["coverage"].items()}
    cb.edges = cb_state["edges"]

    preds = []
    for mv in movers.itertuples():
        prow = tal[tal.playerId == mv.player_id]
        prow = prow.iloc[0] if len(prow) else pd.Series(dtype=float)
        sc = Scenario(player_id=int(mv.player_id), as_of_date=args.as_of,
                      dest_team=int(mv.dest_team), line_no=int(mv.line_no))
        p = project_v0(sc, windows, lines, prow)
        band = cb.band(p["xgf_pct"], p["talent_n_eff"])
        preds.append({"player_id": int(mv.player_id), "dest_team": int(mv.dest_team),
                      "line_no": int(mv.line_no), "effective_date": str(mv.effective_date),
                      "pred_xgf_pct": round(p["xgf_pct"], 2),
                      "band_lo": round(band["lo"], 2), "band_hi": round(band["hi"], 2),
                      "achieved_coverage": round(band["achieved_coverage"], 3),
                      "scenario_hash": p["scenario_hash"]})

    out = seal(preds, str(ART / "sealed_2026.json"), model_version="cin_v0")
    assert verify(out["file"])
    print(json.dumps(out, indent=2))
    print("\nPOST THIS PUBLICLY (dated):", out["publish_this"])
