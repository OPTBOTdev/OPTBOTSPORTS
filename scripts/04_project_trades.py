"""Ad-hoc projection CLI — the demo backend.

  python scripts/04_project_trades.py --player 8478402 --dest 16 --line 1
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.cin.scenario import Scenario  # noqa: E402
from optbot.cin.project import project_v0  # noqa: E402
from optbot.cin.support import check_environment_swap  # noqa: E402
from optbot.cin.conformal import ConformalBands  # noqa: E402
from optbot.priors.talent import snapshot  # noqa: E402

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--player", type=int, required=True)
    ap.add_argument("--dest", type=int, required=True)
    ap.add_argument("--line", type=int, default=2)
    ap.add_argument("--as-of", default="2026-07-01")
    args = ap.parse_args()

    windows = pd.read_parquet(ART / "perfect_windows.parquet")
    lines = pd.read_parquet(ART / "lines.parquet")
    asof = pd.read_parquet(ART / "talent_asof.parquet")
    K = json.loads((ART / "talent_K.json").read_text())["K_minutes"]
    tal = snapshot(asof, args.as_of, K)
    prow = tal[tal.playerId == args.player]
    prow = prow.iloc[0] if len(prow) else pd.Series(dtype=float)

    sc = Scenario(player_id=args.player, as_of_date=args.as_of,
                  dest_team=args.dest, line_no=args.line)
    p = project_v0(sc, windows, lines, prow)
    verdict = check_environment_swap(p["n_synth_windows"],
                                     float(prow.get("talent_n_eff", 0)))
    p["support"] = verdict.__dict__
    try:
        cb_state = json.loads((ART / "conformal_bands.json").read_text())
        cb = ConformalBands(); cb.q_ = {int(k): tuple(v) for k, v in cb_state["q"].items()}
        cb.coverage_ = {int(k): v for k, v in cb_state["coverage"].items()}
        cb.edges = cb_state["edges"]
        p["band80"] = cb.band(p["xgf_pct"], p["talent_n_eff"])
    except FileNotFoundError:
        p["band80"] = "run 05_backtest.py first"
    print(json.dumps(p, indent=2))
