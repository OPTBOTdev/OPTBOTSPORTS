"""Score Marcel + carryover on the ledger — the bar, standalone (no CIN needed).
Run this the moment the ledger exists: it tells you the target RMSE before any
of your own modeling enters the picture.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.baselines import marcel as mb  # noqa: E402
from optbot.data.ledger import actual_post_move  # noqa: E402

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    ledger = pd.read_parquet(ART / "ledger.parquet")
    obs = pd.read_parquet(ART / "player_game_obs_rebuilt.parquet")
    rows = []
    for mv in ledger.itertuples():
        a = actual_post_move(obs, mv.player_id, mv.move_date)
        if not a["ok"]:
            continue
        rows.append({"player_id": mv.player_id, "move_date": mv.move_date,
                     "actual": a["xgf_pct"],
                     "marcel": mb.project(obs, mv.player_id, mv.move_date)["xgf_pct"],
                     "carryover": mb.carryover(obs, mv.player_id, mv.move_date)["xgf_pct"]})
    df = pd.DataFrame(rows)
    df.to_parquet(ART / "baseline_scores.parquet", index=False)
    for c in ("marcel", "carryover"):
        rmse = float(np.sqrt(((df[c] - df.actual) ** 2).mean()))
        print(f"{c:10s} RMSE(xGF%) = {rmse:.2f}   n={len(df)}")
    print("\n^ THE BAR. cin v0 must beat the marcel line with CI clear of zero.")
