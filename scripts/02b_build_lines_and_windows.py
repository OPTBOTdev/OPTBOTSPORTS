"""Build lines.parquet (role slots) and perfect_windows.parquet (v0 join surface).
Unblocks 04_project_trades / 05_backtest.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.data.lines import build_lines  # noqa: E402
from optbot.data.build_perfect_windows import build  # noqa: E402

ART = Path(r"D:\optbot\artifacts")
SPINE = ART / "window_spine_dedup.parquet"

if __name__ == "__main__":
    if not (ART / "lines.parquet").exists():
        print("1/2 lines ...")
        spine_cols = ["playerId", "teamId", "gamePk", "date", "seconds",
                      "strength_global", "window_id", "season"]
        spine = pd.read_parquet(SPINE, columns=spine_cols)
        lines = build_lines(spine)
        lines.to_parquet(ART / "lines.parquet", index=False)
        print(f"   {len(lines):,} player-game role rows")
        del spine, lines
    else:
        print("1/2 lines ... exists, skipping")

    print("2/2 perfect windows ...")
    stats = build(str(SPINE),
                  r"D:\baseline_model_output\player_windows_with_baseline_2*.parquet",
                  str(ART / "talent_asof.parquet"),
                  str(ART / "lines.parquet"),
                  str(ART / "perfect_windows.parquet"))
    print("  ", stats)
