"""Fix 1: the window spine carries full-row duplicates in ALL 8 seasons
(~14.5% extra rows from the combine step). Writes a deduplicated spine + report.

ALSO logs which windows were affected, because anything downstream that summed
`seconds` over this file double-counted exposure for those windows.
"""
import sys
from pathlib import Path

import pandas as pd

SRC = r"D:\combined_player_windows_2017_2024.parquet"
OUT = r"D:\optbot\artifacts\window_spine_dedup.parquet"
REP = r"D:\optbot\artifacts\spine_dedup_report.txt"

if __name__ == "__main__":
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    n0 = len(df)
    dup_mask = df.duplicated()                      # full-row
    affected = df.loc[df.duplicated(keep=False), ["season", "gamePk"]].drop_duplicates()
    df = df[~dup_mask]
    # after full-row dedupe, any REMAINING key dupes are real conflicts -> fail loudly
    conflict = df.duplicated(["gamePk", "window_id", "playerId", "teamId"]).sum()
    with open(REP, "w") as f:
        f.write(f"rows before: {n0:,}\nrows after:  {len(df):,}\n"
                f"removed:     {n0 - len(df):,} full-row duplicates\n"
                f"affected games: {affected.gamePk.nunique():,} across "
                f"{affected.season.nunique()} seasons\n"
                f"remaining key conflicts: {conflict}\n")
    if conflict:
        print(f"FATAL: {conflict} non-identical key duplicates remain — inspect before use")
        sys.exit(1)
    df.to_parquet(OUT, index=False)
    print(open(REP).read())
    print("wrote", OUT)
