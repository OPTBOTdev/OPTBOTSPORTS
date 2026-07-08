"""Perfect Windows v2: join people payload + window outcomes onto v1.
Drops 2017-18 (no extraction source, no talent obs). Validates the completed blocks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")

if __name__ == "__main__":
    pw = pd.read_parquet(ART / "perfect_windows.parquet")
    pw = pw[pw.season != 20172018]
    po = pd.read_parquet(ART / "people_outcomes_all.parquet")
    keys = ["season", "gamePk", "window_id", "playerId"]
    v2 = pw.merge(po.drop(columns=["teamId", "seconds"]), on=keys, how="inner")
    print(f"v1 rows (7 seasons): {len(pw):,}  ->  v2 rows: {len(v2):,} "
          f"({len(v2)/len(pw):.2%} matched)")

    # block validation (full pw_v1 contract lands when intrinsic priors join)
    assert v2.duplicated(keys).sum() == 0, "dup keys in v2"
    s = v2.sample(min(len(v2), 150_000), random_state=0)
    for c in ("with_seconds", "vs_seconds"):
        ok = s[c].map(lambda v: bool(np.all(np.asarray(v)[:-1] >= np.asarray(v)[1:]))).mean()
        assert ok == 1.0, f"{c} sort contract broken: {ok:.4%}"
    ycols = [c for c in v2.columns if c.startswith("y_")]
    assert v2[ycols].notna().all().all(), "NaNs in y block"
    print("contracts: keys unique | people sorted | y complete  ALL PASS")
    print("y cols:", ycols)

    v2["schema_version"] = "pw_v2"
    v2.to_parquet(ART / "perfect_windows_v2.parquet", index=False)
    print("wrote perfect_windows_v2.parquet",
          f"({v2.memory_usage(deep=False).sum()/1e9:.1f} GB in-memory)")
