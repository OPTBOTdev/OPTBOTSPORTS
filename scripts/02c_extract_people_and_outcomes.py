"""Perfect Windows v2 feeder: extract the people payload + window outcomes
from final_windows game CSVs (schema-stable across all 7 seasons — verified).

Per (season, gamePk, window_id, playerId):
  with_ids[5], with_seconds[5]   <- teammates_onice_ids_w/_sec_w, SORTED SECONDS DESC
  vs_ids[5],   vs_seconds[5]     <- opponents (same treatment)
  y block: xGF xGA GF GA SF SA + hits/blocks/takeaways/giveaways

Sorting note: raw lists are NOT seconds-ordered (verified on 2024 sample) — the
legacy first-K-not-top-K bug lived here. We sort at extraction, contract enforces.

Resumable: one parquet part per season; skips parts that exist.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEASON_DIRS = {
    20182019: r"D:\2018", 20192020: r"D:\2019", 20202021: r"D:\2020",
    20212022: r"D:\2021", 20222023: r"D:\2022", 20232024: r"D:\2023",
    20242025: r"D:\2024",
    # rebuilt seasons live in the API vault (certified V1 engine, Jul 9)
    20252026: r"C:\Users\lilli\Downloads\API\API\Final\2025\derived",
    20162017: r"C:\Users\lilli\Downloads\API\API\Final\2016\derived",
    20172018: r"C:\Users\lilli\Downloads\API\API\Final\2017\derived",
}
ART = Path(r"D:\optbot\artifacts")
K = 5

USECOLS = ["date",   # per-game date — freeze protocol + ledger detection need it
           "gamePk", "window_id", "playerId", "teamId", "seconds", "duration",
           "teammates_onice_ids_w", "teammates_onice_sec_w",
           "opponents_onice_ids_w", "opponents_onice_sec_w",
           "xGF", "xGA", "GF", "GA", "SF", "SA",
           "hits_personal", "blocks_personal", "takeaways_forced", "giveaways_committed",
           # v3 additions (F2): entry-side stint context + multistint flag source
           "shift_count_in_window", "entered_after_start", "entry_offset_s",
           "onice_elapsed_at_window_start", "time_since_last_shift_s"]

Y_RENAME = {"xGF": "y_xGF", "xGA": "y_xGA", "GF": "y_GF", "GA": "y_GA",
            "SF": "y_SF", "SA": "y_SA", "hits_personal": "y_hits",
            "blocks_personal": "y_blocks", "takeaways_forced": "y_takeaways",
            "giveaways_committed": "y_giveaways"}


def parse_sorted(ids_str, sec_str, k=K):
    """pipe-delimited -> (ids[k], secs[k]) sorted seconds DESC, zero-padded."""
    if not isinstance(ids_str, str) or not ids_str:
        return [0] * k, [0.0] * k
    try:
        ids = np.array(ids_str.split("|"), dtype=np.int64)
        secs = np.array(str(sec_str).split("|"), dtype=np.float64)
    except ValueError:
        return [0] * k, [0.0] * k
    n = min(len(ids), len(secs))
    ids, secs = ids[:n], secs[:n]
    order = np.argsort(-secs, kind="stable")
    ids, secs = ids[order][:k], secs[order][:k]
    pad = k - len(ids)
    return (list(ids) + [0] * pad, list(secs) + [0.0] * pad)


def do_season(season: int) -> Path:
    out = ART / f"people_outcomes_{season}.parquet"
    if out.exists():
        print(f"{season}: exists, skip")
        return out
    d = SEASON_DIRS[season]
    sub = "windows" if "derived" in d else "final_windows"
    suf = "" if "derived" in d else "_xg"     # vault seasons are pre-xg-fill
    files = [f for f in sorted(glob.glob(f"{d}\\{sub}\\player_windows_train_*{suf}.csv"))
             if "backup" not in f and ("_xg" in f) == (suf == "_xg")]
    rows = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp, usecols=lambda c: c in USECOLS)
        w = df[["gamePk", "window_id", "playerId", "teamId", "seconds"]].copy()
        w["date"] = df["date"].iloc[0] if "date" in df.columns and len(df) else None
        parsed_w = [parse_sorted(a, b) for a, b in
                    zip(df["teammates_onice_ids_w"], df["teammates_onice_sec_w"])]
        parsed_v = [parse_sorted(a, b) for a, b in
                    zip(df["opponents_onice_ids_w"], df["opponents_onice_sec_w"])]
        w["with_ids"] = [p[0] for p in parsed_w]
        w["with_seconds"] = [p[1] for p in parsed_w]
        w["vs_ids"] = [p[0] for p in parsed_v]
        w["vs_seconds"] = [p[1] for p in parsed_v]
        for src, dst in Y_RENAME.items():
            w[dst] = df[src] if src in df.columns else np.nan
        for c in ("shift_count_in_window", "entered_after_start", "entry_offset_s",
                  "onice_elapsed_at_window_start", "time_since_last_shift_s"):
            w[c] = df[c] if c in df.columns else np.nan
        w["is_multistint"] = (w["shift_count_in_window"].fillna(1) > 1).astype("int8")
        rows.append(w)
        if (i + 1) % 250 == 0:
            print(f"  {season}: {i+1}/{len(files)}")
    res = pd.concat(rows, ignore_index=True)
    res["season"] = season
    # per-season sanity before write
    dupe = res.duplicated(["gamePk", "window_id", "playerId"]).mean()
    assert dupe < 0.001, f"{season}: {dupe:.3%} dup keys in extraction source!"
    res.to_parquet(out, index=False)
    print(f"{season}: {len(res):,} rows -> {out.name}  (dupe {dupe:.4%})")
    return out


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or list(SEASON_DIRS)
    parts = [do_season(s) for s in seasons]
    allp = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    allp.to_parquet(ART / "people_outcomes_all.parquet", index=False)
    print(f"TOTAL {len(allp):,} rows -> people_outcomes_all.parquet")
    zero_y = float((allp["y_xGF"] == 0).mean())
    print(f"sanity: y_xGF==0 share {zero_y:.1%} (window-level zeros are normal; ~70-90%)")
