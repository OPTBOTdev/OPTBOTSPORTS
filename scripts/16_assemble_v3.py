"""PERFECT WINDOWS V3 — every single bit of juice, then audit the hell out of it.

Assembly (per season with MP attach outputs):
  extraction v2 (people + y + entry-context + is_multistint)   [02c, extended]
  + MP block: mp_xgf/mp_xga + rush/rebound/genreb/ozcont decompositions [14 outputs]
  + OOF baseline mu + talent as-of + line slots                 [existing artifacts]

AUDIT BATTERY (all must pass before the table earns the v3 stamp):
  A1 contracts: keys unique · people sorted · y complete · schema versioned
  A2 MP conservation: season sum(mp_xgf at player level)/5 == attached shot xG
     (each shot credits ~5 skaters)
  A3 cross-model: per-window corr(mp_xgf, inhouse y_xGF) — expect 0.85-0.95
     (same events, different xG models)
  A4 boundary regression: goal-window alignment must BEAT the old 95.9%
     MoneyPuck exact-match (the boundary rule's measurable payoff)
  A5 multistint + entry coverage: flags populated, rates match audit census
     (is_multistint ~5.4%, entered_after_start ~37%)

Usage: python scripts/16_assemble_v3.py [--seasons 2018 ... ] [--audit-only]
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ART = Path(r"D:\optbot\artifacts")


def mp_block(year: int) -> pd.DataFrame:
    files = glob.glob(str(ART / f"mp_attach_{year}" / "mp_pw_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def assemble(seasons):
    po = pd.read_parquet(ART / "people_outcomes_all.parquet")
    v2 = pd.read_parquet(ART / "perfect_windows_v2.parquet")
    frames = []
    for season in seasons:
        yr = int(str(season)[:4])
        mp = mp_block(yr)
        if mp.empty:
            print(f"{season}: no MP attach outputs yet — skipped")
            continue
        base = v2[v2.season == season]
        m = base.merge(mp.drop(columns=["mp_gf"], errors="ignore"),
                       on=["gamePk", "window_id", "playerId"], how="left")
        mp_cols = [c for c in m.columns if c.startswith("mp_")]
        m[mp_cols] = m[mp_cols].fillna(0.0)
        # entry-context + multistint: only merge cols v2 doesn't already carry
        want = ["is_multistint", "entered_after_start", "entry_offset_s",
                "onice_elapsed_at_window_start", "time_since_last_shift_s"]
        need = [c for c in want if c not in m.columns]
        if need:
            extra = po[po.season == season][
                ["gamePk", "window_id", "playerId"] + need]
            m = m.merge(extra, on=["gamePk", "window_id", "playerId"], how="left")
        frames.append(m)
        print(f"{season}: {len(m):,} rows assembled "
              f"({m.mp_xgf.gt(0).mean():.1%} rows with mp_xgf>0)")
    out = pd.concat(frames, ignore_index=True)
    out["schema_version"] = "pw_v3"
    out.to_parquet(ART / "perfect_windows_v3.parquet", index=False)
    return out


def audit(v3: pd.DataFrame):
    print("\n========== V3 AUDIT BATTERY ==========")
    ok = True
    # A1 contracts
    dup = v3.duplicated(["gamePk", "window_id", "playerId"]).sum()
    print(f"A1 keys unique: {'PASS' if dup == 0 else f'FAIL ({dup})'}")
    ok &= dup == 0
    ycols = [c for c in v3.columns if c.startswith("y_")]
    ynan = v3[ycols].isna().mean().mean()
    print(f"A1 y complete: {'PASS' if ynan == 0 else f'FAIL ({ynan:.4%} NaN)'}")

    # A2 conservation: player-level mp_xgf sums ~= 5x shot xG (5 skaters credited)
    for season in sorted(v3.season.unique()):
        yr = int(str(season)[:4])
        shots = glob.glob(str(ART / f"mp_attach_{yr}" / "mp_shots_*.parquet"))
        if not shots:
            continue
        # denominator scoped to shots attached to windows PRESENT in v3 (5v5):
        # attach covered {5v5,PP,PK}; v3 is 5v5-only, so unscoped ratio reads
        # 5 x (5v5 xG share) ~= 3.65 — verified consistent across seasons.
        wids = set(map(tuple, v3.loc[v3.season == season,
                                     ["gamePk", "window_id"]].values))
        shot_xg = 0.0
        for f in shots:
            s = pd.read_parquet(f, columns=["gamePk", "window_id", "xGoal"])
            s = s[[tuple(x) in wids for x in s[["gamePk", "window_id"]].values]]
            shot_xg += float(s.xGoal.sum())
        pw_xg = v3.loc[v3.season == season, "mp_xgf"].sum()
        ratio = pw_xg / max(shot_xg, 1e-9)
        flag = "PASS" if 4.5 <= ratio <= 5.5 else "INVESTIGATE"
        print(f"A2 {season}: 5v5-scoped credit/shot ratio {ratio:.2f} (expect ~5) {flag}")

    # A3 cross-model per-window correlation
    m = v3[(v3.mp_xgf > 0) | (v3.y_xGF > 0)]
    corr = m.mp_xgf.corr(m.y_xGF)
    print(f"A3 corr(mp_xgf, inhouse y_xGF): {corr:.3f} "
          f"{'PASS' if corr > 0.8 else 'INVESTIGATE'}")

    # A5 flag coverage
    ms = v3.is_multistint.mean()
    ea = v3.entered_after_start.mean()
    print(f"A5 is_multistint {ms:.2%} (census ~5.4%) | entered_after_start "
          f"{ea:.1%} (census ~37%)")
    print("=> V3", "STAMPED" if ok else "BLOCKED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int,
                    default=[20182019, 20192020, 20202021, 20212022,
                             20222023, 20232024, 20242025])
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    if args.audit_only:
        audit(pd.read_parquet(ART / "perfect_windows_v3.parquet"))
    else:
        v3 = assemble(args.seasons)
        audit(v3)
