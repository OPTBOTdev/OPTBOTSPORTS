"""Script 25 — DEEP AUDIT of the assembled trainer surface (paranoia pass).

Checks the things that would train a confident lie if wrong:
  A1 focal player must NEVER appear in his own with_ids / vs_ids (identity leak)
  A2 list alignment: every member-feature list same length as its ids list
  A3 padding consistency: ids==0 <=> seconds<=0 (loader masks by seconds)
  A4 no duplicate real member ids within a window side
  A5 with/vs disjoint (nobody on both teams at once)
  A6 coach join: same (gamePk, teamId) always same coach; 2 teams -> 2 coaches
  A7 b2b rate sane (8-25%), rest_days in [1,7]
  A8 member form scale sane (league xGF60 ~ 1.8-3.2), ages in [17,48]
  A9 ai_OZ_start/ai_DZ_start only when after_icing
  A10 exposure: seconds>0, seconds <= exposure_duration_w (+1s slack)
  A11 strength mix: what share is 5v5 (trainer filter must exist)

Usage: python scripts/25_surface_deep_audit.py [--season 20232024] [--n 200000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")
FAIL = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=20232024)
    ap.add_argument("--n", type=int, default=200000)
    args = ap.parse_args()
    df = pd.read_parquet(ART / "trainer_surface" / f"season={args.season}.parquet")
    if args.n and len(df) > args.n:
        df = df.sample(args.n, random_state=0).reset_index(drop=True)
    print(f"auditing {len(df):,} rows of season {args.season}\n")

    # A1 — the identity-leak check
    focal_in_with = focal_in_vs = 0
    for pid, w, v in zip(df.playerId, df.with_ids, df.vs_ids):
        if w is not None and pid in set(int(x) for x in w):
            focal_in_with += 1
        if v is not None and pid in set(int(x) for x in v):
            focal_in_vs += 1
    check("A1 focal not in own with_ids", focal_in_with == 0,
          f"{focal_in_with} violations")
    check("A1 focal not in vs_ids", focal_in_vs == 0, f"{focal_in_vs} violations")

    # A2 — alignment
    bad_len = 0
    feats = ["with_seconds", "with_form", "with_gsc", "with_age", "with_hand",
             "with_pos", "with_form_n"]
    for i in range(len(df)):
        L = len(df.with_ids.iat[i])
        if any(len(df[c].iat[i]) != L for c in feats
               if df[c].iat[i] is not None):
            bad_len += 1
    check("A2 member-list alignment (with side)", bad_len == 0,
          f"{bad_len} rows misaligned")

    # A3 — padding convention
    ids = np.concatenate([np.asarray(x, dtype=float) for x in df.with_ids])
    secs = np.concatenate([np.asarray(x, dtype=float) for x in df.with_seconds])
    pad_mismatch = int((((ids == 0) & (secs > 0)) | ((ids > 0) & (secs <= 0))).sum())
    check("A3 padding ids==0 <=> seconds<=0", pad_mismatch == 0,
          f"{pad_mismatch} slots mismatched")

    # A4 — dup members
    dups = sum(1 for w in df.with_ids
               if (lambda r: len(r) != len(set(r)))([x for x in w if x > 0]))
    check("A4 no duplicate real members", dups == 0, f"{dups} rows with dups")

    # A5 — with/vs disjoint
    overlap = sum(1 for w, v in zip(df.with_ids, df.vs_ids)
                  if set(int(x) for x in w if x > 0)
                  & set(int(x) for x in v if x > 0))
    check("A5 with ∩ vs empty", overlap == 0, f"{overlap} rows overlap")

    # A6 — coach join consistency
    cg = df.groupby(["gamePk", "teamId"]).coach_id.nunique()
    check("A6 one coach per (game, team)", int((cg > 1).sum()) == 0,
          f"{int((cg > 1).sum())} violations")
    per_game = df.groupby("gamePk").coach_id.nunique()
    two = float((per_game == 2).mean())
    check("A6 two distinct coaches per game (>=95%)", two >= 0.95, f"{two:.1%}")

    # A7 — schedule sanity
    rest_ok = df.rest_days_team.between(1, 7).mean()
    game_lvl = df.drop_duplicates(["gamePk", "teamId"])
    b2b = float(game_lvl.b2b_team.mean())
    check("A7 rest_days in [1,7]", rest_ok > 0.999, f"{rest_ok:.2%}")
    check("A7 b2b rate sane", 0.08 <= b2b <= 0.25, f"{b2b:.1%} of team-games")

    # A8 — scales
    forms = np.concatenate([np.asarray(x, dtype=float) for x in df.with_form])
    forms = forms[~np.isnan(forms)]
    ages = np.concatenate([np.asarray(x, dtype=float) for x in df.with_age])
    ages = ages[~np.isnan(ages)]
    if len(forms):
        check("A8 form scale (median 1.8-3.2 xGF60)",
              1.8 <= float(np.median(forms)) <= 3.2,
              f"median {np.median(forms):.2f}, p5-p95 "
              f"[{np.percentile(forms,5):.2f},{np.percentile(forms,95):.2f}]")
    else:
        print("[INFO] A8 form: no values this season (expected for the earliest "
              "surface season — UNK shell mode until prior season is built)")
    check("A8 ages in [17,48]",
          bool((ages.min() >= 17) and (ages.max() <= 48)),
          f"range [{ages.min():.1f},{ages.max():.1f}]")

    # A9 — icing derivation
    bad_ai = int(((df.ai_OZ_start + df.ai_DZ_start) > 0)
                 [~df.after_icing.astype(bool)].sum())
    check("A9 ai_* only when after_icing", bad_ai == 0, f"{bad_ai} violations")

    # A10 — exposure
    check("A10 seconds>0", bool((df.seconds > 0).all()))
    over = float((df.seconds > df.exposure_duration_w + 1).mean())
    check("A10 seconds <= window duration", over == 0.0, f"{over:.3%} over")

    # A11 — strength mix (informational + trainer must filter)
    mix = df.strength_global.astype(str).value_counts(normalize=True)
    print(f"[INFO] strength mix: {dict(mix.round(3))}")

    print(f"\n{'ALL CLEAR' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
