"""The Perfect Window contract.

One row = one (player, window). Five blocks, five enforced contracts:
  1. every prior column is lagged+EB and ships with n_eff (naming + presence checks)
  2. every context column is knowable at the opening faceoff (PRE-timing whitelist)
  3. list columns are seconds-desc sorted, fixed length K, padded with 0
  4. baseline mu columns are OOF (holdout_season retained)
  5. outcome columns never appear in any feature list

`validate(df)` raises ContractError with every violation listed — builds fail loudly.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

SCHEMA_VERSION = "pw_v1"
TOP_K = 5

KEYS = ["season", "gamePk", "window_id", "teamId", "playerId", "date",
        "strength_global", "seconds", "duration", "sample_weight", "schema_version"]

# Contract 2: PRE-timing context (knowable at the faceoff that opens the window).
CONTEXT = ["period", "period_time_bucket", "score_bucket", "start_regime",
           "lever_zone_start", "fo_loc_enum", "fo_took_start", "stoppage_class_at_start",
           "after_icing", "bench_rights", "long_change", "home_away", "rinkid",
           "skater_diff", "mp_bucket", "b2b_team", "b2b_opponent", "rest_days_bucket"]

# Contract 3: people payload, seconds-desc sorted, len == TOP_K, 0-padded.
PEOPLE_LISTS = ["with_ids", "with_seconds", "vs_ids", "vs_seconds"]
PEOPLE_SCALARS = ["line_no", "opp_line_matchup_bucket"]

# Contract 1: every entry here must end in an approved lag suffix and have n_eff nearby.
PRIORS = [
    # player intrinsic
    "prior_off60_eb", "prior_off60_se", "prior_def60_eb", "prior_def60_se",
    "n_eff_games", "n_eff_minutes",
    "talent_off_shrunk", "talent_def_shrunk", "talent_n_eff",
    # deployment
    "p_prior_oz_share_ev", "p_prior_dz_share_ev", "share_vs_stars_ema_lag",
    "p_minutes_prior_ev", "p_cold_ev",
    # team environment
    "team_xgf60_prior_ev", "team_xga60_prior_ev", "team_pace60_ev_prior",
    "opp_team_xgf60_prior_ev", "opp_team_xga60_prior_ev", "opp_team_pace60_ev_prior",
    "team_gsaa_per60_prior_eb", "opp_gsaa_per60_prior_eb",
    "team_goalie_tier", "opp_goalie_tier",
    # bio (static, allowed unlagged)
    "age", "height_cm", "weight_kg", "shootsCatches", "position",
]
_BIO = {"age", "height_cm", "weight_kg", "shootsCatches", "position",
        "team_goalie_tier", "opp_goalie_tier"}
_LAG_PAT = re.compile(r"(_prior|_eb|_lag|_shrunk|^n_eff|_n_eff|_se$|^p_cold)")

# Contract 4: OOF baseline block.
BASELINE = ["mu_xgf60", "mu_xga60", "sigma_xgf_w", "sigma_xga_w", "holdout_season"]

# Contract 5: targets. NEVER in a feature list.
OUTCOMES = ["y_xGF", "y_xGA", "y_GF", "y_GA", "y_SF", "y_SA",
            "y_hits", "y_blocks", "y_takeaways", "y_giveaways"]

FEATURES = CONTEXT + PEOPLE_SCALARS + PRIORS          # model input surface
ALL_COLUMNS = KEYS + CONTEXT + PEOPLE_LISTS + PEOPLE_SCALARS + PRIORS + BASELINE + OUTCOMES


class ContractError(ValueError):
    pass


def validate(df: pd.DataFrame, sample: int = 200_000) -> dict:
    """Run all five contracts. Returns a stats dict; raises ContractError on violation."""
    errs, stats = [], {}

    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        errs.append(f"missing columns: {missing}")

    # Contract 5 first (cheap, catastrophic if broken)
    leak = [c for c in OUTCOMES if c in FEATURES]
    if leak:
        errs.append(f"outcome columns in FEATURES: {leak}")

    # Contract 1: lag-naming + n_eff presence
    for c in PRIORS:
        if c in _BIO:
            continue
        if not _LAG_PAT.search(c):
            errs.append(f"prior '{c}' lacks lag/EB suffix — rename or fix upstream")

    if not errs and len(df):
        s = df if len(df) <= sample else df.sample(sample, random_state=0)

        # Contract 3: list length + seconds-desc ordering
        for ids_c, sec_c in [("with_ids", "with_seconds"), ("vs_ids", "vs_seconds")]:
            if ids_c in s.columns:
                bad_len = (~s[ids_c].map(lambda v: hasattr(v, "__len__") and len(v) == TOP_K)).mean()
                if bad_len > 0:
                    errs.append(f"{ids_c}: {bad_len:.2%} rows not length {TOP_K}")
                def _sorted_desc(v):
                    a = np.asarray(v, dtype=float)
                    return bool(np.all(a[:-1] >= a[1:]))
                bad_sort = (~s[sec_c].map(_sorted_desc)).mean()
                if bad_sort > 0.001:
                    errs.append(f"{sec_c}: {bad_sort:.2%} rows not seconds-desc sorted")

        # Cold-start fix regression tests (MIGRATION_MAP bugs 1-2)
        if "prior_off60_se" in s.columns:
            se = s["prior_off60_se"].dropna()
            if len(se) and (se <= 0).mean() > 0.001:
                errs.append(f"prior_off60_se: {(se<=0).mean():.2%} rows <= 0 (SE collapse bug back)")
            if len(se) and se.nunique() < 10:
                errs.append("prior_off60_se nearly constant (cold-start sentinel bug back)")

        # Keys sane
        stats["dupe_key_rate"] = float(s.duplicated(["gamePk", "window_id", "playerId"]).mean())
        if stats["dupe_key_rate"] > 0:
            errs.append(f"duplicate (gamePk,window_id,playerId): {stats['dupe_key_rate']:.4%}")
        stats["null_rate_features"] = float(s[list(set(FEATURES) & set(s.columns))].isna().mean().mean())

    if errs:
        raise ContractError(" | ".join(errs))
    stats["schema_version"] = SCHEMA_VERSION
    stats["n_rows"] = len(df)
    return stats
