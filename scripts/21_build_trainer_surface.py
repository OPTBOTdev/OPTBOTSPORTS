"""Script 21 — THE TRAINER SURFACE: one audited table the GPU trainer reads.

Assembles, per (window, focal player):
  CONTEXT   all SYNTH_CONTEXT cols + derived ai_OZ_start / ai_DZ_start
  SCHEDULE  rest_days_team + b2b_team (computed from game dates) ·
            games_since_team_change · coach_id + coach_tenure_games (joined)
  PRIORS    the 10 team/goalie prior cols (pass-through, already lag-audited)
  PEOPLE    with_ids/with_seconds/vs_ids/vs_seconds (order preserved) PLUS
            aligned member-feature lists: form (season-lagged, F-1 safe),
            gsc, age-at-game, hand, position — the dynamic shells
  TALENT    focal raw_off/raw_def/talent_n_eff + baseline mu (trunk channel;
            NOT tower input)
  TARGETS   MP primary (mp_xgf, mp_xga, rush/rebound/chaos splits) ·
            in-house aux (y_xGF/y_xGA) · counts (GF/GA/SF/SA) ·
            micro (hits/blocks/take/give — trunk-only heads, T19)
  WEIGHTS   seconds (exposure) · is_multistint (downweight flag)

Output: artifacts/trainer_surface/season=<S>.parquet (one part per season)
        artifacts/trainer_surface_manifest.json (coverage + sha256 + counts)

Legality: assert_legal() runs on the scalar feature list; duration is carried
ONLY under the name `exposure_duration_w` (loss weighting — never an input;
stop-bias +51% measured).

Usage: python scripts/21_build_trainer_surface.py [--seasons 20232024 ...]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.contracts import tower_schema as ts  # noqa: E402

ART = Path(r"D:\optbot\artifacts")
OUT = ART / "trainer_surface"

CONTEXT = [c for c in ts.SYNTH_CONTEXT if c not in ("ai_OZ_start", "ai_DZ_start")]
PRIORS = ts.SYNTH_TEAM_PRIORS
MARG_PRESENT = ["seconds", "shift_count_in_window", "time_since_last_shift_s",
                "onice_elapsed_at_window_start", "entered_after_start",
                "entry_offset_s", "is_multistint"]
TALENT = ["raw_off", "raw_def", "talent_n_eff", "mu_xgf60", "mu_xga60"]
TARGETS = ["mp_xgf", "mp_xga", "mp_xgf_rush", "mp_xgf_rebound", "mp_xreb_gen",
           "mp_xozcont", "y_xGF", "y_xGA", "y_GF", "y_GA", "y_SF", "y_SA",
           "y_hits", "y_blocks", "y_takeaways", "y_giveaways"]
KEYS = ["season", "gamePk", "window_id", "teamId", "playerId", "date_x",
        "strength_global", "line_no", "games_since_team_change"]


def schedule_table(w_keys: pd.DataFrame) -> pd.DataFrame:
    """rest_days_team + b2b_team from each team's own game-date sequence."""
    g = (w_keys[["gamePk", "teamId", "date_x"]].drop_duplicates()
         .sort_values(["teamId", "date_x"]))
    g["date_x"] = pd.to_datetime(g.date_x)
    g["rest_days_team"] = (g.groupby("teamId").date_x.diff().dt.days
                           .clip(upper=7).fillna(7))
    g["b2b_team"] = (g.rest_days_team == 1).astype(int)
    return g[["gamePk", "teamId", "rest_days_team", "b2b_team"]]


def member_lists(df: pd.DataFrame, side: str, shells, gsc, bio) -> pd.DataFrame:
    """Explode with_/vs_ id lists, join shells+bio, regroup to aligned lists."""
    ids_col = f"{side}_ids"
    e = df[["season", "gamePk", "date_x", ids_col]].copy()
    e["__row"] = np.arange(len(e))
    e = e.explode(ids_col).rename(columns={ids_col: "mid"})
    e = e[e.mid.notna()]
    e["mid"] = e.mid.astype("int64")
    e = e.merge(shells, left_on=["mid", "season"],
                right_on=["playerId", "season"], how="left")
    e = e.merge(gsc, left_on=["mid", "gamePk"],
                right_on=["playerId", "gamePk"], how="left")
    e = e.merge(bio, left_on="mid", right_on="playerId", how="left")
    e["age"] = ((pd.to_datetime(e.date_x) - pd.to_datetime(e.birthDate))
                .dt.days / 365.25)
    agg = e.groupby("__row").agg(
        form=("form_prior_xgf60", list), form_n=("form_neff_seasons", list),
        gsc=("member_gsc", list), age=("age", list),
        hand=("shootsCatches", list), pos=("position", list))
    agg.columns = [f"{side}_{c}" for c in agg.columns]
    out = pd.DataFrame(index=np.arange(len(df)))
    return out.join(agg).set_index(df.index)


def build_season(season: int, shells, gsc, coach, bio) -> pd.DataFrame:
    cols = KEYS + CONTEXT + PRIORS + MARG_PRESENT + TALENT + TARGETS \
        + ["duration", "with_ids", "with_seconds", "vs_ids", "vs_seconds"]
    df = pq.read_table(ART / "perfect_windows_v3.parquet", columns=cols,
                       filters=[("season", "=", season)]).to_pandas()
    if df.empty:
        return df
    # icing-context split the contract wants (after_icing x zone start)
    oz = df.lever_zone_start.astype(str).str.upper().str.startswith("O")
    dz = df.lever_zone_start.astype(str).str.upper().str.startswith("D")
    df["ai_OZ_start"] = (df.after_icing.astype(bool) & oz).astype(int)
    df["ai_DZ_start"] = (df.after_icing.astype(bool) & dz).astype(int)
    df = df.merge(schedule_table(df), on=["gamePk", "teamId"], how="left")
    df = df.merge(coach, on=["gamePk", "teamId"], how="left")
    df["coach_id"] = df.coach_id.fillna("UNK")
    df["coach_tenure_games"] = df.coach_tenure_games.fillna(0)
    for side in ("with", "vs"):
        df = pd.concat([df, member_lists(df, side, shells, gsc, bio)], axis=1)
    # duration may exist only as a loss weight, never as an input
    df = df.rename(columns={"duration": "exposure_duration_w"})
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    args = ap.parse_args()
    # legality gate BEFORE any work (ai_* derived cols are in the contract)
    scalar_inputs = (ts.SYNTH_CONTEXT + ts.SYNTH_SCHEDULE + ts.SYNTH_TEAM_PRIORS
                     + [m for m in ts.MARG if m != "duration"])
    ts.assert_legal([c for c in scalar_inputs if c != "last_shift_len_s"])

    shells_all = pd.read_parquet(ART / "member_shells.parquet")
    shells = shells_all[["playerId", "season", "form_prior_xgf60",
                         "form_neff_seasons"]].drop_duplicates(["playerId", "season"])
    gsc = shells_all[["playerId", "gamePk", "member_gsc"]].drop_duplicates(
        ["playerId", "gamePk"])
    coach = pd.read_parquet(ART / "coach_table.parquet")
    bio = pd.read_parquet(ART / "player_bio.parquet")[
        ["playerId", "birthDate", "shootsCatches", "position"]]

    seasons = args.seasons or sorted(
        pq.read_table(ART / "perfect_windows_v3.parquet",
                      columns=["season"]).to_pandas().season.unique())
    OUT.mkdir(exist_ok=True)
    manifest = {"schema_version": "surface_v1", "seasons": {}, "coverage": {}}
    for s in seasons:
        df = build_season(int(s), shells, gsc, coach, bio)
        if df.empty:
            print(f"{s}: EMPTY — skipped")
            continue
        fp = OUT / f"season={s}.parquet"
        df.to_parquet(fp, index=False)
        cov = {c: round(float(df[c].notna().mean()), 4)
               for c in df.columns if not c.startswith(("with_", "vs_"))}
        # member-list coverage over REAL members only (id>0; zeros are padding
        # the loader masks via seconds<=0 — counting them fakes missingness)
        for side in ("with", "vs"):
            ids_flat = np.array([x for lst in df[f"{side}_ids"]
                                 for x in (lst if lst is not None else [])],
                                dtype=float)
            real = ids_flat > 0
            for feat in ("form", "age", "hand"):
                flat = np.array([v for lst in df[f"{side}_{feat}"]
                                 for v in (lst if isinstance(lst, (list, np.ndarray))
                                           else [])], dtype=object)
                if len(flat) != len(ids_flat) or not real.any():
                    cov[f"{side}_{feat}"] = 0.0
                    continue
                miss = pd.isna(pd.Series(flat[real]).astype(object)).mean()
                cov[f"{side}_{feat}"] = round(1 - float(miss), 4)
        manifest["seasons"][str(s)] = {"rows": len(df),
                                       "sha256": hashlib.sha256(
                                           fp.read_bytes()).hexdigest()[:16]}
        manifest["coverage"][str(s)] = cov
        print(f"{s}: {len(df):,} rows -> {fp.name}")
    manifest["columns"] = sorted(df.columns.tolist())
    (ART / "trainer_surface_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print("manifest written: trainer_surface_manifest.json")
