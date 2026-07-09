#!/usr/bin/env python3

import sys
import os
import glob
import pandas as pd
import numpy as np


SHRINK_K_EVENTS = 5.0   # pseudo-events for shrinking toward league mean
EPS = 1e-6


def _read_csvs_any(path_like: str) -> pd.DataFrame:
    """
    Read a single CSV, a directory of CSVs, or a glob pattern into one DataFrame.
    """
    if os.path.isdir(path_like):
        files = sorted(glob.glob(os.path.join(path_like, "*.csv")))
    elif any(ch in path_like for ch in ["*", "?", "["]):
        files = sorted(glob.glob(path_like))
    else:
        files = [path_like]

    frames = []
    for f in files:
        if os.path.isfile(f):
            try:
                frames.append(pd.read_csv(f))
            except Exception as e:
                print(f"[WARN] failed reading {f}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_ev_toi(ev_path: str) -> pd.DataFrame:
    """
    Load EV TOI with expected columns: gameid, teamid, playerId, season,
    and either ev_minutes or ev_seconds.
    """
    ev = _read_csvs_any(ev_path)

    required = {"gameid", "teamid", "playerId", "season"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"EV TOI file is missing required columns: {missing}")

    # derive ev_minutes
    if "ev_minutes" in ev.columns:
        ev["ev_minutes"] = ev["ev_minutes"].astype(float)
    elif "ev_seconds" in ev.columns:
        ev["ev_minutes"] = ev["ev_seconds"].astype(float) / 60.0
    else:
        raise ValueError("EV TOI file must have either 'ev_minutes' or 'ev_seconds'")

    return ev[["gameid", "teamid", "playerId", "season", "ev_minutes"]]


def load_penalties(pen_path: str) -> pd.DataFrame:
    """
    Load per-game penalty counts from penalties_rollup, and construct
    'effective' strength-changing penalties at EV:

      pen_effective_taken = minor + major (non-coincidental)
      pen_effective_drawn = minor + major (non-coincidental)
    """
    pen = _read_csvs_any(pen_path)
    if pen.empty:
        return pen

    # Make sure columns exist (they should, from your rollup)
    needed_cols = [
        "gameid", "teamid", "playerId",
        "pen_minor_taken", "pen_minor_drawn",
        "pen_major_taken", "pen_major_drawn",
        "coincidental_flag",
    ]
    for c in needed_cols:
        if c not in pen.columns:
            pen[c] = 0

    # effective strength-changing penalties = minors + majors
    pen["pen_effective_taken"] = pen["pen_minor_taken"] + pen["pen_major_taken"]
    pen["pen_effective_drawn"] = pen["pen_minor_drawn"] + pen["pen_major_drawn"]

    # Zero out coincidentals for "effective" counts so this reflects
    # actual special-teams swings, not offsetting penalties.
    mask_clean = (pen["coincidental_flag"] == 0)
    pen.loc[~mask_clean, ["pen_effective_taken", "pen_effective_drawn"]] = 0

    return pen


def _apply_shrink_and_z(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Given grouped penalty rates with per-60 raw rates and pen_neff_events,
    shrink toward league (per-season) mean and compute z-scores + log_neff_pen.

    Expects columns:
      season,
      p_prior_pen_taken60_ev,
      p_prior_pen_drawn60_ev,
      pen_neff_events
    """
    # Compute per-season league means/stds for raw per-60 rates
    # Only use rows with actual EV minutes for stats
    stats = (
        grp[grp["ev_minutes_total"] > 0]
        .groupby("season")
        .agg(
            taken_mean=("p_prior_pen_taken60_ev", "mean"),
            taken_std=("p_prior_pen_taken60_ev", "std"),
            drawn_mean=("p_prior_pen_drawn60_ev", "mean"),
            drawn_std=("p_prior_pen_drawn60_ev", "std"),
        )
        .reset_index()
    )

    grp = grp.merge(stats, on="season", how="left")

    # Fallback if std is 0 or NaN
    grp["taken_std"] = grp["taken_std"].fillna(0.0)
    grp["drawn_std"] = grp["drawn_std"].fillna(0.0)

    # Shrink toward league means using pen_neff_events
    neff = grp["pen_neff_events"].clip(lower=0.0)
    w = neff / (neff + SHRINK_K_EVENTS)

    grp["pen_taken60_shrunk"] = (
        w * grp["p_prior_pen_taken60_ev"] + (1.0 - w) * grp["taken_mean"]
    )
    grp["pen_drawn60_shrunk"] = (
        w * grp["p_prior_pen_drawn60_ev"] + (1.0 - w) * grp["drawn_mean"]
    )

    # Z-scores of shrunk rates
    grp["prior_pen_taken60_z"] = (
        (grp["pen_taken60_shrunk"] - grp["taken_mean"]) / (grp["taken_std"] + EPS)
    )
    grp["prior_pen_drawn60_z"] = (
        (grp["pen_drawn60_shrunk"] - grp["drawn_mean"]) / (grp["drawn_std"] + EPS)
    )

    # Log n_eff for Phase A
    grp["log_neff_pen"] = np.log1p(neff)

    return grp


def build_penalty_priors(pen: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-(season, playerId) penalty priors from per-game penalties + EV TOI.

    Outputs:
      - raw per-60 rates
      - n_eff proxies
      - shrunk per-60
      - z-scored shrunk rates
      - log_neff_pen
      - evidence flag
    """
    if ev.empty:
        return pd.DataFrame(
            columns=[
                "season", "playerId",
                "p_prior_pen_taken60_ev",
                "p_prior_pen_drawn60_ev",
                "p_prior_pen_delta60_ev",
                "pen_effective_taken_total",
                "pen_effective_drawn_total",
                "pen_events_total",
                "pen_neff_events",
                "pen_neff_minutes",
                "ev_minutes_total",
                "ev_games_with_toi",
                "prior_pen_taken60_z",
                "prior_pen_drawn60_z",
                "log_neff_pen",
                "evidence_pen_flag",
            ]
        )

    # merge penalties onto EV TOI (we want all EV TOI rows, even if 0 penalties)
    if pen.empty:
        merged = ev.copy()
        merged["pen_effective_taken"] = 0.0
        merged["pen_effective_drawn"] = 0.0
        merged["coincidental_flag"] = 0
    else:
        merged = ev.merge(
            pen[
                [
                    "gameid",
                    "teamid",
                    "playerId",
                    "pen_effective_taken",
                    "pen_effective_drawn",
                    "coincidental_flag",
                ]
            ],
            on=["gameid", "teamid", "playerId"],
            how="left",
        )
        merged["pen_effective_taken"] = merged["pen_effective_taken"].fillna(0).astype(float)
        merged["pen_effective_drawn"] = merged["pen_effective_drawn"].fillna(0).astype(float)
        merged["coincidental_flag"] = merged["coincidental_flag"].fillna(0).astype(int)

    # aggregate to (season, playerId)
    grp = (
        merged
        .groupby(["season", "playerId"], dropna=False)
        .agg(
            ev_minutes_total=("ev_minutes", "sum"),
            pen_effective_taken_total=("pen_effective_taken", "sum"),
            pen_effective_drawn_total=("pen_effective_drawn", "sum"),
            ev_games_with_toi=("gameid", pd.Series.nunique),
        )
        .reset_index()
    )

    grp["pen_events_total"] = (
        grp["pen_effective_taken_total"] + grp["pen_effective_drawn_total"]
    )

    # n_eff proxies for later shrink/log_neff_pen
    grp["pen_neff_events"] = grp["pen_events_total"]
    grp["pen_neff_minutes"] = grp["ev_minutes_total"] / 60.0

    # init raw per-60 priors
    grp["p_prior_pen_taken60_ev"] = 0.0
    grp["p_prior_pen_drawn60_ev"] = 0.0

    has_ev = grp["ev_minutes_total"] > 0
    grp.loc[has_ev, "p_prior_pen_taken60_ev"] = (
        60.0 * grp.loc[has_ev, "pen_effective_taken_total"] / grp.loc[has_ev, "ev_minutes_total"]
    )
    grp.loc[has_ev, "p_prior_pen_drawn60_ev"] = (
        60.0 * grp.loc[has_ev, "pen_effective_drawn_total"] / grp.loc[has_ev, "ev_minutes_total"]
    )
    grp["p_prior_pen_delta60_ev"] = (
        grp["p_prior_pen_drawn60_ev"] - grp["p_prior_pen_taken60_ev"]
    )

    # simple evidence flag
    def evidence_flag(row):
        if row["ev_minutes_total"] <= 0:
            return "NO_EV"
        if row["pen_events_total"] >= 3 and row["ev_minutes_total"] >= 300:
            return "OK"
        elif row["ev_minutes_total"] >= 300 and row["pen_events_total"] == 0:
            return "CLEAN_BUT_RARE"
        else:
            return "LOW_SAMPLE"

    grp["evidence_pen_flag"] = grp.apply(evidence_flag, axis=1)

    # shrink + z + log_neff_pen
    grp = _apply_shrink_and_z(grp)

    out_cols = [
        "season",
        "playerId",
        # raw per-60
        "p_prior_pen_taken60_ev",
        "p_prior_pen_drawn60_ev",
        "p_prior_pen_delta60_ev",
        # totals / evidence
        "pen_effective_taken_total",
        "pen_effective_drawn_total",
        "pen_events_total",
        "pen_neff_events",
        "pen_neff_minutes",
        "ev_minutes_total",
        "ev_games_with_toi",
        # shrunk + z + log
        "pen_taken60_shrunk",
        "pen_drawn60_shrunk",
        "prior_pen_taken60_z",
        "prior_pen_drawn60_z",
        "log_neff_pen",
        "evidence_pen_flag",
    ]
    return grp[out_cols].sort_values(["season", "playerId"]).reset_index(drop=True)


def build_priors_from_rollup(per_game_path: str, out_path: str):
    """
    Simple mode: take a per-game penalties CSV that already includes ev_minutes,
    aggregate to per-(season, playerId) priors, shrink + z + log, and write to CSV.
    """
    df = pd.read_csv(per_game_path)

    if "ev_minutes" not in df.columns:
        print("[ERROR] ev_minutes not found in per-game file; run penalties_rollup.py with --wins_dir or --ev.")
        sys.exit(1)

    for c in ["pen_minor_taken", "pen_major_taken", "pen_minor_drawn", "pen_major_drawn", "coincidental_flag"]:
        if c not in df.columns:
            df[c] = 0

    df["pen_effective_taken"] = df["pen_minor_taken"] + df["pen_major_taken"]
    df["pen_effective_drawn"] = df["pen_minor_drawn"] + df["pen_major_drawn"]

    # Zero out coincidentals for "effective" counts here as well
    mask_clean = (df["coincidental_flag"] == 0)
    df.loc[~mask_clean, ["pen_effective_taken", "pen_effective_drawn"]] = 0

    if "season" not in df.columns:
        df["season"] = ""

    grp = (
        df.groupby(["season", "playerId"], dropna=False)
          .agg(
              ev_minutes_total=("ev_minutes", "sum"),
              pen_effective_taken_total=("pen_effective_taken", "sum"),
              pen_effective_drawn_total=("pen_effective_drawn", "sum"),
              ev_games_with_toi=("gameid", pd.Series.nunique),
          )
          .reset_index()
    )

    grp["pen_events_total"] = grp["pen_effective_taken_total"] + grp["pen_effective_drawn_total"]

    # n_eff proxies
    grp["pen_neff_events"] = grp["pen_events_total"]
    grp["pen_neff_minutes"] = grp["ev_minutes_total"] / 60.0

    grp["p_prior_pen_taken60_ev"] = 0.0
    grp["p_prior_pen_drawn60_ev"] = 0.0

    has_ev = grp["ev_minutes_total"] > 0
    grp.loc[has_ev, "p_prior_pen_taken60_ev"] = (
        60.0 * grp.loc[has_ev, "pen_effective_taken_total"] / grp.loc[has_ev, "ev_minutes_total"]
    )
    grp.loc[has_ev, "p_prior_pen_drawn60_ev"] = (
        60.0 * grp.loc[has_ev, "pen_effective_drawn_total"] / grp.loc[has_ev, "ev_minutes_total"]
    )

    grp["p_prior_pen_delta60_ev"] = grp["p_prior_pen_drawn60_ev"] - grp["p_prior_pen_taken60_ev"]

    def evidence_flag(row):
        if row["ev_minutes_total"] <= 0:
            return "NO_EV"
        if row["pen_events_total"] >= 3 and row["ev_minutes_total"] >= 300:
            return "OK"
        elif row["ev_minutes_total"] >= 300 and row["pen_events_total"] == 0:
            return "CLEAN_BUT_RARE"
        else:
            return "LOW_SAMPLE"

    grp["evidence_pen_flag"] = grp.apply(evidence_flag, axis=1)

    # shrink + z + log_neff_pen
    grp = _apply_shrink_and_z(grp)

    cols = [
        "season",
        "playerId",
        # raw per-60
        "p_prior_pen_taken60_ev",
        "p_prior_pen_drawn60_ev",
        "p_prior_pen_delta60_ev",
        # totals / evidence
        "pen_effective_taken_total",
        "pen_effective_drawn_total",
        "pen_events_total",
        "pen_neff_events",
        "pen_neff_minutes",
        "ev_minutes_total",
        "ev_games_with_toi",
        # shrunk + z + log
        "pen_taken60_shrunk",
        "pen_drawn60_shrunk",
        "prior_pen_taken60_z",
        "prior_pen_drawn60_z",
        "log_neff_pen",
        "evidence_pen_flag",
    ]
    grp[cols].to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(grp)} rows")


def main(pen_path: str, ev_path: str, out_path: str):
    """
    Legacy mode: penalties CSV/dir/glob + EV TOI CSV/dir/glob.
    """
    pen = load_penalties(pen_path)
    ev = load_ev_toi(ev_path)
    priors = build_penalty_priors(pen, ev)
    priors.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(priors)} rows")


if __name__ == "__main__":
    # Support two modes:
    # 1) Simple (recommended):
    #       python build_penalty_priors.py <per_game_penalties_with_evtoi.csv> <out_priors.csv>
    # 2) Legacy:
    #       python build_penalty_priors.py <penalties_csv|dir|glob> <ev_toi_csv|dir|glob> <out_csv>
    if len(sys.argv) == 3:
        build_priors_from_rollup(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(
            "Usage:\n"
            "  python build_penalty_priors.py <per_game_penalties_with_evtoi.csv> <out_priors.csv>\n"
            "  or\n"
            "  python build_penalty_priors.py <penalties_csv|dir|glob> <ev_toi_csv|dir|glob> <out_csv>"
        )
        sys.exit(1)
