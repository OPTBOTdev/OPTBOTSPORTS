"""Dataset QA suite — run against the REAL source parquets before anything builds.

Checks the four source families the MVP consumes:
  A. window spine        (combined_player_windows_2017_2024.parquet)
  B. player-game obs     (phaseC_player_game_observations.parquet)   <- talent prior input
  C. player windows+mu   (player_windows_with_baseline_*.parquet)    <- scenario/projection input
  D. OOF team windows    (baseline_oof_team_windows.parquet)

Each check returns (name, PASS/WARN/FAIL, detail). FAILs block the pipeline.
"""
from __future__ import annotations
import glob
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _res(name, status, detail):
    return {"check": name, "status": status, "detail": detail}


def check_window_spine(path: str) -> list[dict]:
    out = []
    f = pq.ParquetFile(path)
    out.append(_res("spine.exists", PASS, f"{f.metadata.num_rows:,} rows x {f.metadata.num_columns} cols"))
    cols = ["season", "gamePk", "window_id", "teamId", "playerId", "date",
            "strength_global", "seconds", "duration"]
    df = pd.read_parquet(path, columns=cols)
    dup = df.duplicated(["gamePk", "window_id", "playerId", "teamId"]).mean()
    out.append(_res("spine.dupes", PASS if dup == 0 else FAIL, f"{dup:.5%} duplicate keys"))
    bad_expo = ((df["seconds"] <= 0) | (df["seconds"] > df["duration"] + 1)).mean()
    out.append(_res("spine.exposure", PASS if bad_expo < 0.001 else FAIL,
                    f"{bad_expo:.4%} rows seconds<=0 or seconds>duration"))
    per_season = df.groupby("season")["gamePk"].nunique()
    short = per_season[per_season < 1000]  # covid seasons are legitimately short
    out.append(_res("spine.season_coverage",
                    PASS if len(short) <= 2 else WARN,
                    f"games/season: {per_season.to_dict()}"))
    null_dates = df["date"].isna().mean()
    out.append(_res("spine.dates", PASS if null_dates == 0 else FAIL,
                    f"{null_dates:.4%} null dates (freeze protocol depends on this column)"))
    return out


def check_player_game_obs(path: str) -> list[dict]:
    out = []
    df = pd.read_parquet(path)
    out.append(_res("obs.exists", PASS, f"{len(df):,} rows"))
    need = {"playerId", "season", "gamePk", "date", "strength_global", "toi_sec",
            "resid_xgf60_game", "resid_xga60_game", "n_eff_toi_games"}
    miss = need - set(df.columns)
    out.append(_res("obs.columns", PASS if not miss else FAIL, f"missing: {sorted(miss)}" if miss else "all present"))

    ev = df[df.strength_global == "5v5"]
    # residuals must be ~centered per season (OOF discipline says mean ~ 0)
    m = ev.groupby("season")["resid_xgf60_game"].mean()
    worst = m.abs().max()
    out.append(_res("obs.residual_centering", PASS if worst < 0.25 else WARN,
                    f"per-season mean resid_xgf60 max |.|={worst:.3f}  ({m.round(3).to_dict()})"))
    # n_eff sanity: decayed n_eff legitimately plateaus (add ~1 game, decay the stack),
    # so small dips are normal. Only LARGE drops (>25% of level) indicate a rebuild bug.
    srt = ev.sort_values(["playerId", "season", "date"])
    g = srt.groupby(["playerId", "season"])["n_eff_toi_games"]
    rel_drop = (g.diff() / g.shift(1).clip(lower=1e-6)).dropna()
    frac_big_drop = (rel_drop < -0.25).mean()
    out.append(_res("obs.n_eff_sane", PASS if frac_big_drop < 0.005 else FAIL,
                    f"{frac_big_drop:.3%} of within-player steps drop >25% (plateau dips are normal)"))
    # actuals-present canary: the Jan-2026 file shipped with y == 0 everywhere.
    y_zero = float((ev["y_xgf_onice_w"] == 0).mean()) if "y_xgf_onice_w" in ev.columns else 1.0
    out.append(_res("obs.actuals_present", PASS if y_zero < 0.60 else FAIL,
                    f"{y_zero:.1%} of rows have y_xgf_onice_w == 0 "
                    "(>60% means the actuals join failed; residuals are just -mu)"))
    # residual scale sanity (xg60 residuals should live in single digits)
    p99 = ev["resid_xgf60_game"].abs().quantile(0.99)
    out.append(_res("obs.residual_scale", PASS if p99 < 15 else FAIL, f"|resid_xgf60| p99 = {p99:.2f}"))
    nan_rate = ev[["resid_xgf60_game", "resid_xga60_game"]].isna().mean().mean()
    out.append(_res("obs.nans", PASS if nan_rate < 0.005 else WARN, f"{nan_rate:.3%} NaN residuals (5v5)"))
    return out


def check_windows_with_baseline(glob_pat: str) -> list[dict]:
    out = []
    files = sorted(glob.glob(glob_pat))
    per_season = [f for f in files if any(ch.isdigit() for ch in f.rsplit("_", 1)[-1])]
    out.append(_res("mu.files", PASS if len(per_season) >= 8 else WARN,
                    f"{len(per_season)} per-season files"))
    for fp in per_season[-2:]:  # deepest checks on the two newest seasons
        df = pd.read_parquet(fp, columns=["strength_global", "mu_xgf60", "mu_xga60",
                                          "sigma_xgf_w", "mu_xgf_p10", "mu_xgf_p90", "seconds"])
        # label drift across generations: '5v5' vs '5V5_5v5' vs '5V5'
        ev = df[df.strength_global.astype(str).str.replace("_", "").str.lower()
                  .isin(["5v5", "5v55v5"])]
        tag = fp.rsplit("_", 1)[-1].replace(".parquet", "")
        nan = ev[["mu_xgf60", "mu_xga60"]].isna().mean().mean()
        out.append(_res(f"mu.{tag}.nans", PASS if nan < 0.001 else FAIL, f"{nan:.4%} NaN mu (5v5)"))
        # the old quantile-blowup bug: p90 must exceed p10 and stay finite/sane
        blow = ((ev["mu_xgf_p90"] < ev["mu_xgf_p10"]) | (ev["mu_xgf_p90"].abs() > 1e3)).mean()
        out.append(_res(f"mu.{tag}.quantiles", PASS if blow < 0.001 else FAIL,
                        f"{blow:.4%} rows p90<p10 or |p90|>1e3 (quantile blowup regression)"))
        neg_sig = (ev["sigma_xgf_w"] <= 0).mean()
        out.append(_res(f"mu.{tag}.sigma", PASS if neg_sig < 0.001 else FAIL, f"{neg_sig:.4%} sigma<=0"))
        lvl = ev["mu_xgf60"].median()
        out.append(_res(f"mu.{tag}.level", PASS if 1.0 < lvl < 4.5 else WARN,
                        f"median mu_xgf60={lvl:.2f} (league 5v5 ~2.2-2.9)"))
    return out


def check_oof_team_windows(path: str) -> list[dict]:
    out = []
    df = pd.read_parquet(path, columns=["season", "holdout_season"]) if _has_cols(
        path, ["season", "holdout_season"]) else None
    if df is None:
        return [_res("oof.holdout_meta", FAIL, "holdout_season column missing — cannot verify OOF discipline")]
    same = (df["season"] == df["holdout_season"]).mean()
    out.append(_res("oof.holdout_meta", PASS if same > 0.999 else FAIL,
                    f"{same:.2%} rows predicted by a model that held their season out "
                    "(expected ~100%: each row scored by the fold that excluded it)"))
    return out


def _has_cols(path, cols):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    return all(c in names for c in cols)


def run_all(cfg: dict) -> list[dict]:
    p = cfg["paths"]
    results = []
    results += check_window_spine(p["window_spine"])
    results += check_player_game_obs(p["player_game_obs"])
    results += check_windows_with_baseline(p["player_windows_baseline_glob"])
    results += check_oof_team_windows(f"{p['baseline_dir']}/baseline_oof_team_windows.parquet")
    return results


def print_report(results: list[dict]) -> bool:
    ok = True
    for r in results:
        flag = {"PASS": "  ", "WARN": "! ", "FAIL": "XX"}[r["status"]]
        print(f"[{flag}] {r['status']:4s} {r['check']:32s} {r['detail']}")
        ok &= r["status"] != FAIL
    print("\n=>", "ALL CLEAR — pipeline may proceed" if ok else "FAILURES — pipeline blocked")
    return ok
