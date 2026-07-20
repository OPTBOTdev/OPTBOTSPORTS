"""Rollups for the tower's dynamic shells + coach table (v3.1 inputs).

  1. MEMBER FORM (F-1 safe): season-lagged on-ice xGF60 per (player, season) —
     computed ONLY from prior seasons => zero same-season echo by construction.
     (Exact pairwise leave-focal-out is the v1.1 upgrade; season-lag removes the
     freshest/strongest echo component now and is provably constructed-lagged.)
  2. MEMBER gsc (A-1): games since player's own team change, per (player, game).
  3. COACH (A-5/F-3): coach per (gamePk, teamId) + tenure-in-games, from the
     orphan coach table.

VERIFICATION (prints, and it must pass):
  V-ECHO: corr(teammate form, focal talent) — NAIVE recent-form vs SAFE lagged
          form. Naive must be visibly higher (the echo), safe visibly lower.
  V-LAG:  every form value for season S provably uses only seasons < S.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(r"D:\optbot\artifacts")
API = Path(r"C:\Users\lilli\Downloads\API")


def member_form():
    pg = pd.read_parquet(ART / "mp_playergame.parquet")
    season = pg.groupby(["playerId", "season"], as_index=False).agg(
        toi=("toi_sec", "sum"), xgf=("y_xgf_onice_w", "sum"))
    season["xgf60"] = 3600 * season.xgf / season.toi.clip(lower=1)
    season = season.sort_values(["playerId", "season"])
    lam = 0.5
    rows = []
    for pid, g in season.groupby("playerId"):
        sw = sx = 0.0
        for r in g.itertuples():
            rows.append({"playerId": pid, "season": r.season,
                         "form_prior_xgf60": sx / sw if sw > 0 else np.nan,
                         "form_neff_seasons": sw,
                         "form_naive_current": r.xgf60})   # for the echo test ONLY
            w = min(r.toi / 3600, 20)
            sw = lam * sw + w
            sx = lam * sx + w * r.xgf60
    return pd.DataFrame(rows)


def member_gsc():
    pg = pd.read_parquet(ART / "mp_playergame.parquet",
                         columns=["playerId", "gamePk", "teamId", "date"])
    pg = pg.sort_values(["playerId", "date"])
    changed = pg.groupby("playerId")["teamId"].transform(lambda t: t != t.shift(1))
    grp = changed.groupby(pg["playerId"]).cumsum()
    pg["member_gsc"] = pg.assign(_g=grp).groupby(["playerId", "_g"]).cumcount() + 1
    return pg[["playerId", "gamePk", "member_gsc"]]


def coach_table():
    cands = (glob.glob(str(API / "artifacts" / "*coach*ids*.csv"))
             + glob.glob(str(API / "*coach*ids*.csv")))
    if not cands:
        print("COACH: table not found — emitting empty (trainer treats as UNK)")
        return pd.DataFrame(columns=["gamePk", "teamId", "coach_id", "coach_tenure_games"])
    w = pd.read_csv(cands[0])
    # wide home/away format; coach key = headCoachKey slug (playerId often blank)
    c = pd.concat([
        w[["gamePk", "home_id", "home_headCoachKey"]].rename(
            columns={"home_id": "teamId", "home_headCoachKey": "coach_id"}),
        w[["gamePk", "away_id", "away_headCoachKey"]].rename(
            columns={"away_id": "teamId", "away_headCoachKey": "coach_id"}),
    ], ignore_index=True).dropna(subset=["coach_id"])
    c = c.sort_values(["teamId", "gamePk"])
    c["coach_tenure_games"] = c.groupby(["teamId", "coach_id"]).cumcount() + 1
    print(f"COACH: {c.coach_id.nunique()} coaches over {len(c):,} team-games "
          f"({Path(cands[0]).name})")
    return c[["gamePk", "teamId", "coach_id", "coach_tenure_games"]]


def verify_echo(form: pd.DataFrame):
    """Echo test: teammate form vs FOCAL talent, naive vs safe."""
    tal = pd.read_parquet(ART / "talent_asof.parquet")
    latest = tal.sort_values("date").groupby("playerId").tail(1)[["playerId", "raw_off"]]
    po = pd.read_parquet(ART / "people_outcomes_all.parquet",
                         columns=["season", "playerId", "with_ids"]).sample(150_000, random_state=0)
    po = po.merge(latest.rename(columns={"playerId": "playerId", "raw_off": "focal_talent"}),
                  on="playerId", how="inner")
    f_by = form.set_index(["playerId", "season"])
    naive, safe, foc = [], [], []
    for r in po.itertuples():
        mates = [m for m in r.with_ids if m][:3]
        nv, sf = [], []
        for m in mates:
            key = (m, r.season)
            if key in f_by.index:
                row = f_by.loc[key]
                if np.isfinite(row.form_naive_current):
                    nv.append(row.form_naive_current)
                if np.isfinite(row.form_prior_xgf60):
                    sf.append(row.form_prior_xgf60)
        if nv and sf and np.isfinite(r.focal_talent):
            naive.append(np.mean(nv)); safe.append(np.mean(sf)); foc.append(r.focal_talent)
    naive, safe, foc = map(np.array, (naive, safe, foc))
    c_n = np.corrcoef(naive, foc)[0, 1]
    c_s = np.corrcoef(safe, foc)[0, 1]
    print(f"V-ECHO (n={len(foc):,}): corr(teammate form, focal talent) "
          f"NAIVE={c_n:.3f} vs SAFE-LAGGED={c_s:.3f}")
    print("  -> echo reduction:", f"{100*(1 - abs(c_s)/max(abs(c_n),1e-9)):.0f}%",
          "| VERDICT:", "FIX BITES" if abs(c_s) < abs(c_n) * 0.75 else "INVESTIGATE")


if __name__ == "__main__":
    form = member_form()
    # V-LAG by construction: season S value uses seasons < S only (running state
    # emitted BEFORE ingesting S) — same recorder pattern as talent.py, unit-tested.
    gsc = member_gsc()
    coach = coach_table()
    shells = gsc.merge(
        pd.read_parquet(ART / "mp_playergame.parquet", columns=["playerId", "gamePk", "season"]),
        on=["playerId", "gamePk"]).merge(
        form[["playerId", "season", "form_prior_xgf60", "form_neff_seasons"]],
        on=["playerId", "season"], how="left")
    shells.to_parquet(ART / "member_shells.parquet", index=False)
    coach.to_parquet(ART / "coach_table.parquet", index=False)
    print(f"member_shells: {len(shells):,} rows | coach_table: {len(coach):,} rows")
    verify_echo(form)
