"""SHOT PROFILES (F23 part 1) — bucketed shooting personalities per player.

Per (player, entering-season): from all his PRIOR shots (season-lagged, decayed),
using ARENA-ADJUSTED geometry (immune to T19 scorer bias):

  GRID       5 distance x 3 angle cells -> share of shots from each cell
  TYPES      wrist/snap/slap/tip/backhand/wrap shares
  PROCESS    rush share, rebound share, one-timer proxy, off-wing rate
  QUALITY    mean xGoal/shot (selectivity), mean 1-xGoal ("chance of save" faced),
             xShotWasOnGoal (expected accuracy)
  FINISHING  goals - xGoal per 100 shots (shooting talent), EB-shrunk by n
  CHAOS      mean xRebound, xFroze, xPlayContinuedInZone of his shots

Data: MP archive 2007-2024 + shots_2025 (two decades). Season-lagged: the profile
stamped "entering season S" uses only shots from seasons < S, decayed (hl=3yr).

Usage: python scripts/15_shot_profiles.py [--sniff-only]
Writes artifacts/shot_profiles.parquet + prints the Ovechkin/Matthews/Burns sniff.
"""
import argparse
from pathlib import Path

import pandas as pd

ART = Path(r"D:\optbot\artifacts")
DIST_EDGES = [0, 10, 20, 35, 50, 200]        # crease/slot/mid/point/long
DIST_LABELS = ["crease", "slot", "mid", "point", "long"]
ANG_EDGES = [0, 15, 35, 90]                  # center/mid/wide
ANG_LABELS = ["center", "midangle", "wide"]
HL_YEARS = 3.0

USE = ["season", "shooterPlayerId", "shooterName", "arenaAdjustedShotDistance",
       "shotAngleAdjusted", "shotType", "shotRush", "shotRebound", "goal",
       "xGoal", "xShotWasOnGoal", "xRebound", "xFroze", "xPlayContinuedInZone",
       "offWing", "speedFromLastEvent", "timeSinceLastEvent", "shotOnEmptyNet",
       "isPlayoffGame"]


def load_all() -> pd.DataFrame:
    parts = []
    it = pd.read_csv(r"D:\shots_2007-2024.zip", usecols=lambda c: c in USE,
                     chunksize=1_000_000)
    for c in it:
        parts.append(c)
    parts.append(pd.read_csv(r"D:\shots_2025.csv", usecols=lambda c: c in USE))
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.isPlayoffGame == 0) & (df.shotOnEmptyNet == 0)
            & df.shooterPlayerId.notna()].copy()
    df["shooterPlayerId"] = df.shooterPlayerId.astype("int64")
    df["dist_b"] = pd.cut(df.arenaAdjustedShotDistance, DIST_EDGES,
                          labels=DIST_LABELS, right=False)
    df["ang_b"] = pd.cut(df.shotAngleAdjusted.abs(), ANG_EDGES,
                         labels=ANG_LABELS, right=False)
    df["one_timer"] = ((df.timeSinceLastEvent <= 2)
                       & (df.speedFromLastEvent >= 10)).astype(int)
    return df


def season_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season): that season's raw profile. Lagging happens after."""
    g = df.groupby(["shooterPlayerId", "season"])
    out = g.agg(shots=("xGoal", "size"), xg_sum=("xGoal", "sum"),
                goals=("goal", "sum"), on_goal_x=("xShotWasOnGoal", "mean"),
                rush=("shotRush", "mean"), rebound=("shotRebound", "mean"),
                one_timer=("one_timer", "mean"), offwing=("offWing", "mean"),
                x_reb=("xRebound", "mean"), x_froze=("xFroze", "mean"),
                x_ozcont=("xPlayContinuedInZone", "mean"),
                name=("shooterName", "last"))
    grid = (df.groupby(["shooterPlayerId", "season", "dist_b", "ang_b"],
                       observed=True).size().unstack(["dist_b", "ang_b"],
                                                     fill_value=0))
    grid.columns = [f"cell_{d}_{a}" for d, a in grid.columns]
    grid = grid.div(grid.sum(axis=1).clip(lower=1), axis=0)
    typ = (df.groupby(["shooterPlayerId", "season"])["shotType"]
             .value_counts(normalize=True).unstack(fill_value=0))
    typ.columns = [f"type_{str(c).lower()}" for c in typ.columns]
    return out.join(grid).join(typ).reset_index()


def lag_and_decay(sp: pd.DataFrame, K_shots: float = 150.0) -> pd.DataFrame:
    """Entering-season profile: decayed weighted mean of PRIOR seasons only."""
    lam = 0.5 ** (1.0 / HL_YEARS)
    val_cols = [c for c in sp.columns if c not in
                ("shooterPlayerId", "season", "name", "shots", "goals", "xg_sum")]
    rows = []
    for pid, g in sp.sort_values("season").groupby("shooterPlayerId"):
        w_sum = 0.0
        acc = dict.fromkeys(val_cols, 0.0)
        fin_g = fin_x = n_shots = 0.0
        for r in g.itertuples():
            if w_sum > 0:
                shrink = n_shots / (n_shots + K_shots)
                rows.append({"playerId": pid, "entering_season": r.season,
                             "name": r.name, "prior_shots": n_shots,
                             "finishing_per100": shrink * 100 * (fin_g - fin_x)
                             / max(n_shots, 1),
                             **{c: acc[c] / w_sum for c in val_cols}})
            w = r.shots
            w_sum = lam * w_sum + w
            for c in val_cols:
                acc[c] = lam * acc[c] + w * getattr(r, c)
            fin_g = lam * fin_g + r.goals
            fin_x = lam * fin_x + r.xg_sum
            n_shots = lam * n_shots + r.shots
    return pd.DataFrame(rows)


def sniff(prof: pd.DataFrame):
    latest = prof.sort_values("entering_season").groupby("playerId").tail(1)
    picks = {8471214: "Ovechkin", 8479318: "Matthews", 8470613: "Burns",
             8478402: "McDavid", 8480839: "Pettersson?"}
    cells = [c for c in prof.columns if c.startswith("cell_")]
    for pid, nm in picks.items():
        m = latest[latest.playerId == pid]
        if m.empty:
            continue
        r = m.iloc[0]
        top = sorted(((c, r[c]) for c in cells), key=lambda x: -x[1])[:3]
        print(f"\n{r['name']} (prior shots {r.prior_shots:.0f}):")
        print("  top cells: " + " | ".join(f"{c[5:]}={v:.0%}" for c, v in top))
        print(f"  one-timer {r.one_timer:.0%} · rush {r.rush:.0%} · offwing "
              f"{r.offwing:.0%} · selectivity(xG/shot) {r.xg_sum/max(r.prior_shots,1) if 'xg_sum' in r else float('nan')}")
        print(f"  finishing {r.finishing_per100:+.1f} goals/100 shots above xG "
              f"· chaos(xReb) {r.x_reb:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sniff-only", action="store_true")
    args = ap.parse_args()
    print("loading two decades of shots...")
    df = load_all()
    print(f"{len(df):,} shots, {df.shooterPlayerId.nunique():,} shooters, "
          f"seasons {df.season.min()}-{df.season.max()}")
    sp = season_profiles(df)
    prof = lag_and_decay(sp)
    prof.to_parquet(ART / "shot_profiles.parquet", index=False)
    print(f"wrote {len(prof):,} (player, entering-season) profiles")
    sniff(prof)
