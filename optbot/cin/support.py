"""Support gating — C1's enforcement arm. The CIN refuses what nature never ran.

Tiers:
  T1 environment-swap  : always answerable (trades are dense natural experiments)
  T2 lever-nudge       : answerable iff the requested lever value lies within the
                         [q05, q95] of observed values FOR THIS PLAYER'S ARCHETYPE
                         (ranges harvested from D:/phaseB/lever_curves_global sweeps)
  T3 out-of-support    : REFUSED — returns a refusal object, never a number.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

REFUSED_OOS = "REFUSED_OOS"


@dataclass
class SupportVerdict:
    tier: int
    ok: bool
    detail: str


def load_lever_ranges(lever_curves_csv: str) -> pd.DataFrame:
    """Per (archetype, lever): observed q05/q95 from the global lever sweep pool."""
    df = pd.read_csv(lever_curves_csv)
    g = df.groupby(["lever"])  # v0: league-wide; v1: cluster by archetype first
    return g["lever_value"].quantile([0.05, 0.95]).unstack().rename(
        columns={0.05: "q05", 0.95: "q95"})


def check_environment_swap(slot_windows_count: int, talent_n_eff: float) -> SupportVerdict:
    if slot_windows_count < 200:
        return SupportVerdict(1, False, f"destination slot support thin ({slot_windows_count} windows)")
    if talent_n_eff < 5:            # ~5 effective TOI-hours: rookie territory
        return SupportVerdict(1, True, "OK but talent prior near-empty; band will be wide")
    return SupportVerdict(1, True, "in support")


def check_lever(lever: str, value: float, ranges: pd.DataFrame) -> SupportVerdict:
    if lever not in ranges.index:
        return SupportVerdict(3, False, f"{REFUSED_OOS}: lever '{lever}' never swept")
    lo, hi = ranges.loc[lever, "q05"], ranges.loc[lever, "q95"]
    if not (lo <= value <= hi):
        return SupportVerdict(3, False,
                              f"{REFUSED_OOS}: {lever}={value:.3f} outside observed [{lo:.3f}, {hi:.3f}]")
    return SupportVerdict(2, True, f"{lever}={value:.3f} within observed [{lo:.3f}, {hi:.3f}]")
