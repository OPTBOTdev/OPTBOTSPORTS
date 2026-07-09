"""Conformal counterfactual calibration (C4). Replaces the deleted 12%-coverage Kalman.

Split-conformal on the SWITCH population: signed errors from the frozen-at-t0 backtest,
binned by talent_n_eff. The band we show for a hypothetical trade has measured coverage
on real trades. Coverage is re-reported every time bands are fit; shipping a band without
its achieved coverage number is a contract violation.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd


class ConformalBands:
    def __init__(self, target: float = 0.80, n_eff_edges=(0, 8, 15, 1e9)):
        # n_eff is in DECAYED TOI-hours (talent_n_eff units): half-life-40 caps
        # veterans near ~17h, so edges 8/15 split rookie / mid / established.
        # (The old 25/50 edges put everyone in bin 0 — degenerate, F3 fix.)
        self.target = target
        self.edges = list(n_eff_edges)
        self.q_: dict[int, tuple[float, float]] = {}
        self.coverage_: dict[int, float] = {}

    def _bin(self, n_eff):
        return int(np.digitize([n_eff], self.edges[1:-1])[0])

    def fit(self, backtest: pd.DataFrame, pred_col="pred_xgf_pct",
            actual_col="actual_xgf_pct", n_eff_col="talent_n_eff") -> "ConformalBands":
        err = backtest[actual_col] - backtest[pred_col]
        lo_q, hi_q = (1 - self.target) / 2, 1 - (1 - self.target) / 2
        for b in range(len(self.edges) - 1):
            m = backtest[n_eff_col].map(self._bin) == b
            e = err[m]
            if len(e) < 20:                       # merge thin bins into global
                e = err
            self.q_[b] = (float(e.quantile(lo_q)), float(e.quantile(hi_q)))
            inside = ((e >= self.q_[b][0]) & (e <= self.q_[b][1])).mean()
            self.coverage_[b] = float(inside)
        return self

    def band(self, pred: float, n_eff: float) -> dict:
        b = self._bin(n_eff)
        lo, hi = self.q_[b]
        return {"lo": pred + lo, "hi": pred + hi,
                "target_coverage": self.target,
                "achieved_coverage": self.coverage_[b],   # ALWAYS shipped with the band
                "bin": b}

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"target": self.target, "edges": self.edges,
                       "q": self.q_, "coverage": self.coverage_}, f, indent=2)
