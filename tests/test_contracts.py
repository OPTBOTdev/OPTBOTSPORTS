"""Contract + core-math tests. Fast, no data files needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optbot.contracts import window_schema as ws
from optbot.priors.talent import _decayed_running
from optbot.cin.conformal import ConformalBands


def test_outcomes_never_in_features():
    assert not set(ws.OUTCOMES) & set(ws.FEATURES)


def test_prior_naming_contract():
    for c in ws.PRIORS:
        if c in ws._BIO:
            continue
        assert ws._LAG_PAT.search(c), f"{c} lacks lag suffix"


def test_decayed_running_is_lagged():
    """The value at row i must EXCLUDE game i — the leak-proof property."""
    df = pd.DataFrame({"playerId": [1, 1, 1], "v": [10.0, 20.0, 30.0],
                       "w": [1.0, 1.0, 1.0]})
    r = _decayed_running(df, "v", "w", half_life_games=1e9)
    assert np.isnan(r["raw"].iloc[0])          # nothing known before game 1
    assert r["raw"].iloc[1] == 10.0            # only game 1 known before game 2
    assert r["raw"].iloc[2] == pytest.approx(15.0)  # mean(10,20) before game 3


def test_shrinkage_direction():
    """Low n_eff -> shrunk toward 0; high n_eff -> keeps signal."""
    n_lo, n_hi, K = 1.0, 100.0, 25.0
    raw = 1.0
    assert n_lo / (n_lo + K) * raw < 0.1
    assert n_hi / (n_hi + K) * raw > 0.7


def test_conformal_coverage_on_synthetic():
    rng = np.random.default_rng(0)
    n = 2000
    bt = pd.DataFrame({
        "pred_xgf_pct": 50 + rng.normal(0, 1, n),
        "talent_n_eff": rng.uniform(0, 100, n),
    })
    bt["actual_xgf_pct"] = bt.pred_xgf_pct + rng.normal(0, 3, n)
    cb = ConformalBands(target=0.8).fit(bt)
    covs = list(cb.coverage_.values())
    assert all(0.74 <= c <= 0.86 for c in covs), covs


def test_conformal_band_ships_coverage():
    bt = pd.DataFrame({"pred_xgf_pct": np.zeros(100),
                       "actual_xgf_pct": np.random.default_rng(1).normal(0, 1, 100),
                       "talent_n_eff": np.full(100, 50.0)})
    b = ConformalBands(target=0.8).fit(bt).band(50.0, 50.0)
    assert "achieved_coverage" in b and b["lo"] < 50.0 < b["hi"]


def test_validator_catches_outcome_leak(monkeypatch):
    monkeypatch.setattr(ws, "FEATURES", ws.FEATURES + ["y_xGF"])
    df = pd.DataFrame({c: [] for c in ws.ALL_COLUMNS})
    with pytest.raises(ws.ContractError, match="outcome columns in FEATURES"):
        ws.validate(df)
