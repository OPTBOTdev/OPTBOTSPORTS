"""Projection engine.

v0 (MVP, no neural net on the path):
    projected_xgf60 = E_slot[ mu_xgf60 ]  +  talent_off_shrunk
    projected_xga60 = E_slot[ mu_xga60 ]  -  talent_def_shrunk
  where E_slot is the seconds-weighted mean OOF baseline over the scenario's synthetic
  windows (the destination environment) and talent_* is the player's frozen prior.
  Interpretable in one sentence to a GM's analytics staff. That is a feature.

v1 (neural): identical interface, but ŷ comes from PhaseB v2 forward passes over the
  synthetic windows. Registered via `engine=` so the backtest harness scores both.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .scenario import Scenario, build_windows


def project_v0(sc: Scenario, windows: pd.DataFrame, lines: pd.DataFrame,
               player_row: pd.Series) -> dict:
    synth = build_windows(sc, windows, lines, player_row)
    w = synth["seconds"].clip(lower=1).values
    env_xgf = float(np.average(synth["mu_xgf60"], weights=w))
    env_xga = float(np.average(synth["mu_xga60"], weights=w))

    xgf60 = env_xgf + float(player_row.get("talent_off_shrunk", 0.0))
    xga60 = env_xga - float(player_row.get("talent_def_shrunk", 0.0))
    xga60 = max(xga60, 0.1)

    return {
        "engine": "v0_baseline_plus_talent",
        "scenario_hash": synth["scenario_hash"].iloc[0],
        "env_xgf60": env_xgf, "env_xga60": env_xga,
        "talent_off": float(player_row.get("talent_off_shrunk", 0.0)),
        "talent_def": float(player_row.get("talent_def_shrunk", 0.0)),
        "talent_n_eff": float(player_row.get("talent_n_eff", 0.0)),
        "xgf60": xgf60, "xga60": xga60,
        "xgf_pct": 100.0 * xgf60 / (xgf60 + xga60),
        "n_synth_windows": int(len(synth)),
    }


def project_v1_neural(sc: Scenario, windows: pd.DataFrame, lines: pd.DataFrame,
                      player_row: pd.Series, model, featurizer) -> dict:
    """PhaseB v2 over the same synthetic windows. model/featurizer injected so this
    module stays torch-free; see models/phase_b/train.py for the loading side."""
    synth = build_windows(sc, windows, lines, player_row)
    X = featurizer(synth)
    mu = model.predict_mu(X)                     # (n, heads) in per-60 space
    w = synth["seconds"].clip(lower=1).values
    xgf60 = float(np.average(mu[:, 0], weights=w))
    xga60 = float(np.average(mu[:, 1], weights=w))
    return {"engine": "v1_phaseB", "scenario_hash": synth["scenario_hash"].iloc[0],
            "xgf60": xgf60, "xga60": xga60,
            "xgf_pct": 100.0 * xgf60 / (xgf60 + xga60),
            "n_synth_windows": int(len(synth))}
