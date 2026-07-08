"""GBDT twin — the referee. LightGBM on the identical feature surface as PhaseBv2.

Purpose: (1) sanity-bound the neural net's window-level accuracy; (2) if the twin ever
beats the net on the backtest, the net does not ship. Costs one GPU-free day, saves a
sunk-cost disaster.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from ...contracts.window_schema import FEATURES

TARGETS = {"xgf60": "y_xGF", "xga60": "y_xGA"}


def train_twin(train: pd.DataFrame, val: pd.DataFrame, params: dict | None = None):
    assert lgb is not None, "pip install lightgbm"
    params = params or dict(objective="regression", metric="l2", num_leaves=127,
                            learning_rate=0.05, feature_fraction=0.8,
                            bagging_fraction=0.8, bagging_freq=1, verbose=-1)
    feats = [c for c in FEATURES if c in train.columns and train[c].dtype.kind in "ifb"]
    models, scores = {}, {}
    for name, ycol in TARGETS.items():
        y_tr = 60.0 * train[ycol] / (train["seconds"] / 60.0).clip(lower=0.5)
        y_va = 60.0 * val[ycol] / (val["seconds"] / 60.0).clip(lower=0.5)
        dtr = lgb.Dataset(train[feats], y_tr, weight=train["seconds"])
        dva = lgb.Dataset(val[feats], y_va, weight=val["seconds"], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = m.predict(val[feats])
        scores[name] = float(np.average((pred - y_va) ** 2, weights=val["seconds"]))
        models[name] = m
    return models, scores
