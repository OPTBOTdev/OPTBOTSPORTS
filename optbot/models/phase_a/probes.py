"""The 5-probe leakage scorecard — no Phase A checkpoint ships without this table.

Fixes the audit's 'evaluation vacuum': the elaborate anti-leakage design was never
validated. Each probe is a closed-form ridge fit on frozen embeddings (fast, no
training loop), run every epoch and at release.

  P1  h_skill -> teamId            want LOW  (identity must not encode employer)
  P2  h_skill -> oz_share          want LOW  (identity must not encode deployment)
  P3  h_skill -> next-season prior_off60   want HIGH (identity must mean talent)
  P4  same-player next-season retrieval top-5  want HIGH (stability across context)
  P5  cross-team twin-distance ratio       want ~1  (same player, different team,
                                            no farther apart than same-team pairs)
"""
from __future__ import annotations
import numpy as np


def _ridge_r2(X, y, lam=1e-2):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    Xb = np.c_[X, np.ones(len(X))]
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    resid = y - Xb @ w
    return 1 - resid.var() / max(y.var(), 1e-12)


def _linear_probe_acc(X, labels, lam=1e-2):
    classes, y = np.unique(labels, return_inverse=True)
    Y = np.eye(len(classes))[y]
    Xb = np.c_[X, np.ones(len(X))]
    W = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ Y)
    return float((np.argmax(Xb @ W, 1) == y).mean()), 1.0 / len(classes)


def scorecard(emb: np.ndarray, meta, next_emb=None, next_meta=None) -> dict:
    """emb: (N, d) h_skill per player-season; meta: DataFrame with columns
    playerId, teamId, oz_share, prior_off60_next. next_*: following season."""
    out = {}
    acc, chance = _linear_probe_acc(emb, meta["teamId"].values)
    out["P1_team_probe_acc"] = acc
    out["P1_pass"] = acc < chance * 3            # <3x chance = acceptably scrubbed
    out["P2_ozshare_r2"] = _ridge_r2(emb, meta["oz_share"].values)
    out["P2_pass"] = out["P2_ozshare_r2"] < 0.10
    out["P3_nextprior_r2"] = _ridge_r2(emb, meta["prior_off60_next"].values)
    out["P3_pass"] = out["P3_nextprior_r2"] > 0.30

    if next_emb is not None:
        a = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        b = next_emb / np.linalg.norm(next_emb, axis=1, keepdims=True)
        sim = a @ b.T
        ids_a = meta["playerId"].values
        ids_b = next_meta["playerId"].values
        match = ids_a[:, None] == ids_b[None, :]
        rank_hit = []
        for i in range(len(a)):
            if match[i].any():
                top5 = np.argsort(-sim[i])[:5]
                rank_hit.append(bool(match[i, top5].any()))
        out["P4_retrieval_top5"] = float(np.mean(rank_hit)) if rank_hit else np.nan
        out["P4_pass"] = out["P4_retrieval_top5"] > 0.5

        # twin ratio: same-player-cross-team distance / same-player-same-team distance
        d_cross, d_same = [], []
        for i in range(len(a)):
            j = np.flatnonzero(match[i])
            if len(j):
                d = 1 - sim[i, j[0]]
                (d_cross if meta["teamId"].values[i] != next_meta["teamId"].values[j[0]]
                 else d_same).append(d)
        if d_cross and d_same:
            out["P5_twin_ratio"] = float(np.mean(d_cross) / max(np.mean(d_same), 1e-9))
            out["P5_pass"] = out["P5_twin_ratio"] < 1.5
    out["ship"] = all(v for k, v in out.items() if k.endswith("_pass"))
    return out
