"""Sealed predictions — freeze, hash, publish, score.

seal():   projections for every 2026-27 mover -> canonical JSON -> SHA-256.
          Post the hash publicly (dated). The file stays private until meetings.
verify(): recompute the hash from the file — proves no edits since sealing.
score():  weekly — join actuals, update the live scoreboard table.
"""
from __future__ import annotations
import hashlib
import json
from datetime import date

import pandas as pd


def canonical(preds: list[dict]) -> bytes:
    return json.dumps(sorted(preds, key=lambda p: p["player_id"]),
                      sort_keys=True, separators=(",", ":")).encode()


def seal(preds: list[dict], out_json: str, model_version: str) -> dict:
    payload = {"sealed_on": str(date.today()), "model_version": model_version,
               "predictions": preds}
    blob = canonical(preds)
    digest = hashlib.sha256(blob).hexdigest()
    payload["sha256_of_predictions"] = digest
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    return {"file": out_json, "sha256": digest,
            "publish_this": f"OptBot 2026-27 sealed projections sha256:{digest}"}


def verify(sealed_json: str) -> bool:
    with open(sealed_json) as f:
        payload = json.load(f)
    return hashlib.sha256(canonical(payload["predictions"])).hexdigest() \
        == payload["sha256_of_predictions"]


def score(sealed_json: str, obs: pd.DataFrame, horizon_games: int = 40) -> pd.DataFrame:
    from ..data.ledger import actual_post_move
    with open(sealed_json) as f:
        payload = json.load(f)
    rows = []
    for p in payload["predictions"]:
        a = actual_post_move(obs, p["player_id"], p["effective_date"], horizon_games)
        inside = (a["ok"] and p["band_lo"] <= a["xgf_pct"] <= p["band_hi"]) if a["ok"] else None
        rows.append({**p, "actual_xgf_pct": a.get("xgf_pct"), "gp_so_far": a.get("gp", 0),
                     "inside_band": inside})
    return pd.DataFrame(rows)
