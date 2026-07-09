#!/usr/bin/env python3
"""
Score per-shot xG using trained pure models (5v5 / PP / PK).

- Reads shots_train_*.csv from --shots_dir
- Adds column 'xg' (calibrated probability) ONLY for unblocked shots in periods 1–3
- Selects the correct model by strength (5V5, PP, PK); skips EN/EA/4v4/OT/etc.
- Writes scored CSVs into --out_dir with *_scored.csv suffix
"""
import os, glob, argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

# -------------------- strength helpers --------------------
def derive_strength_col(df: pd.DataFrame, prefer_col: str = "strength") -> pd.Series:
    """Normalize strength into {'5V5','PP','PK','OTHER','EN','EA'}."""
    if prefer_col in df.columns:
        s = df[prefer_col].astype(str).str.upper().str.replace(" ", "", regex=False)
        s = s.replace({"EVEN":"5V5","EV":"5V5"})
        s = s.replace({"EA":"EA","EA_HOME":"EA","EA_AWAY":"EA",
                       "EN":"EN","EN_FOR":"EN","EN_AGAINST":"EN"})
        return s
    us = pd.to_numeric(df.get("us_skaters", np.nan), errors="coerce")
    them = pd.to_numeric(df.get("them_skaters", np.nan), errors="coerce")
    out = np.where((us==5)&(them==5), "5V5",
          np.where(us>them, "PP",
          np.where(us<them, "PK", "OTHER")))
    return pd.Series(out, index=df.index)

# -------------------- categorical hygiene --------------------
def canon_cat_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    s2 = s2.mask(s.isna(), "__unk__")
    s2 = s2.replace({"nan": "__unk__", "": "__unk__"})
    return s2

def align_categories(X: pd.DataFrame, cats: List[str], saved_levels: Dict[str, List[str]] | None) -> None:
    saved_levels = saved_levels or {}
    for c in cats:
        X[c] = canon_cat_series(X[c])
        levels = saved_levels.get(c)
        if levels:
            ok = X[c].isin(levels)
            X.loc[~ok, c] = "__unk__"
            X[c] = pd.Categorical(X[c], categories=levels)
        else:
            lvls = sorted(X[c].dropna().unique().tolist())
            if "__unk__" not in lvls:
                lvls.append("__unk__")
            X[c] = X[c].where(X[c].isin(lvls), "__unk__")
            X[c] = pd.Categorical(X[c], categories=lvls)

# -------------------- model loader --------------------
def load_bundle(models_root: str, strength: str, mode: str = "pure") -> dict:
    path = Path(models_root) / strength / mode / "xg_lightgbm_isotonic.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    return joblib.load(path)

# -------------------- scoring --------------------
def score_block(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    model = bundle["calibrated_model"]
    feat_cols = list(bundle["features"])
    cat_cols = list(bundle.get("categoricals", []))
    saved_cats = bundle.get("saved_categories", None)

    X = df[feat_cols].copy()

    if cat_cols:
        align_categories(X, cat_cols, saved_cats)

    num_cols = [c for c in feat_cols if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    good_mask = X[num_cols].notna().all(axis=1)
    probs = np.full(len(df), np.nan, dtype=float)
    if good_mask.any():
        p = model.predict_proba(X.loc[good_mask])[:, 1]
        eps = float(bundle["config"].get("epsilon", 1e-5))
        p = np.clip(p, eps, 1 - eps)
        probs[good_mask.to_numpy()] = p
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots_dir", required=True, help="Folder with shots_train_*.csv")
    ap.add_argument("--models_root", required=True, help="Artifacts root (contains 5V5/PP/PK subdirs)")
    ap.add_argument("--out_dir", required=True, help="Where scored CSVs go")
    ap.add_argument("--strength_col", default="strength", help="Strength column name (default: 'strength')")
    ap.add_argument("--unblocked_col", default="is_unblocked", help="1 for SOG+miss, 0 for blocked")
    ap.add_argument("--mode", default="pure", choices=["pure"], help="Model mode (v1 uses 'pure')")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load once per strength
    bundles = {
        "5V5": load_bundle(args.models_root, "5V5", args.mode),
        "PP":  load_bundle(args.models_root, "PP",  args.mode),
        "PK":  load_bundle(args.models_root, "PK",  args.mode),
    }

    files = sorted(glob.glob(os.path.join(args.shots_dir, "shots_train_*.csv")))
    if not files:
        raise FileNotFoundError(f"No shots_train_*.csv found under {args.shots_dir}")
    print(f"Scoring {len(files)} file(s) from {args.shots_dir}")

    for f in files:
        df = pd.read_csv(f)

        # Period filter (ONLY 1–3)  # NEW
        if "period" not in df.columns:
            raise ValueError(f"{os.path.basename(f)} missing 'period' column")
        df["_PERIOD_OK"] = pd.to_numeric(df["period"], errors="coerce").between(1, 3)

        # Normalize strength
        df["_STRENGTH_TMP"] = derive_strength_col(df, args.strength_col)

        # Unblocked flag
        if args.unblocked_col not in df.columns:
            raise ValueError(f"{os.path.basename(f)} missing '{args.unblocked_col}'")
        unblk = pd.to_numeric(df[args.unblocked_col], errors="coerce").fillna(0).astype(int)
        df["_UNBLOCKED"] = (unblk == 1)

        # Init xg
        df["xg"] = np.nan

        # Score by (period 1–3) & (unblocked) & (strength)
        base_mask = df["_PERIOD_OK"] & df["_UNBLOCKED"]  # NEW

        for label, key in [("5V5","5V5"), ("PP","PP"), ("PK","PK")]:
            mask = base_mask & (df["_STRENGTH_TMP"] == label)
            if not mask.any():
                continue
            probs = score_block(df.loc[mask], bundles[key])
            df.loc[mask, "xg"] = probs

        # Stats
        n_total = len(df)
        n_p13 = int(df["_PERIOD_OK"].sum())
        n_unblk = int((df["_PERIOD_OK"] & df["_UNBLOCKED"]).sum())
        n_scored = int(df["xg"].notna().sum())
        print(f"{os.path.basename(f)}: total={n_total:,}  p1-3={n_p13:,}  unblocked_p1-3={n_unblk:,}  scored={n_scored:,}")

        # Write
        out_path = Path(args.out_dir) / (Path(f).stem + "_scored.csv")
        df.drop(columns=["_STRENGTH_TMP","_UNBLOCKED","_PERIOD_OK"], inplace=True)
        df.to_csv(out_path, index=False)

    print("Done.")

if __name__ == "__main__":
    main()
