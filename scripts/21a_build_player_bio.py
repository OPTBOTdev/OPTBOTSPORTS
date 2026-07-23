"""Script 21a — player bio table (the join script 21 was always missing).

Every member of every window (focal + with + vs) gets: birthDate, shootsCatches,
position, height_cm, weight_kg. Sources, in trust order:
  1. local players_ages_<season>.json (birthDate; 2022 file never existed)
  2. D:\\2023\\missing_bio_manual.csv (hand-collected stragglers)
  3. NHL API player landing (fills hand/position everywhere + any missing DOB)

Output: artifacts/player_bio.parquet — one row per playerId, provenance column.
Age is NOT stored per-player here; the trainer computes age-at-window from
birthDate + game date (bio is time-invariant => lag-symmetric by construction,
but it still goes through the script-13 audit like everything else).

Usage: python scripts/21a_build_player_bio.py [--skip-api]
"""
import argparse
import glob
import json
import time
from pathlib import Path

import pandas as pd
import requests

ART = Path(r"D:\optbot\artifacts")
API = "https://api-web.nhle.com/v1/player/{pid}/landing"


def all_member_ids() -> list[int]:
    w = pd.read_parquet(ART / "perfect_windows_v3.parquet",
                        columns=["playerId", "with_ids", "vs_ids"])
    ids = set(int(x) for x in w.playerId.unique())
    for c in ("with_ids", "vs_ids"):
        for arr in w[c].dropna().values:
            ids.update(int(x) for x in arr if x)
    return sorted(ids)


def seed_local() -> dict[int, dict]:
    bio: dict[int, dict] = {}
    for fp in glob.glob(r"D:\20*\scored\players_ages_*.json"):
        d = json.load(open(fp))
        for pid, v in d.get("players", d).items():
            if isinstance(v, dict) and v.get("birthDate"):
                bio.setdefault(int(pid), {})["birthDate"] = v["birthDate"]
                bio[int(pid)].setdefault("provenance", "ages_json")
    man = pd.read_csv(r"D:\2023\missing_bio_manual.csv")
    for r in man.itertuples():
        b = bio.setdefault(int(r.playerId), {})
        b.setdefault("birthDate", r.birthDate)
        if isinstance(r.shootsCatches, str):
            b.setdefault("shootsCatches", r.shootsCatches)
        b.setdefault("provenance", "manual_csv")
    return bio


def fetch_api(ids, bio, throttle=0.12):
    need = [p for p in ids if not all(
        k in bio.get(p, {}) for k in ("birthDate", "shootsCatches", "position"))]
    print(f"API fetch: {len(need)}/{len(ids)} players need fields")
    sess = requests.Session()
    miss = []
    for i, pid in enumerate(need):
        try:
            r = sess.get(API.format(pid=pid), timeout=15)
            if r.status_code != 200:
                miss.append(pid)
                continue
            d = r.json()
            b = bio.setdefault(pid, {})
            b.setdefault("birthDate", d.get("birthDate"))
            b["shootsCatches"] = d.get("shootsCatches") or b.get("shootsCatches")
            b["position"] = d.get("position")
            b["height_cm"] = d.get("heightInCentimeters")
            b["weight_kg"] = d.get("weightInKilograms")
            b["name"] = (d.get("firstName", {}).get("default", "") + " "
                         + d.get("lastName", {}).get("default", "")).strip()
            b["provenance"] = b.get("provenance", "") + "+api"
        except requests.RequestException:
            miss.append(pid)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(need)} fetched, {len(miss)} misses")
        time.sleep(throttle)
    if miss:
        print(f"RETRY pass for {len(miss)} misses")
        for pid in list(miss):
            try:
                r = sess.get(API.format(pid=pid), timeout=20)
                if r.status_code == 200:
                    d = r.json()
                    b = bio.setdefault(pid, {})
                    b.setdefault("birthDate", d.get("birthDate"))
                    b["shootsCatches"] = d.get("shootsCatches")
                    b["position"] = d.get("position")
                    b["provenance"] = b.get("provenance", "") + "+api_retry"
                    miss.remove(pid)
            except requests.RequestException:
                pass
            time.sleep(0.4)
    return miss


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-api", action="store_true")
    args = ap.parse_args()
    ids = all_member_ids()
    print(f"{len(ids)} unique members across all windows")
    bio = seed_local()
    print(f"local seed: {sum(1 for p in ids if p in bio)} with birthDate")
    misses = [] if args.skip_api else fetch_api(ids, bio)
    rows = [{"playerId": p, **bio.get(p, {})} for p in ids]
    df = pd.DataFrame(rows)
    df.to_parquet(ART / "player_bio.parquet", index=False)
    for col in ("birthDate", "shootsCatches", "position"):
        cov = df[col].notna().mean() if col in df else 0.0
        print(f"coverage {col}: {cov:.2%}")
    print(f"wrote {len(df)} rows; unresolved after retry: {misses}")
