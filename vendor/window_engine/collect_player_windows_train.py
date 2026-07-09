#!/usr/bin/env python3
"""
Collect + filter per-game `player_windows_train_*.csv` files into a single clean folder.

Use-case
--------
During season builds you can end up with some "empty" outputs (header only) or
tiny files. This script copies (or moves) only the "good" files into one folder,
and writes a manifest of kept/dropped.

Default behavior is SAFE:
- copies files (does not delete originals)
- drops header-only / tiny / too-few-rows files

Example (2025-26 season):
  python collect_player_windows_train.py \
    --src API/Final/2025/derived/windows \
    --dst API/Final/2025/derived/windows/player_windows_train_clean \
    --min-rows 50
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import Dict, List, Tuple


def _count_lines_fast(path: str) -> int:
    """Count '\n' in binary chunks. Returns number of lines (best effort)."""
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            n += b.count(b"\n")
    # If file doesn't end with newline but has content, count_lines by newline may be off by 1.
    # We'll treat this as "at least n+1 lines" if size>0.
    try:
        if os.path.getsize(path) > 0:
            return n + 1
    except Exception:
        pass
    return n


def _read_header(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        line = f.readline()
    return [c.strip() for c in (line or "").strip().split(",") if c.strip()]


def _is_good(
    path: str,
    *,
    min_rows: int,
    min_bytes: int,
    required_cols: List[str],
) -> Tuple[bool, Dict[str, str]]:
    meta: Dict[str, str] = {}
    try:
        size = os.path.getsize(path)
    except Exception:
        size = -1
    meta["bytes"] = str(size)
    if size >= 0 and size < min_bytes:
        meta["reason"] = f"too_small_bytes<{min_bytes}"
        return False, meta

    header = _read_header(path)
    meta["has_header"] = "1" if bool(header) else "0"
    if not header:
        meta["reason"] = "missing_header"
        return False, meta

    missing = [c for c in required_cols if c not in header]
    if missing:
        meta["reason"] = f"missing_cols:{'|'.join(missing)}"
        return False, meta

    lines = _count_lines_fast(path)
    # lines includes header; data rows approx lines-1
    data_rows = max(0, lines - 1)
    meta["rows"] = str(data_rows)
    if data_rows < min_rows:
        meta["reason"] = f"too_few_rows<{min_rows}"
        return False, meta

    return True, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect + filter player_windows_train CSVs into one folder.")
    ap.add_argument("--src", default=os.path.join("API", "Final", "2025", "derived", "windows"))
    ap.add_argument("--dst", default=os.path.join("API", "Final", "2025", "derived", "windows", "player_windows_train_clean"))
    ap.add_argument("--pattern", default="player_windows_train_", help="Filename prefix to match")
    ap.add_argument("--min-rows", type=int, default=50, help="Minimum data rows required to keep a file")
    ap.add_argument("--min-bytes", type=int, default=2_000, help="Minimum file size (bytes) required to keep a file")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite in dst if already exists")
    ap.add_argument("--manifest", default=None, help="Write a manifest CSV (default: <dst>/manifest.csv)")
    args = ap.parse_args()

    src = str(args.src)
    dst = str(args.dst)
    os.makedirs(dst, exist_ok=True)

    manifest_path = str(args.manifest or os.path.join(dst, "manifest.csv"))

    # Required columns we expect from perfect_windows player_windows_train output
    required_cols = ["gamePk", "teamId", "playerId", "seconds", "xGF", "xGA"]

    files = []
    for fn in os.listdir(src) if os.path.isdir(src) else []:
        if not fn.startswith(args.pattern) or not fn.endswith(".csv"):
            continue
        files.append(fn)
    files.sort()

    kept = dropped = 0
    rows_out: List[Dict[str, str]] = []

    for fn in files:
        src_path = os.path.join(src, fn)
        dst_path = os.path.join(dst, fn)

        ok, meta = _is_good(
            src_path,
            min_rows=int(args.min_rows),
            min_bytes=int(args.min_bytes),
            required_cols=required_cols,
        )
        rec = {"file": fn, "src": src_path, "dst": dst_path, "kept": "1" if ok else "0"}
        rec.update(meta)
        rows_out.append(rec)

        if not ok:
            dropped += 1
            continue

        if (not args.overwrite) and os.path.exists(dst_path):
            # treat as kept, but don't copy again
            kept += 1
            continue

        if args.move:
            shutil.move(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
        kept += 1

    # Write manifest
    fieldnames = sorted({k for r in rows_out for k in r.keys()})
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"Source: {src}")
    print(f"Dest:   {dst}")
    print(f"Kept:   {kept}")
    print(f"Dropped:{dropped}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





