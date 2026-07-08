"""Gate zero: QA every source dataset. FAIL blocks the pipeline.
Usage:  python scripts/00_validate_data.py  [--config configs/default.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402
from optbot.data.validate import run_all, print_report  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    results = run_all(cfg)
    ok = print_report(results)
    out = Path(cfg["paths"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    with open(out / "data_validation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(0 if ok else 1)
