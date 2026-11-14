"""Inspect downloaded datasets: shape, classes, duplicates and leakage checks.

    python scripts/check_data.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.data import class_counts, load_unsw, load_wustl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))["data"]

    for split in ("unsw_train", "unsw_test"):
        try:
            ds = load_unsw(cfg[split], keep_id=True)
        except FileNotFoundError:
            print(f"[missing] {cfg[split]}")
            continue
        print(f"\n=== {split}: {ds.X.shape[0]:,} rows, {ds.X.shape[1]} columns")
        print(class_counts(ds).to_string())
        dup = pd.concat([ds.X.drop(columns="id"), ds.y_multi], axis=1).duplicated().mean()
        print(f"duplicate rows (ignoring id): {100 * dup:.1f}%")
        # Leakage check: can the row id alone predict the label?
        rho = np.corrcoef(ds.X["id"], ds.y_bin)[0, 1]
        print(f"corr(id, label) = {rho:+.3f}   (far from 0 => 'id' leaks the label)")

    try:
        ds = load_wustl(cfg["wustl"], keep_leaky=True)
    except FileNotFoundError:
        print(f"[missing] {cfg['wustl']}")
    else:
        print(f"\n=== wustl: {ds.X.shape[0]:,} rows, {ds.X.shape[1]} columns")
        print(class_counts(ds).to_string())
        if "Target" in ds.X.columns:
            agree = (ds.X["Target"].astype(int) == ds.y_bin).mean()
            print(f"'Target' equals the binary label on {100 * agree:.2f}% of rows (=> leakage if used as input)")


if __name__ == "__main__":
    main()
