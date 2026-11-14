"""End-to-end smoke test on synthetic UNSW-shaped data (no downloads needed).

    python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

NUMERIC = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload", "dload",
           "sloss", "dloss", "sinpkt", "dinpkt", "smean", "dmean", "ct_srv_src", "ct_dst_src_ltm"]
CLASSES = {"Normal": 0.45, "Generic": 0.2, "Exploits": 0.15, "Fuzzers": 0.1, "DoS": 0.06,
           "Backdoor": 0.03, "Worms": 0.01}


def fake_unsw(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = rng.choice(list(CLASSES), n, p=list(CLASSES.values()))
    shift = {c: i for i, c in enumerate(CLASSES)}
    df = pd.DataFrame({c: np.abs(rng.normal(1 + 0.4 * np.vectorize(shift.get)(cats) * (j % 3), 1.0, n))
                       for j, c in enumerate(NUMERIC)})
    df.insert(0, "id", np.arange(1, n + 1))
    df.insert(2, "proto", rng.choice(["tcp", "udp", "arp"], n))
    df.insert(3, "service", np.where(cats == "Exploits", "http", rng.choice(["-", "dns", "ftp"], n)))
    df.insert(4, "state", rng.choice(["FIN", "INT", "CON"], n))
    df["attack_cat"] = cats
    df["label"] = (cats != "Normal").astype(int)
    return df


def main() -> None:
    root = os.path.join(os.path.dirname(__file__), "..")
    tmp = tempfile.mkdtemp(prefix="xda_smoke_")
    fake_unsw(12000, 0).to_csv(f"{tmp}/train.csv", index=False)
    fake_unsw(6000, 1).to_csv(f"{tmp}/test.csv", index=False)

    cfg = yaml.safe_load(open(os.path.join(root, "configs/default.yaml")))
    cfg["data"].update(unsw_train=f"{tmp}/train.csv", unsw_test=f"{tmp}/test.csv")
    cfg["monitor"]["window"] = 500
    cfg["shap"].update(background=50, reference_size=200, window_sample=100)
    cfg_path = f"{tmp}/cfg.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))

    py = sys.executable
    common = ["--config", cfg_path, "--max-epochs", "2"]
    for cmd in (
        [py, "scripts/run_static.py", *common, "--variants", "all", "--seeds", "1", "2",
         "--out", f"{tmp}/static"],
        [py, "scripts/run_drift.py", *common, "--family", "DoS", "--adapt", "--out", f"{tmp}/drift"],
        [py, "scripts/run_drift.py", *common, "--drift", "sudden", "gradual", "--class-mix", "Exploits",
         "--out", f"{tmp}/drift"],
        [py, "scripts/run_explain.py", *common, "--ks", "5", "10", "--n-explain", "300",
         "--out", f"{tmp}/explain"],
        [py, "scripts/stats.py", "--static", f"{tmp}/static", "--drift", f"{tmp}/drift"],
        [py, "scripts/analyze_errors.py", "--static", f"{tmp}/static", "--drift", f"{tmp}/drift"],
        [py, "scripts/make_figures.py", "--results", tmp, "--out", f"{tmp}/figures"],
        [py, "scripts/compare_base.py", "--results", tmp],
    ):
        print("\n$", " ".join(cmd[1:]), flush=True)
        subprocess.run(cmd, cwd=root, check=True)
    print(f"\nSmoke test passed. Outputs in {tmp}")


if __name__ == "__main__":
    main()
