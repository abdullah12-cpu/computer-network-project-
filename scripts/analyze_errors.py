"""Step 11: failure analysis from saved results.

  * residual confusions: largest off-diagonal cells (row-normalised) for each variant
  * monitor misses: scenarios where the XDA monitor never alarmed, with its signals
  * false alarms: every pre-onset XDA alarm, which signal fired and by how much

Per-class SHAP explanations of errors are produced by run_explain.py.

    python scripts/analyze_errors.py --static results/static --drift results/drift
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def confusions(folder: str, variants=("base_clean", "xda_full"), top: int = 8) -> None:
    for v in variants:
        for f in sorted(glob.glob(f"{folder}/*_{v}_s1.json")):
            r = json.load(open(f))
            cm = np.array(r["confusion_matrix"], dtype=float)
            names = r["class_names"]
            rate = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            np.fill_diagonal(rate, 0)
            pairs = sorted(((rate[i, j], names[i], names[j], int(cm[i, j]))
                            for i in range(len(names)) for j in range(len(names)) if rate[i, j] > 0), reverse=True)
            print(f"\n== {os.path.basename(f)[:-5]}: largest confusions (true -> predicted)")
            for p, a, b, n in pairs[:top]:
                print(f"  {a:15s} -> {b:15s} {100 * p:5.1f}%  ({n} flows)")


def monitor_failures(folder: str) -> None:
    rows_fa, rows_miss = [], []
    for f in sorted(glob.glob(f"{folder}/*.json")):
        d = json.load(open(f))
        scen = os.path.basename(f)[:-5]
        log = d["log"]
        if d["scores"]["xda"]["missed"]:
            post = [w for w in log if w["drifted"]]
            rows_miss.append({"scenario": scen,
                              "max_n_shap_shifted": max(w["n_shap_shifted"] for w in post),
                              "min_jaccard": min(w["jaccard"] for w in post),
                              "max_share_shift": max(w["share_shift"] for w in post),
                              "threshold_m": post[0]["signals"]["ks"]["threshold"]})
        for w in log:
            if w["drifted"] or not w["alarms"]["xda"]:
                continue
            fired = {k: s for k, s in w["signals"].items() if s["fired"]}
            # Margin over threshold, relative; small margins point to threshold calibration/sampling noise.
            margins = {k: abs(s["value"] - s["threshold"]) / (abs(s["threshold"]) + 1e-9) for k, s in fired.items()}
            kind = "near threshold (sampling noise / calibration)" if max(margins.values()) < 0.25 else "clear shift (benign variation?)"
            rows_fa.append({"scenario": scen, "window": w["window"], "signals": ",".join(fired),
                            "max_rel_margin": round(max(margins.values()), 3), "class": kind,
                            "top_shifted": [n for n, _ in w["top_shifted"][:3]]})
    print("\n== XDA monitor misses")
    print(pd.DataFrame(rows_miss).to_string(index=False) if rows_miss else "  none")
    print("\n== XDA false alarms (pre-onset windows)")
    print(pd.DataFrame(rows_fa).to_string(index=False) if rows_fa else "  none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default="results/static")
    ap.add_argument("--drift", default="results/drift")
    args = ap.parse_args()
    confusions(args.static)
    monitor_failures(args.drift)


if __name__ == "__main__":
    main()
