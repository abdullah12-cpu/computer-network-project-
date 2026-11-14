"""Step 12: report figures from saved results.

  fig_confusion_<ds>_<task>.png   base_clean vs xda_full confusion matrices (row-normalised)
  fig_recall_<ds>_<task>.png      per-class recall for every variant (rare-class view)
  fig_drift_<scenario>.png        monitor alarms and detector accuracy per window
  fig_monitors.png                mean detection delay and false-alarm rate per monitor
  fig_topk_<tag>.png              SHAP-guided Top-k: Macro-F1 and latency vs k
  fig_fidelity_<tag>.png          F1 drop when masking top-k vs random features

    python scripts/make_figures.py --results results --out figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MONITOR_ORDER = ["xda", "shap", "shap_ks_only", "input_ks", "confidence_ks", "kswin", "adwin", "ddm"]


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def confusion_and_recall(folder: str, out: str) -> None:
    runs = [json.load(open(f)) for f in glob.glob(f"{folder}/*.json")]
    if not runs:
        return
    groups = {}
    for r in runs:
        groups.setdefault((r["dataset"], r["task"]), []).append(r)

    for (ds, task), rs in groups.items():
        # Confusion matrices for seed 1 (or the first available seed).
        pick = {}
        for r in sorted(rs, key=lambda r: r["seed"]):
            pick.setdefault(r["variant"], r)
        shown = [v for v in ("base_repro", "base_clean", "xda_full") if v in pick]
        if shown:
            fig, axes = plt.subplots(1, len(shown), figsize=(5.2 * len(shown), 4.6))
            for ax, v in zip(np.atleast_1d(axes), shown):
                cm = np.array(pick[v]["confusion_matrix"], float)
                cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                names = pick[v]["class_names"]
                ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
                ax.set_yticks(range(len(names)), names, fontsize=7)
                for i in range(len(names)):
                    for j in range(len(names)):
                        if cm[i, j] >= 0.01:
                            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                    color="white" if cm[i, j] > 0.5 else "black")
                ax.set_title(f"{v}  (macro-F1 {pick[v]['macro_f1']:.3f})", fontsize=9)
                ax.set_xlabel("predicted")
                ax.set_ylabel("true")
            _save(fig, out, f"fig_confusion_{ds}_{task}.png")

        # Per-class recall, mean over seeds, for every variant.
        rec = pd.DataFrame([{"variant": r["variant"], **{c: v["recall"] for c, v in r["per_class"].items()}}
                            for r in rs]).groupby("variant").mean()
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * rec.shape[1] + 2), 4))
        rec.T.plot.bar(ax=ax, width=0.85)
        ax.set_ylabel("recall (mean over seeds)")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-class recall, {ds} {task}")
        ax.legend(fontsize=7, ncol=3)
        _save(fig, out, f"fig_recall_{ds}_{task}.png")


def drift(folder: str, out: str) -> None:
    files = sorted(glob.glob(f"{folder}/*_s1*.json"))
    rows = []
    for f in sorted(glob.glob(f"{folder}/*.json")):
        d = json.load(open(f))
        for mon, s in d["scores"].items():
            rows.append({"scenario": os.path.basename(f)[:-5], "monitor": mon,
                         "delay": s["delay_windows"], "detected": s["detected"], "far": s["false_alarm_rate"]})
    for f in files:
        d = json.load(open(f))
        log = d["log"]
        onset = next((w["window"] for w in log if w["drifted"]), len(log))
        mons = [m for m in MONITOR_ORDER if m in log[0]["alarms"]]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                       gridspec_kw={"height_ratios": [len(mons), 2]})
        for k, m in enumerate(mons):
            xs = [w["window"] for w in log if w["alarms"][m]]
            ax1.scatter(xs, [k] * len(xs), marker="s", s=40, color="tab:red" if m == "xda" else "tab:gray")
        ax1.set_yticks(range(len(mons)), mons, fontsize=8)
        ax1.set_title(os.path.basename(f)[:-5].replace("_", " "))
        ax2.plot([w["window"] for w in log], [w["detector_acc_binary"] for w in log], "o-", label="XDA detector")
        if "online_knn_acc_binary" in log[0]:
            ax2.plot([w["window"] for w in log], [w["online_knn_acc_binary"] for w in log], "s--", label="online KNN")
        ax2.set_ylabel("binary acc")
        ax2.set_xlabel("window")
        ax2.legend(fontsize=7)
        for ax in (ax1, ax2):
            ax.axvline(onset - 0.5, color="black", ls=":", lw=1)
        _save(fig, out, f"fig_drift_{os.path.basename(f)[:-5]}.png")

    if rows:
        df = pd.DataFrame(rows)
        df = df[~df["scenario"].str.contains("chronological")]
        n_post = 10
        df["delay_eff"] = df["delay"].where(df["detected"], n_post)
        agg = df.groupby("monitor")[["delay_eff", "far"]].mean().reindex([m for m in MONITOR_ORDER if m in set(df["monitor"])])
        fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
        agg["delay_eff"].plot.bar(ax=a, color=["tab:red" if m == "xda" else "tab:gray" for m in agg.index])
        a.set_ylabel("mean detection delay (windows; miss = 10)")
        agg["far"].plot.bar(ax=b, color=["tab:red" if m == "xda" else "tab:gray" for m in agg.index])
        b.set_ylabel("mean false-alarm rate")
        _save(fig, out, "fig_monitors.png")


def explain(folder: str, out: str) -> None:
    for f in sorted(glob.glob(f"{folder}/explain_*.json")):
        d = json.load(open(f))
        tag = os.path.basename(f)[len("explain_"):-5]
        sw = pd.DataFrame(d["topk_sweep"])
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.plot(sw["k"], sw["macro_f1"], "o-", color="tab:blue")
        ax.set_xlabel("number of SHAP-ranked features k")
        ax.set_ylabel("macro-F1", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(sw["k"], sw["latency_ms_per_flow"], "s--", color="tab:orange")
        ax2.set_ylabel("latency (ms/flow)", color="tab:orange")
        ax.set_title(f"SHAP-guided Top-k, {tag}")
        _save(fig, out, f"fig_topk_{tag}.png")

        fd = pd.DataFrame(d["fidelity"])
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        w = 0.38
        ax.bar(fd["k"].astype(str), fd["drop_top"], width=-w, align="edge", label="top-k SHAP masked")
        ax.bar(fd["k"].astype(str), fd["drop_random"], width=w, align="edge", label="random k masked")
        ax.set_xlabel("k features zeroed")
        ax.set_ylabel("macro-F1 drop")
        ax.set_title(f"Explanation fidelity, {tag}")
        ax.legend(fontsize=8)
        _save(fig, out, f"fig_fidelity_{tag}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    confusion_and_recall(f"{args.results}/static", args.out)
    drift(f"{args.results}/drift", args.out)
    explain(f"{args.results}/explain", args.out)


if __name__ == "__main__":
    main()
