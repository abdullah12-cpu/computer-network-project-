"""Build the Phase 4 paper figures (vector PDF) from the Phase 3 quick-run results.

    python paper/figures/make_paper_figures.py

Every number plotted is read from results_quick/; nothing is typed in by hand.
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results_quick")
GREY, LIGHT, ACCENT, INK = "#8c8c8c", "#c8c8c8", "#1f5fbf", "#222222"
COL, FULL = 3.45, 7.1          # IEEE column / full text width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path):
    with open(os.path.join(RES, path), encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name))
    plt.close(fig)
    print("wrote", name)


def fig_recall():
    rows = []
    with open(os.path.join(RES, "comparison", "recall_unsw_multiclass.csv"), encoding="utf-8") as f:
        for r in list(csv.reader(f))[1:]:
            rows.append((r[0], float(r[1]), float(r[3]), float(r[4])))
    rows.sort(key=lambda r: r[3] - r[2])            # largest gain on top
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    for i, (cls, paper, base, xda) in enumerate(rows):
        ax.plot([base, xda], [i, i], color=LIGHT, lw=2, zorder=1)
        ax.scatter(paper, i, s=22, facecolor="white", edgecolor=INK, lw=0.8, zorder=2)
        ax.scatter(base, i, s=22, color=GREY, zorder=3)
        ax.scatter(xda, i, s=26, color=ACCENT, zorder=4)
        ax.text(1.04, i, f"{xda - base:+.2f}", va="center", fontsize=7,
                color=INK if xda >= base else GREY, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(-0.03, 1.0)
    ax.set_xlabel("Recall")
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="Base paper (reported)*"),
         plt.Line2D([], [], marker="o", ls="", color=GREY, label="Base CNN (leak-free)"),
         plt.Line2D([], [], marker="o", ls="", color=ACCENT, label="XDA-IDS")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig_recall.pdf")


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.0))
    for ax, (variant, title) in zip(axes, [("base_clean", "(a) Base CNN, leak-free"),
                                           ("xda_full", "(b) XDA-IDS")]):
        d = load(f"static/unsw_multiclass_{variant}_s1.json")
        cm = np.asarray(d["confusion_matrix"], float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        n = len(d["class_names"])
        for i in range(n):
            for j in range(n):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(n), d["class_names"], rotation=55, ha="right")
        ax.set_yticks(range(n), d["class_names"])
        ax.set_xlabel("Predicted class")
        ax.set_title(f"{title}, Macro-F1 {d['macro_f1']:.3f}")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("True class")
    fig.tight_layout(w_pad=2)
    save(fig, "fig_confusion.pdf")


def fig_monitors():
    names = {"xda": "XDA (SHAP + input)", "shap": "SHAP signals", "shap_ks_only": "SHAP-KS only",
             "input_ks": "Input KS", "confidence_ks": "Confidence KS", "kswin": "KSWIN",
             "adwin": "ADWIN (labels)", "ddm": "DDM (labels)"}
    rows = []
    with open(os.path.join(RES, "comparison", "drift_capability.csv"), encoding="utf-8") as f:
        for r in list(csv.reader(f))[1:]:
            if r[0] in names:
                rows.append((r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4])))
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    # (dx, dy, ha) label offsets in points, chosen so no label overlaps a point or another label
    offs = {"xda": (0, -13, "center"), "shap": (0, 8, "center"), "shap_ks_only": (9, -7, "left"),
            "input_ks": (9, -4, "left"), "confidence_ks": (9, 4, "left"), "kswin": (0, 8, "center"),
            "adwin": (9, -2, "left"), "ddm": (0, 8, "right")}
    for key, n, det, delay, far in rows:
        ours = key in ("xda", "shap", "shap_ks_only")
        ax.scatter(far, delay, s=18 + 50 * det, color=ACCENT if ours else GREY, alpha=0.9,
                   edgecolor="white", lw=0.5, zorder=3)
        dx, dy, ha = offs[key]
        ax.annotate(names[key], (far, delay), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6.5, ha=ha, va="center", color=INK)
    ax.axvline(0.05, color=GREY, ls=":", lw=0.8)
    ax.set_xlabel("False-alarm rate on drift-free windows")
    ax.set_ylabel("Mean detection delay (windows)")
    ax.set_xlim(-0.02, 0.36)
    ax.set_ylim(-0.4, 5.0)
    ax.grid(color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig_monitors.pdf")


def fig_timeline():
    d = load("drift/E3_unsw_DoS_s1.json")
    log = d["log"]
    order = [("xda", "XDA"), ("shap", "SHAP signals"), ("shap_ks_only", "SHAP-KS only"),
             ("input_ks", "Input KS"), ("confidence_ks", "Confidence KS"), ("kswin", "KSWIN"),
             ("adwin", "ADWIN"), ("ddm", "DDM")]
    onset = next(w["window"] for w in log if w["drifted"])
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for i, (key, label) in enumerate(order):
        y = len(order) - 1 - i
        ax.axhline(y, color="#eeeeee", lw=0.6, zorder=0)
        hits = [w["window"] for w in log if w["alarms"][key]]
        col = [ACCENT if key in ("xda", "shap", "shap_ks_only") else GREY] * len(hits)
        ax.scatter(hits, [y] * len(hits), marker="s", s=16, c=col, zorder=2)
    ax.axvline(onset - 0.5, color=INK, ls="--", lw=0.8)
    ax.text(onset - 0.3, len(order) - 0.4, "unseen DoS flows start", fontsize=6.5, va="bottom")
    ax.set_yticks(range(len(order)), [o[1] for o in reversed(order)])
    ax.set_xticks(range(0, len(log), 2))
    ax.set_xlim(-0.6, len(log) - 0.4)
    ax.set_ylim(-0.6, len(order) + 0.2)
    ax.set_xlabel("Window (5,000 flows each)")
    save(fig, "fig_timeline.pdf")


def fig_topk():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.8))
    for ax, ds, title in zip(axes, ["unsw", "wustl"], ["(a) UNSW-NB15", "(b) WUSTL-IIoT-2021"]):
        with open(os.path.join(RES, "explain", f"topk_{ds}_multiclass_s1.csv"), encoding="utf-8") as f:
            rows = [(int(r["k"]), float(r["macro_f1"])) for r in csv.DictReader(f)]
        ks, f1 = [r[0] for r in rows], [r[1] for r in rows]
        x = range(len(ks))
        full = f1[-1]
        best = next(i for i, v in enumerate(f1) if v >= full)
        ax.axhline(full, color=GREY, ls=":", lw=0.8)
        ax.plot(x, f1, color=LIGHT, lw=1.2, zorder=1)
        ax.scatter(x, f1, s=12, color=GREY, zorder=2)
        ax.scatter([best], [f1[best]], s=22, color=ACCENT, zorder=3)
        ax.annotate(f"{f1[best]:.3f}", (best, f1[best]), xytext=(4, -9), textcoords="offset points",
                    ha="left", fontsize=6.5, color=INK)
        ax.margins(y=0.15)
        ax.set_xticks(list(x), [str(k) if i < len(ks) - 1 else f"all\n({k})" for i, k in enumerate(ks)])
        ax.set_title(title)
        ax.set_xlabel("Top-k SHAP features")
    axes[0].set_ylabel("Macro-F1")
    fig.tight_layout(w_pad=1)
    save(fig, "fig_topk.pdf")


if __name__ == "__main__":
    fig_recall()
    fig_confusion()
    fig_monitors()
    fig_timeline()
    fig_topk()
