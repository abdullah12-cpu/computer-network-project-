"""Step 9: statistical and comparative analysis of saved results.

  static : mean +/- std over seeds, per-flow bootstrap 95% CI of Macro-F1,
           Wilcoxon signed-rank + paired t-test (xda_full vs each variant, over seeds),
           McNemar on per-flow predictions (xda_full vs base_clean, same split)
  drift  : per-monitor mean delay / false-alarm rate, Friedman test + Nemenyi post-hoc
           ranking across all E3/E4 scenarios

    python scripts/stats.py --static results/static --drift results/drift
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.report import tee_stdout  # noqa: E402

REFERENCE = "xda_full"


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [f1_score(y_true[i], y_pred[i], average="macro", zero_division=0)
            for i in (rng.integers(0, len(y_true), len(y_true)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def static_stats(folder: str) -> None:
    runs = [json.load(open(f)) for f in glob.glob(f"{folder}/*.json")]
    if not runs:
        print(f"no static results in {folder}")
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} |
                       {f"recall_{c}": v["recall"] for c, v in r["per_class"].items()} for r in runs])

    for (dataset, task), g in df.groupby(["dataset", "task"]):
        print(f"\n================ {dataset} / {task}")
        metrics = ["accuracy", "macro_f1", "weighted_f1", "fpr", "mcc"] + \
                  sorted(c for c in g.columns if c.startswith("recall_"))
        table = g.groupby("variant")[metrics].agg(["mean", "std"])
        print(table.xs("mean", axis=1, level=1).round(4).T.to_string())

        # Per-flow bootstrap CI of Macro-F1, averaged over seeds.
        ci = {}
        for variant in g["variant"].unique():
            bounds = []
            for f in glob.glob(f"{folder}/{dataset}_{task}_{variant}_s*_pred.npz"):
                p = np.load(f)
                bounds.append(bootstrap_ci(p["y_true"], p["y_pred"]))
            if bounds:
                ci[variant] = tuple(np.mean(bounds, 0).round(4))
        print("\nMacro-F1 bootstrap 95% CI (mean over seeds):", ci)

        if REFERENCE not in set(g["variant"]):
            continue
        ref = g[g["variant"] == REFERENCE].set_index("seed")
        rows = []
        for variant in sorted(set(g["variant"]) - {REFERENCE}):
            other = g[g["variant"] == variant].set_index("seed")
            seeds = ref.index.intersection(other.index)
            if len(seeds) < 2:
                continue
            for metric in ["macro_f1"] + [c for c in metrics if c.startswith("recall_")]:
                a, b = ref.loc[seeds, metric].to_numpy(), other.loc[seeds, metric].to_numpy()
                if np.allclose(a, b):
                    continue
                w = wilcoxon(a, b, alternative="greater")
                t = ttest_rel(a, b, alternative="greater")
                rows.append({"vs": variant, "metric": metric, "n_seeds": len(seeds),
                             "mean_diff": float((a - b).mean()), "wilcoxon_p_1sided": float(w.pvalue),
                             "ttest_p_1sided": float(t.pvalue)})
        if rows:
            print(f"\n{REFERENCE} better than ...? (one-sided; with 5 seeds the smallest possible "
                  f"Wilcoxon p is 0.031 one-sided / 0.0625 two-sided)")
            print(pd.DataFrame(rows).round(4).to_string(index=False))

        # McNemar: xda_full vs the reproduced base CNN on the same clean split.
        for seed in sorted(ref.index):
            fa = f"{folder}/{dataset}_{task}_{REFERENCE}_s{seed}_pred.npz"
            fb = f"{folder}/{dataset}_{task}_base_clean_s{seed}_pred.npz"
            if not (os.path.exists(fa) and os.path.exists(fb)):
                continue
            a, b = np.load(fa), np.load(fb)
            if not np.array_equal(a["idx_test"], b["idx_test"]):
                print(f"seed {seed}: test splits differ, McNemar skipped")
                continue
            ca, cb = a["y_pred"] == a["y_true"], b["y_pred"] == b["y_true"]
            tbl = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
            res = mcnemar(tbl, exact=False, correction=True)
            print(f"McNemar seed {seed}: xda right/base wrong={tbl[0][1]}, xda wrong/base right={tbl[1][0]}, "
                  f"p={res.pvalue:.2e}")


def drift_stats(folder: str) -> None:
    files = sorted(glob.glob(f"{folder}/*.json"))
    if not files:
        print(f"no drift results in {folder}")
        return
    delay, far, missed = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    summaries = []
    for f in files:
        d = json.load(open(f))
        scen = os.path.basename(f)[:-5]
        n_post = sum(w["drifted"] for w in d["log"])
        for mon, s in d["scores"].items():
            far[scen][mon] = s["false_alarm_rate"]
            if n_post == 0:      # natural-order stream without injected drift: false alarms only
                continue
            # A missed drift counts as the worst possible delay (whole post-onset period).
            delay[scen][mon] = s["delay_windows"] if s["detected"] else n_post
            missed[scen][mon] = s["missed"]
        summaries.append({"scenario": scen, **{k: v for k, v in d["summary"].items() if not isinstance(v, (list, dict))}})

    D, F, Ms = pd.DataFrame(delay).T, pd.DataFrame(far).T, pd.DataFrame(missed).T
    print("\n================ drift monitors")
    print(pd.DataFrame({"mean_delay": D.mean(), "missed": Ms.sum(),
                        "mean_false_alarm": F.mean()}).round(3).sort_values("mean_delay").to_string())
    print("\nper-scenario summary:")
    print(pd.DataFrame(summaries).round(3).to_string(index=False))

    for name, M in [("delay", D), ("false-alarm rate", F)]:
        M = M.dropna(axis=1)
        if M.shape[0] >= 3 and M.shape[1] >= 3:
            stat = friedmanchisquare(*[M[c] for c in M.columns])
            print(f"\nFriedman ({name}): chi2={stat.statistic:.2f}, p={stat.pvalue:.3g}")
            print("mean ranks (lower is better):", M.rank(axis=1).mean().sort_values().round(2).to_dict())
            try:
                import scikit_posthocs as sp
                print(sp.posthoc_nemenyi_friedman(M.to_numpy()).set_axis(M.columns, axis=0).set_axis(M.columns, axis=1).round(3))
            except ImportError:
                print("(pip install scikit-posthocs for the Nemenyi post-hoc table)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default="results/static")
    ap.add_argument("--drift", default="results/drift")
    ap.add_argument("--report", default=None, help="default: <results>/stats_report.txt")
    args = ap.parse_args()
    report = args.report or os.path.join(os.path.dirname(os.path.normpath(args.static)), "stats_report.txt")
    with tee_stdout(report):
        static_stats(args.static)
        drift_stats(args.drift)


if __name__ == "__main__":
    main()
