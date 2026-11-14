"""Compare XDA-IDS with the base paper (Udurume et al., 2026).

Columns per table: what the paper reports, what the paper's own code prints
(reference/base_notebook_outputs.txt), our reproduction of the base pipeline
(base_repro, as published; base_clean, leak-free protocol) and XDA-IDS (xda_full).

    python scripts/compare_base.py --results results
Writes <results>/comparison/comparison.md and one CSV per table.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.report import tee_stdout  # noqa: E402

OURS = ["base_repro", "base_clean", "xda_full"]


def fmt(vals) -> str:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "-"
    return f"{np.mean(vals):.3f}" if len(vals) == 1 else f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join([str(i)] + [str(v) for v in row]) + " |" for i, row in zip(df.index, df.values)]
    return "\n".join(lines)


def load_static(folder: str) -> list[dict]:
    return [json.load(open(f)) for f in glob.glob(f"{folder}/*.json")]


def headline(runs, ref) -> dict[str, pd.DataFrame]:
    tables = {}
    for ds in ("unsw", "wustl"):
        for task in ("binary", "multiclass"):
            rep = ref["reported"][ds][task]["cnn"]
            code = ref.get("base_code_output", {}).get(ds, {}).get(task, {}).get("cnn", {})
            rows = {
                "Paper reported (CNN)": {"accuracy": f"{rep['accuracy']:.3f}", "F1 (paper, avg unstated)": f"{rep['f1']:.3f}",
                                         "fpr": f"{ref['fpr']['unsw_binary']:.3f}" if (ds, task) == ("unsw", "binary") else "-"},
                "Base paper's code output": {"accuracy": f"{code['accuracy']:.3f}" if code else "-",
                                             "macro_f1": f"{code['macro_f1']:.2f}" if code else "-",
                                             "weighted_f1": f"{code['weighted_f1']:.2f}" if code else "-"},
            }
            for v in OURS:
                rs = [r for r in runs if (r["dataset"], r["task"], r["variant"]) == (ds, task, v)]
                if rs:
                    rows[f"Ours: {v} (n={len(rs)})"] = {m: fmt([r[m] for r in rs])
                                                        for m in ("accuracy", "macro_f1", "weighted_f1", "fpr", "mcc")}
            df = pd.DataFrame(rows).T.reindex(columns=["accuracy", "F1 (paper, avg unstated)", "macro_f1",
                                                       "weighted_f1", "fpr", "mcc"]).fillna("-")
            df.index.name = f"{ds} {task}"
            tables[f"headline_{ds}_{task}"] = df
    return tables


def rare_classes(runs, ref) -> dict[str, pd.DataFrame]:
    tables = {}
    for ds in ("unsw", "wustl"):
        paper = ref["per_class_recall"].get(f"{ds}_multiclass", {})
        rs = {v: [r for r in runs if (r["dataset"], r["task"], r["variant"]) == (ds, "multiclass", v)] for v in OURS}
        classes = sorted(set(paper) | {c for v in rs.values() for r in v for c in r["per_class"]})
        if not classes:
            continue
        rows = {}
        for c in classes:
            row = {"paper (Fig.)": f"{paper[c]:.3f}" if c in paper else "-"}
            for v in OURS:
                row[v] = fmt([r["per_class"][c]["recall"] for r in rs[v] if c in r["per_class"]])
            a = [r["per_class"][c]["recall"] for r in rs["xda_full"] if c in r["per_class"]]
            b = [r["per_class"][c]["recall"] for r in rs["base_clean"] if c in r["per_class"]]
            row["xda_full - base_clean"] = f"{np.mean(a) - np.mean(b):+.3f}" if a and b else "-"
            rows[c] = row
        df = pd.DataFrame(rows).T
        df.index.name = f"{ds} multiclass recall"
        tables[f"recall_{ds}_multiclass"] = df
    return tables


def efficiency_explanation(runs, explain_folder, ref) -> pd.DataFrame:
    eff, ex = ref["efficiency"]["cnn"], ref["explanation"]
    xda = [r for r in runs if r["variant"] == "xda_full" and r["dataset"] == "unsw"]
    rows = {
        "latency (ms/flow)": {"paper": f"{eff['latency_ms_per_flow']:.2f} (hardware not stated)",
                              "XDA-IDS": fmt([r["latency_ms_per_flow"] for r in xda]) + " (1 CPU thread)"},
        "model size (MB)": {"paper": f"{eff['memory_mb']:.1f} (runtime memory)",
                            "XDA-IDS": fmt([r["model_mb"] for r in xda]) + " (weights file)"},
        "parameters (M)": {"paper": f"{eff['params_m']:.2f}",
                           "XDA-IDS": fmt([r["params"] / 1e6 for r in xda if r.get("params")])},
    }
    for f in sorted(glob.glob(f"{explain_folder}/explain_unsw_multiclass_*.json"))[:1]:
        d = json.load(open(f))
        fid = {r["k"]: r for r in d["fidelity"]}
        f3 = fid.get(3)
        p = ex["fidelity_top3_masked"]
        rows["fidelity: F1 drop, top-3 masked"] = {
            "paper": f"{p['f1_full'] - p['f1_masked']:.3f} (F1 {p['f1_full']:.3f} -> {p['f1_masked']:.3f})",
            "XDA-IDS": (f"{f3['drop_top']:.3f} (random-3 control: {f3['drop_random']:.3f})" if f3 else "-")}
        lo, hi = ex["stability_top10_jaccard"]
        st = d["stability"]
        rows["stability: top-10 Jaccard"] = {"paper": f"{lo:.2f}-{hi:.2f} (5 bootstraps)",
                                             "XDA-IDS": f"{st['jaccard_mean']:.2f} ± {st['jaccard_std']:.2f} "
                                                        f"({st['n_boot']} bootstraps)"}
        k10 = next((r for r in d["topk_sweep"] if r["k"] == 10), None)
        rows["SHAP Top-10 model"] = {"paper": f"F1 {ex['topk']['f1']:.3f}, single k",
                                     "XDA-IDS": (f"macro-F1 {k10['macro_f1']:.3f}; full sweep k = "
                                                 + ", ".join(str(r["k"]) for r in d["topk_sweep"])) if k10 else "-"}
    df = pd.DataFrame(rows).T
    df.index.name = "efficiency / explanation"
    return df


def drift_capability(drift_folder) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{drift_folder}/*.json"):
        d = json.load(open(f))
        if not any(w["drifted"] for w in d["log"]):
            continue
        for mon, s in d["scores"].items():
            rows.append({"monitor": mon, "detected": s["detected"],
                         "delay": s["delay_windows"] if s["detected"] else np.nan, "far": s["false_alarm_rate"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).groupby("monitor").agg(
        scenarios=("detected", "size"), detected=("detected", "mean"),
        mean_delay_windows=("delay", "mean"), false_alarm_rate=("far", "mean")).round(3)
    df.loc["base paper"] = ["not evaluated"] * 4
    df.index.name = "drift / unseen attacks (E3, E4)"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--reference", default="reference/base_paper.yaml")
    args = ap.parse_args()
    ref = yaml.safe_load(open(args.reference))
    out = os.path.join(args.results, "comparison")
    os.makedirs(out, exist_ok=True)
    runs = load_static(f"{args.results}/static")

    tables = headline(runs, ref) | rare_classes(runs, ref)
    tables["efficiency_explanation"] = efficiency_explanation(runs, f"{args.results}/explain", ref)
    drift = drift_capability(f"{args.results}/drift")
    if not drift.empty:
        tables["drift_capability"] = drift

    md = ["# XDA-IDS vs base paper (Udurume et al., 2026)", "",
          "Paper numbers are as reported; 'base paper's code output' is what its own notebooks print.",
          "Ours: mean ± std over seeds. base_repro = base pipeline as published (incl. its leaks); "
          "base_clean = same model, leak-free protocol; xda_full = XDA-IDS detector.", ""]
    with tee_stdout(os.path.join(out, "comparison.txt")):
        for name, df in tables.items():
            df.to_csv(os.path.join(out, f"{name}.csv"))
            print(f"\n=== {name}\n{df.to_string()}")
            md += [f"## {name.replace('_', ' ')}", "", md_table(df), ""]
    open(os.path.join(out, "comparison.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"\nwrote {out}/comparison.md")


if __name__ == "__main__":
    main()
