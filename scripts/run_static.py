"""E1/E2 + detector ablation and baselines on the static split (Split A).

CNN variants (ablation, Step 10)
  base_repro    base paper as published in its notebooks (UNSW training CSV only,
                'id' kept, categoricals dropped, scaler fitted before split,
                no dedup, cross-entropy; WUSTL keeps the leaky 'Target' column)
  base_clean    same model and loss, our clean protocol (dedup, no id/Target, scaler on train)
  xda_no_focal  + categorical encoding, cross-entropy
  xda_no_cat    + focal loss, categoricals dropped
  xda_full      + categorical encoding + class-balanced focal loss  (XDA-IDS detector)
Baselines (Step 6), all on the clean protocol with categorical encoding
  lstm  rf  xgb  hybrid_rf_if

    python scripts/run_static.py --dataset unsw --task multiclass --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.baselines import HybridRFIF, RFDetector, XGBDetector  # noqa: E402
from xda.data import load_unsw, load_wustl  # noqa: E402
from xda.losses import FocalLoss, class_balanced_weights  # noqa: E402
from xda.metrics import detection_metrics, efficiency  # noqa: E402
from xda.models import CNN1D, LSTMNet  # noqa: E402
from xda.preprocess import clean, prepare_static  # noqa: E402
from xda.train import fit, predict_proba, set_seed  # noqa: E402

VARIANTS = {
    #               model     protocol encode focal  base_data
    "base_repro":   ("cnn",   "base",  False, False, True),
    "base_clean":   ("cnn",   "xda",   False, False, False),
    "xda_no_focal": ("cnn",   "xda",   True,  False, False),
    "xda_no_cat":   ("cnn",   "xda",   False, True,  False),
    "xda_full":     ("cnn",   "xda",   True,  True,  False),
    "lstm":         ("lstm",  "xda",   True,  False, False),
    "rf":           ("rf",    "xda",   True,  False, False),
    "xgb":          ("xgb",   "xda",   True,  False, False),
    "hybrid_rf_if": ("hybrid", "xda",  True,  False, False),
}
ABLATION = ["base_repro", "base_clean", "xda_no_focal", "xda_no_cat", "xda_full"]


def load(dataset: str, cfg: dict, base_data: bool):
    if dataset == "unsw":
        if base_data:   # exactly what the base notebooks used
            return load_unsw(cfg["data"]["unsw_train"], keep_id=True)
        return load_unsw([cfg["data"]["unsw_train"], cfg["data"]["unsw_test"]])
    return load_wustl(cfg["data"]["wustl"], keep_leaky=base_data)


def train_one(kind, prep, focal, cfg, seed, max_epochs):
    """Returns (fitted model, predict_proba callable, training info)."""
    n_cls, n_feat = len(prep.class_names), len(prep.feature_names)
    if kind in ("cnn", "lstm"):
        model = CNN1D(n_feat, n_cls, **cfg["cnn"]) if kind == "cnn" else LSTMNet(n_feat, n_cls)
        if focal:
            counts = np.bincount(prep.y_train, minlength=n_cls)
            loss = FocalLoss(class_balanced_weights(counts, cfg["focal"]["beta"]), cfg["focal"]["gamma"])
        else:
            loss = nn.CrossEntropyLoss()
        info = fit(model, prep.X_train, prep.y_train, prep.X_val, prep.y_val, loss,
                   max_epochs=max_epochs, patience=cfg["train"]["patience"],
                   batch_size=cfg["train"]["batch_size"], lr=cfg["train"]["lr"])
        return model, (lambda X: predict_proba(model, X)), info

    model = {"rf": lambda: RFDetector(seed), "xgb": lambda: XGBDetector(seed, n_cls),
             "hybrid": lambda: HybridRFIF(prep.normal_index, seed)}[kind]()
    t0 = time.perf_counter()
    model.fit(prep.X_train, prep.y_train)
    return model, model.predict_proba, {"epochs": None, "train_seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "wustl"], default="unsw")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--variants", nargs="+", default=ABLATION,
                    help=f"any of {list(VARIANTS)}; 'all' runs everything")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/static")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds or cfg["seeds"]
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]
    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for variant in variants:
        kind, protocol, encode, focal, base_data = VARIANTS[variant]
        ds = clean(load(args.dataset, cfg, base_data), dedup=(protocol == "xda"))
        if args.dataset == "wustl" and base_data:
            ds.categorical = []          # base notebook simply kept numeric columns

        for seed in seeds:
            tag = f"{args.dataset}_{args.task}_{variant}_s{seed}"
            if os.path.exists(os.path.join(args.out, tag + ".json")):
                print(f"skip {tag} (already done)")
                rows.append(_row(json.load(open(os.path.join(args.out, tag + ".json")))))
                continue
            print(f"\n### {tag}", flush=True)
            set_seed(seed)
            prep = prepare_static(ds, args.task, seed, protocol, encode,
                                  cfg["split"]["test_size"], cfg["split"]["val_size"])
            model, proba_fn, info = train_one(kind, prep, focal, cfg, seed, max_epochs)

            t0 = time.perf_counter()
            proba = proba_fn(prep.X_test)
            test_seconds = time.perf_counter() - t0
            m = detection_metrics(prep.y_test, proba, prep.class_names, prep.normal_index)
            if hasattr(model, "model") and hasattr(model.model, "n_jobs"):
                model.model.n_jobs = 1   # single-thread latency, like the CNN
            m.update(efficiency(model, len(prep.feature_names)))
            m.update({"dataset": args.dataset, "task": args.task, "variant": variant, "seed": seed,
                      "n_features": len(prep.feature_names), "n_train": len(prep.y_train),
                      "n_test": len(prep.y_test), "epochs": info["epochs"],
                      "train_seconds": info["train_seconds"], "test_seconds": test_seconds,
                      "class_names": prep.class_names, "features": prep.feature_names})
            json.dump(m, open(os.path.join(args.out, tag + ".json"), "w"), indent=1)
            # Per-flow predictions for McNemar's test (stats.py).
            np.savez_compressed(os.path.join(args.out, tag + "_pred.npz"),
                                y_true=prep.y_test, y_pred=proba.argmax(1), idx_test=prep.idx_test)
            print(f"  acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  weightedF1 {m['weighted_f1']:.4f}"
                  f"  FPR {m['fpr']:.4f}  test n={m['n_test']}")
            rows.append(_row(m))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, f"{args.dataset}_{args.task}_runs.csv"), index=False)
    cols = ["accuracy", "macro_f1", "weighted_f1", "fpr", "mcc", "pr_auc", "latency_ms_per_flow"] + \
           [c for c in df.columns if c.startswith("recall_")]
    df.groupby("variant")[cols].agg(["mean", "std"]).round(4).to_csv(
        os.path.join(args.out, f"{args.dataset}_{args.task}_summary.csv"))
    print("\n" + df.groupby("variant")[cols].mean().round(4).T.to_string())


def _row(m: dict) -> dict:
    return ({k: v for k, v in m.items() if not isinstance(v, (dict, list))} |
            {f"recall_{c}": v["recall"] for c, v in m["per_class"].items()})


if __name__ == "__main__":
    main()
