"""Step 4 + Step 8 explanation quality and the SHAP-guided Top-k sweep (fixes L6).

  1. train the XDA-IDS detector (xda_full)
  2. SHAP on held-out test flows -> global ranking, fidelity (top-k vs random masking),
     stability (bootstrap Jaccard)
  3. retrain the CNN on the top-k features for each k and record Macro-F1 and latency

    python scripts/run_explain.py --dataset unsw --task multiclass --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.data import load_unsw, load_wustl  # noqa: E402
from xda.explain import Explainer, fidelity, global_ranking, stability  # noqa: E402
from xda.losses import FocalLoss, class_balanced_weights  # noqa: E402
from xda.metrics import detection_metrics, efficiency  # noqa: E402
from xda.models import CNN1D  # noqa: E402
from xda.preprocess import clean, prepare_static  # noqa: E402
from xda.train import fit, predict_proba, set_seed  # noqa: E402


# TTL-type features in UNSW-NB15 are known to encode the testbed rather than attack behaviour.
ARTEFACTS = {"sttl", "dttl", "ct_state_ttl", "sTtl", "dTtl"}


def explain_errors(attr, y, pred, prep, min_wrong: int = 5) -> dict:
    """Step 11: for each class, which classes absorb its errors and which features
    (largest mean SHAP toward the wrong prediction) misled the detector.
    'Normal' errors are the false positives."""
    from collections import Counter
    out = {}
    for c, name in enumerate(prep.class_names):
        wrong = (y == c) & (pred != c)
        if wrong.sum() < min_wrong:
            continue
        mean_attr = attr[wrong].mean(0)          # attributions are for the (wrong) predicted class
        order = np.argsort(-mean_attr)[:5]
        out[name] = {
            "n_wrong": int(wrong.sum()), "n_total": int((y == c).sum()),
            "confused_with": {prep.class_names[k]: v for k, v in Counter(pred[wrong].tolist()).most_common(3)},
            "misleading_features": [(prep.feature_names[j], float(mean_attr[j])) for j in order],
        }
    return out


def train(Xtr, ytr, Xva, yva, n_cls, cfg, max_epochs):
    counts = np.bincount(ytr, minlength=n_cls)
    model = CNN1D(Xtr.shape[1], n_cls, **cfg["cnn"])
    loss = FocalLoss(class_balanced_weights(counts, cfg["focal"]["beta"]), cfg["focal"]["gamma"])
    info = fit(model, Xtr, ytr, Xva, yva, loss, max_epochs=max_epochs, patience=cfg["train"]["patience"],
               batch_size=cfg["train"]["batch_size"], lr=cfg["train"]["lr"])
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "wustl"], default="unsw")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ks", nargs="+", type=int, default=[5, 10, 15, 20, 25])
    ap.add_argument("--n-explain", type=int, default=2000, help="test flows explained (paper: 5000)")
    ap.add_argument("--explainer", choices=["shap", "deep", "gxi"], default="shap")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/explain")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]
    os.makedirs(args.out, exist_ok=True)
    ds = (load_unsw([cfg["data"]["unsw_train"], cfg["data"]["unsw_test"]]) if args.dataset == "unsw"
          else load_wustl(cfg["data"]["wustl"]))
    ds = clean(ds)
    set_seed(args.seed)
    prep = prepare_static(ds, args.task, args.seed, "xda")
    n_cls = len(prep.class_names)
    model, _ = train(prep.X_train, prep.y_train, prep.X_val, prep.y_val, n_cls, cfg, max_epochs)
    model = model.cpu().eval()

    rng = np.random.default_rng(args.seed)
    bg = prep.X_train[rng.choice(len(prep.X_train), cfg["shap"]["background"], replace=False)]
    idx = rng.choice(len(prep.X_test), min(args.n_explain, len(prep.X_test)), replace=False)
    attr, _ = Explainer(model, bg, args.explainer).attributions(prep.X_test[idx], seed=args.seed)
    ranking = global_ranking(attr, prep.feature_names)
    predict = lambda X: predict_proba(model, X).argmax(1)  # noqa: E731

    out = {
        "ranking": ranking,
        "fidelity": fidelity(predict, prep.X_test[idx], prep.y_test[idx], attr, seed=args.seed),
        "stability": stability(attr, k=10, seed=args.seed),
    }
    print("Top-10:", [n for n, _ in ranking[:10]])
    print(pd.DataFrame(out["fidelity"]).round(4).to_string(index=False))
    print("stability:", out["stability"])
    out["errors"] = explain_errors(attr, prep.y_test[idx], predict(prep.X_test[idx]), prep)
    for name, e in out["errors"].items():
        print(f"  {name}: {e['n_wrong']} wrong, confused with {e['confused_with']}, "
              f"misleading features {[f for f, _ in e['misleading_features'][:3]]}")
    top10 = {n for n, _ in ranking[:10]}
    out["artefact_features_in_top10"] = sorted(top10 & ARTEFACTS)
    if out["artefact_features_in_top10"]:
        print("  WARNING: known testbed-artefact features in top-10:", out["artefact_features_in_top10"])

    # SHAP-guided Top-k sweep: retrain on the k highest-ranked features.
    order = [prep.feature_names.index(n) for n, _ in ranking]
    sweep = []
    for k in args.ks + [len(order)]:
        cols = order[:k]
        set_seed(args.seed)
        mk, info = train(prep.X_train[:, cols], prep.y_train, prep.X_val[:, cols], prep.y_val, n_cls, cfg, max_epochs)
        m = detection_metrics(prep.y_test, predict_proba(mk, prep.X_test[:, cols]), prep.class_names,
                              prep.normal_index)
        e = efficiency(mk, k)
        sweep.append({"k": k, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "fpr": m["fpr"],
                      "train_seconds": info["train_seconds"], **e,
                      "features": [prep.feature_names[c] for c in cols]})
        print(f"  k={k:3d}  macroF1 {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"latency {e['latency_ms_per_flow']:.3f} ms")
    out["topk_sweep"] = sweep
    tag = f"{args.dataset}_{args.task}_s{args.seed}"
    json.dump(out, open(os.path.join(args.out, f"explain_{tag}.json"), "w"), indent=1)
    pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in sweep]).to_csv(
        os.path.join(args.out, f"topk_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
