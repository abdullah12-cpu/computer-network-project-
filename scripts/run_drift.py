"""E3 (unseen attack family), E4 (injected drift) and natural-order streams.

Every stream is scored for all monitors:
  label-free : xda (SHAP alarm or input alarm, with diagnosis), shap (SHAP signals only),
               shap_ks_only (ablation), input_ks, confidence_ks, kswin
  label-based: adwin, ddm (on the per-flow error stream; upper bound)
  online     : online_knn (SAM-KNN, prequential with true labels)

    python scripts/run_drift.py --family Exploits DoS Reconnaissance Fuzzers Backdoor --seed 1
    python scripts/run_drift.py --drift sudden gradual incremental recurring --class-mix DoS --seed 1
    python scripts/run_drift.py --dataset wustl --drift sudden --chrono --seed 1
    python scripts/run_drift.py --family Exploits --adapt --seed 1     # with retraining loop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from xda.baselines import OnlineKNN  # noqa: E402
from xda.data import NORMAL, load_unsw, load_wustl  # noqa: E402
from xda.explain import Explainer, ReferenceProfile  # noqa: E402
from xda.losses import FocalLoss, class_balanced_weights  # noqa: E402
from xda.models import CNN1D  # noqa: E402
from xda.monitor import (ConfidenceKSMonitor, ErrorRateMonitor, InputKSMonitor,  # noqa: E402
                         KSWINMonitor, ShapDriftMonitor, diagnosis_accuracy, drift_scores,
                         localisation)
from xda.preprocess import clean, leave_family_out, prepare_static  # noqa: E402
from xda.streams import (chronological_stream, class_mix_stream, novel_attack_stream,  # noqa: E402
                         scale_drift_stream)
from xda.train import expand_head, fine_tune, fit, predict_proba, set_seed  # noqa: E402


def focal_loss(y, n_cls, cfg):
    counts = np.bincount(y, minlength=n_cls)
    return FocalLoss(class_balanced_weights(counts, cfg["focal"]["beta"]), cfg["focal"]["gamma"])


def train_detector(prep, cfg, max_epochs):
    n_cls = len(prep.class_names)
    model = CNN1D(len(prep.feature_names), n_cls, **cfg["cnn"])
    fit(model, prep.X_train, prep.y_train, prep.X_val, prep.y_val, focal_loss(prep.y_train, n_cls, cfg),
        max_epochs=max_epochs, patience=cfg["train"]["patience"], batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"])
    return model.cpu().eval()


class Context:
    """Detector + explainer + reference profile + calibrated label-free monitors."""

    def __init__(self, model, prep, cfg, seed, method, sample):
        self.model, self.prep, self.cfg, self.seed, self.method, self.sample = model, prep, cfg, seed, method, sample
        self.rng = np.random.default_rng(seed)
        sc, mc = cfg["shap"], cfg["monitor"]
        bg = prep.X_train[self.rng.choice(len(prep.X_train), sc["background"], replace=False)]
        self.expl = Explainer(model, bg, method, sc.get("nsamples", 50))

        # Disjoint validation flows for the reference profile and for calibration windows.
        perm = self.rng.permutation(len(prep.X_val))
        ref_idx, self.cal_pool = perm[:sc["reference_size"]], perm[sc["reference_size"]:]
        self.ref = ReferenceProfile.build(self.expl, prep.X_val[ref_idx])
        cal = []
        for _ in range(20):
            idx = self.rng.choice(self.cal_pool, min(sample, len(self.cal_pool)), replace=False)
            attr, pred = self.expl.attributions(prep.X_val[idx], seed=seed)
            cal.append((attr, prep.X_val[idx], pred, predict_proba(model, prep.X_val[idx]).max(1)))

        self.shap_mon = ShapDriftMonitor(self.ref, prep.feature_names, mc["alpha"], mode="full")
        self.shap_mon.calibrate([(a, x, p) for a, x, p, _ in cal], mc["target_far"])
        self.ks_only = ShapDriftMonitor(self.ref, prep.feature_names, mc["alpha"], mode="ks_only")
        self.ks_only.calibrate([(a, x, p) for a, x, p, _ in cal], mc["target_far"])
        self.in_mon = InputKSMonitor(self.ref.inputs, mc["alpha"])
        self.in_mon.calibrate([x for _, x, _, _ in cal], mc["target_far"])
        self.conf_mon = ConfidenceKSMonitor(predict_proba(model, self.ref.inputs).max(1), mc["alpha"])
        self.conf_mon.calibrate([c for _, _, _, c in cal], mc["target_far"])


def run_stream(stream, ctx: Context, adapt: bool):
    prep, model, rng = ctx.prep, ctx.model, ctx.rng
    class_names = list(prep.class_names)
    normal = prep.normal_index
    names = prep.feature_names

    # Stateful river detectors are created fresh for each stream.
    kswin = KSWINMonitor(len(names), seed=ctx.seed)
    kswin.warm_up(ctx.ref.inputs)
    adwin, ddm = ErrorRateMonitor("adwin"), ErrorRateMonitor("ddm")
    knn = OnlineKNN()
    warm = rng.choice(len(prep.X_train), 2000, replace=False)
    knn.learn(prep.X_train[warm], [class_names[i] for i in prep.y_train[warm]])

    alarms = {k: [] for k in ["xda", "shap", "shap_ks_only", "input_ks", "confidence_ks", "kswin", "adwin", "ddm"]}
    results, log, adapted_at = [], [], None
    for i, raw in enumerate(stream.windows):
        X = prep.transform.transform(raw)
        y_str = stream.labels[i]
        is_attack = y_str != NORMAL
        proba = predict_proba(model, X)
        pred = proba.argmax(1)
        flagged = pred != normal
        errors = flagged != is_attack

        s = rng.choice(len(X), min(ctx.sample, len(X)), replace=False)
        tracemalloc.start()
        t0 = time.perf_counter()
        attr, pred_s = ctx.expl.attributions(X[s], seed=ctx.seed)
        t_shap = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = ctx.shap_mon.check(attr, X[s], pred_s)
        t_mon = time.perf_counter() - t0
        mem_mb = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        results.append(r)

        alarms["xda"].append(r.alarm)
        alarms["shap"].append(r.shap_alarm)
        alarms["shap_ks_only"].append(ctx.ks_only.check(attr, X[s], pred_s).shap_alarm)
        alarms["input_ks"].append(ctx.in_mon.check(X[s]))
        alarms["confidence_ks"].append(ctx.conf_mon.check(proba[s].max(1)))   # same sample size as calibration
        alarms["kswin"].append(kswin.check(X[s]))
        alarms["adwin"].append(adwin.check(errors))
        alarms["ddm"].append(ddm.check(errors))
        knn_pred = knn.predict_then_learn(X[s], list(y_str[s]))
        knn_flag = np.array([p is not None and p != NORMAL for p in knn_pred])

        log.append({
            "window": i, "drifted": i >= stream.onset, "diagnosis": r.diagnosis,
            "alarms": {k: bool(v[-1]) for k, v in alarms.items()},
            "n_shap_shifted": r.stats.n_shap_shifted, "n_input_shifted": r.stats.n_input_shifted,
            "jaccard": r.stats.jaccard, "share_shift": r.stats.share_shift,
            "top_shifted": r.shifted_features, "signals": r.signals,
            "localisation": localisation(r.shifted_features, stream.perturbed),
            # Detection quality in this window (ground truth used for scoring only).
            "detector_acc_binary": float((flagged == is_attack).mean()),
            "detector_attack_recall": float(flagged[is_attack].mean()) if is_attack.any() else None,
            "detector_fpr": float(flagged[~is_attack].mean()) if (~is_attack).any() else None,
            "online_knn_acc_binary": float((knn_flag == is_attack[s]).mean()),
            "shap_seconds": t_shap, "monitor_seconds": t_mon, "monitor_peak_mb": mem_mb,
        })
        print(f"  w{i:02d} {'DRIFT' if i >= stream.onset else 'clean'}  xda={r.alarm!s:5} ({r.diagnosis:7})"
              f"  shap={r.shap_alarm!s:5} inKS={alarms['input_ks'][-1]!s:5} kswin={alarms['kswin'][-1]!s:5}"
              f"  adwin={alarms['adwin'][-1]!s:5}  det_acc={log[-1]['detector_acc_binary']:.3f}", flush=True)

        # Retraining loop: the first XDA alarm after onset is confirmed by an analyst who
        # labels this window; fine-tune on it plus a replay sample, then rebuild the profile.
        if adapt and adapted_at is None and i >= stream.onset and r.alarm:
            adapted_at = i
            new = sorted(set(y_str) - set(class_names))
            if new:
                expand_head(model, len(new))
                class_names += new
            y_idx = np.array([class_names.index(v) for v in y_str])
            replay = rng.choice(len(prep.X_train), len(X), replace=False)
            Xa = np.concatenate([X, prep.X_train[replay]])
            ya = np.concatenate([y_idx, prep.y_train[replay]])
            fine_tune(model, Xa, ya, focal_loss(ya, len(class_names), ctx.cfg))
            model.cpu().eval()
            ctx.expl = Explainer(model, ctx.expl.background.numpy(), ctx.method, ctx.expl.nsamples)
            ref_X = np.concatenate([ctx.ref.inputs[: len(ctx.ref.inputs) // 2], X[s][: len(ctx.ref.inputs) // 2]])
            ctx.shap_mon.ref = ReferenceProfile.build(ctx.expl, ref_X)
            ctx.shap_mon.ref_share = np.abs(ctx.shap_mon.ref.shap).mean(0) / np.abs(ctx.shap_mon.ref.shap).mean(0).sum()
            print(f"  >> retrained at window {i} (new classes: {new or 'none'})")

    scores = {k: drift_scores(v, stream.onset) for k, v in alarms.items()}
    post = log[stream.onset:]
    summary = {
        "diagnosis_accuracy": diagnosis_accuracy(results, stream.onset, stream.expected),
        "localisation_mean": (float(np.mean([w["localisation"] for w in post if w["alarms"]["xda"]]))
                              if stream.perturbed and any(w["alarms"]["xda"] for w in post) else None),
        "shap_seconds_per_window": float(np.mean([w["shap_seconds"] for w in log])),
        "monitor_seconds_per_window": float(np.mean([w["monitor_seconds"] for w in log])),
        "monitor_peak_mb": float(np.max([w["monitor_peak_mb"] for w in log])),
        "detector_acc_pre": float(np.mean([w["detector_acc_binary"] for w in log[:stream.onset]] or [np.nan])),
        "detector_acc_post": float(np.mean([w["detector_acc_binary"] for w in post] or [np.nan])),
        "online_knn_acc_post": float(np.mean([w["online_knn_acc_binary"] for w in post] or [np.nan])),
        "adapted_at": adapted_at,
        "detector_acc_after_adapt": (float(np.mean([w["detector_acc_binary"] for w in log[adapted_at + 1:]]))
                                     if adapted_at is not None and adapted_at + 1 < len(log) else None),
        "online_learner": knn.kind,
    }
    return scores, summary, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["unsw", "wustl"], default="unsw")
    ap.add_argument("--family", nargs="*", default=[], help="held-out families for E3")
    ap.add_argument("--drift", nargs="*", default=[], help="sudden gradual incremental recurring (E4)")
    ap.add_argument("--class-mix", nargs="*", default=[], help="class-mixture drift targets (E4), e.g. DoS")
    ap.add_argument("--chrono", action="store_true", help="natural time-ordered stream (WUSTL)")
    ap.add_argument("--adapt", action="store_true", help="enable the retraining loop")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--explainer", choices=["shap", "deep", "gxi"], default="shap")
    ap.add_argument("--window", type=int, default=None, help="flows per window (sensitivity sweep)")
    ap.add_argument("--sample", type=int, default=None, help="SHAP sample per window (sensitivity sweep)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/drift")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]
    win = args.window or cfg["monitor"]["window"]
    sample = args.sample or cfg["shap"]["window_sample"]
    os.makedirs(args.out, exist_ok=True)
    if args.dataset == "unsw":
        ds = clean(load_unsw([cfg["data"]["unsw_train"], cfg["data"]["unsw_test"]]))
    else:
        ds = clean(load_wustl(cfg["data"]["wustl"]))
    tag_extra = f"_w{win}_n{sample}" if (args.window or args.sample) else ""
    tag_extra += "_adapt" if args.adapt else ""

    def path_for(name):
        return f"{args.out}/{name}_s{args.seed}{tag_extra}.json"

    def save(name, scores, summary, log):
        path = path_for(name)
        json.dump({"scores": scores, "summary": summary, "log": log,
                   "config": {"dataset": args.dataset, "window": win, "sample": sample,
                              "explainer": args.explainer, "seed": args.seed}}, open(path, "w"), indent=1)
        print(json.dumps({k: (v["delay_windows"], round(v["false_alarm_rate"], 3)) for k, v in scores.items()}))
        print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=1))

    for fam in args.family:
        if os.path.exists(path_for(f"E3_{args.dataset}_{fam}")):
            print(f"skip E3 {fam} (already done)")
            continue
        print(f"\n### E3 {args.dataset} leave-out family = {fam}, seed {args.seed}")
        set_seed(args.seed)
        prep, known_test, novel = leave_family_out(ds, fam, args.seed)
        model = train_detector(prep, cfg, max_epochs)
        ctx = Context(model, prep, cfg, args.seed, args.explainer, sample)
        stream = novel_attack_stream(known_test, novel, win, seed=args.seed)
        save(f"E3_{args.dataset}_{fam}", *run_stream(stream, ctx, args.adapt))

    kinds = list(args.drift) + [f"class_mix_{t}" for t in args.class_mix] + (["chronological"] if args.chrono else [])
    todo = [k for k in kinds if not os.path.exists(path_for(f"E4_{args.dataset}_{k}"))]
    if kinds and not todo:
        print("all E4 streams already done")
    if todo:
        args.drift = [k for k in args.drift if k in todo]
        args.class_mix = [t for t in args.class_mix if f"class_mix_{t}" in todo]
        args.chrono = args.chrono and "chronological" in todo
        set_seed(args.seed)
        prep = prepare_static(ds, "multiclass", args.seed, "xda")
        model = train_detector(prep, cfg, max_epochs)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        test = ds.subset(np.isin(np.arange(len(ds.X)), prep.idx_test))
        streams = ([scale_drift_stream(test, k, window=win, seed=args.seed) for k in args.drift] +
                   [class_mix_stream(test, t, window=win, seed=args.seed) for t in args.class_mix])
        if args.chrono:
            streams.append(chronological_stream(test, window=win))
        for stream in streams:
            print(f"\n### E4 {args.dataset} drift = {stream.kind}, seed {args.seed}")
            model.load_state_dict(state)          # each stream starts from the same detector
            ctx = Context(model, prep, cfg, args.seed, args.explainer, sample)
            save(f"E4_{args.dataset}_{stream.kind}", *run_stream(stream, ctx, args.adapt))


if __name__ == "__main__":
    main()
