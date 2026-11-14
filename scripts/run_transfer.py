"""E5 cross-dataset transfer (UNSW-NB15 <-> CICIoT2023), binary task.

Only semantically equivalent features are kept (Step 3.8). The mapping is
approximate; check it against the CICIoT2023 feature description before reporting.
Reports in-domain vs cross-domain performance and whether the SHAP monitor fires.

    python scripts/run_transfer.py --ciciot "data/raw/ciciot2023/*.csv" --seed 1
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
from xda.data import Dataset, load_ciciot, load_unsw  # noqa: E402
from xda.explain import Explainer, ReferenceProfile  # noqa: E402
from xda.losses import FocalLoss, class_balanced_weights  # noqa: E402
from xda.metrics import detection_metrics  # noqa: E402
from xda.models import CNN1D  # noqa: E402
from xda.monitor import ShapDriftMonitor  # noqa: E402
from xda.preprocess import clean, prepare_static  # noqa: E402
from xda.train import fit, predict_proba, set_seed  # noqa: E402

SHARED = ["duration", "rate", "ttl", "mean_pkt", "tot_bytes", "proto", "service"]


def harmonise_unsw(ds: Dataset) -> Dataset:
    X = ds.X
    H = pd.DataFrame({
        "duration": X["dur"], "rate": X["rate"], "ttl": X["sttl"], "mean_pkt": X["smean"],
        "tot_bytes": X["sbytes"] + X["dbytes"],
        "proto": X["proto"].where(X["proto"].isin(["tcp", "udp", "icmp"]), "other"),
        "service": X["service"].where(X["service"].isin(["http", "dns", "ssh", "smtp"]), "other"),
    })
    return Dataset("unsw_h", H, ds.y_bin, ds.y_bin.map({0: "Normal", 1: "Attack"}), ["proto", "service"])


def harmonise_ciciot(ds: Dataset) -> Dataset:
    X = ds.X
    col = lambda name: X[name] if name in X.columns else 0  # noqa: E731
    proto = np.select([col("TCP") == 1, col("UDP") == 1, col("ICMP") == 1], ["tcp", "udp", "icmp"], "other")
    service = np.select([(col("HTTP") == 1) | (col("HTTPS") == 1), col("DNS") == 1, col("SSH") == 1,
                         col("SMTP") == 1], ["http", "dns", "ssh", "smtp"], "other")
    H = pd.DataFrame({
        "duration": X["flow_duration"], "rate": X["Rate"],
        "ttl": X["Duration"],          # CICIoT2023 'Duration' is the packet TTL
        "mean_pkt": X["AVG"], "tot_bytes": X["Tot sum"], "proto": proto, "service": service,
    })
    return Dataset("ciciot_h", H, ds.y_bin, ds.y_bin.map({0: "Normal", 1: "Attack"}), ["proto", "service"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ciciot", required=True, help='glob of CICIoT2023 CSV parts, e.g. "data/raw/ciciot2023/*.csv"')
    ap.add_argument("--rows", type=int, default=500_000, help="stratified subsample of CICIoT2023")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/transfer")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    max_epochs = args.max_epochs or cfg["train"]["max_epochs"]
    os.makedirs(args.out, exist_ok=True)
    unsw = clean(harmonise_unsw(load_unsw([cfg["data"]["unsw_train"], cfg["data"]["unsw_test"]])))
    cic = clean(harmonise_ciciot(load_ciciot(sorted(glob.glob(args.ciciot)), args.rows, args.seed)))
    print(f"UNSW {len(unsw.X):,} rows, CICIoT2023 {len(cic.X):,} rows, shared features {SHARED}")

    results = {}
    for src, tgt in [(unsw, cic), (cic, unsw)]:
        name = f"{src.name}->{tgt.name}"
        print(f"\n### {name}")
        set_seed(args.seed)
        prep = prepare_static(src, "binary", args.seed, "xda")
        n_cls = len(prep.class_names)
        model = CNN1D(len(prep.feature_names), n_cls, **cfg["cnn"])
        counts = np.bincount(prep.y_train, minlength=n_cls)
        fit(model, prep.X_train, prep.y_train, prep.X_val, prep.y_val,
            FocalLoss(class_balanced_weights(counts, cfg["focal"]["beta"]), cfg["focal"]["gamma"]),
            max_epochs=max_epochs, patience=cfg["train"]["patience"], batch_size=cfg["train"]["batch_size"])
        model = model.cpu().eval()

        X_tgt = prep.transform.transform(tgt.X)
        y_tgt = np.array([prep.class_names.index(v) for v in tgt.y_multi])
        in_dom = detection_metrics(prep.y_test, predict_proba(model, prep.X_test), prep.class_names, prep.normal_index)
        cross = detection_metrics(y_tgt, predict_proba(model, X_tgt), prep.class_names, prep.normal_index)

        # Does the SHAP monitor notice the domain change? Calibrate on source, test target windows.
        rng = np.random.default_rng(args.seed)
        n = cfg["shap"]["window_sample"]
        expl = Explainer(model, prep.X_train[rng.choice(len(prep.X_train), cfg["shap"]["background"], replace=False)])
        perm = rng.permutation(len(prep.X_val))
        ref = ReferenceProfile.build(expl, prep.X_val[perm[:cfg["shap"]["reference_size"]]], k=5)
        mon = ShapDriftMonitor(ref, prep.feature_names, cfg["monitor"]["alpha"], k=5)
        pool = perm[cfg["shap"]["reference_size"]:]
        cal = []
        for _ in range(20):
            idx = rng.choice(pool, n, replace=False)
            a, p = expl.attributions(prep.X_val[idx])
            cal.append((a, prep.X_val[idx], p))
        mon.calibrate(cal, cfg["monitor"]["target_far"])
        alarms = []
        for _ in range(10):
            idx = rng.choice(len(X_tgt), n, replace=False)
            a, p = expl.attributions(X_tgt[idx])
            r = mon.check(a, X_tgt[idx], p)
            alarms.append({"alarm": r.alarm, "diagnosis": r.diagnosis, "top_shifted": r.shifted_features})

        results[name] = {"in_domain": {k: in_dom[k] for k in ("accuracy", "macro_f1", "fpr")},
                         "cross_domain": {k: cross[k] for k in ("accuracy", "macro_f1", "fpr")},
                         "monitor_alarm_rate": float(np.mean([a["alarm"] for a in alarms])),
                         "monitor_windows": alarms}
        print(json.dumps({k: v for k, v in results[name].items() if k != "monitor_windows"}, indent=1))
    json.dump(results, open(os.path.join(args.out, f"transfer_s{args.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
