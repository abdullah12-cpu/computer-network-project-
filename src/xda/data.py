"""Dataset loaders.

Each loader returns a DataFrame of features plus two label Series:
`y_bin` (0 = normal, 1 = attack) and `y_multi` (string class name).
Categorical columns are kept here; dropping or encoding them is a
preprocessing decision (see preprocess.py).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NORMAL = "Normal"


def resolve_all(spec: str | dict) -> list[str]:
    """Turn a config entry into local file paths.

    A string is a path or glob pattern. A dict {"kaggle": [datasets...], "file": name-or-pattern}
    downloads the first Kaggle dataset that contains matching files (cached by kagglehub).
    """
    if isinstance(spec, str):
        hits = sorted(glob.glob(spec))
        return hits or [spec]
    import kagglehub
    for ds in spec["kaggle"]:
        try:
            root = kagglehub.dataset_download(ds)
        except Exception as e:  # dataset missing or needs login: try the next mirror
            print(f"[kaggle] {ds}: {e}")
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", spec["file"]), recursive=True))
        if hits:
            return hits
        print(f"[kaggle] {ds}: {spec['file']} not found, files: {os.listdir(root)}")
    raise FileNotFoundError(spec["file"])


def resolve(spec: str | dict) -> str:
    return resolve_all(spec)[0]


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y_bin: pd.Series
    y_multi: pd.Series
    categorical: list[str] = field(default_factory=list)
    time: pd.Series | None = None      # flow start time, when the dataset has one

    def subset(self, mask) -> "Dataset":
        mask = np.asarray(mask)
        pick = (lambda s: s[mask].reset_index(drop=True))
        return Dataset(self.name, pick(self.X), pick(self.y_bin), pick(self.y_multi), self.categorical,
                       None if self.time is None else pick(self.time))


# ---------------------------------------------------------------- UNSW-NB15
UNSW_CATEGORICAL = ["proto", "service", "state"]
# 'id' is a row index, not a traffic feature. The base paper's notebooks kept it
# (40 numeric inputs instead of 39), which can leak class order.
UNSW_DROP = ["id"]


def load_unsw(paths: str | dict | list, keep_id: bool = False) -> Dataset:
    """Load one or more UNSW-NB15 partition CSVs (training-set / testing-set)."""
    paths = paths if isinstance(paths, list) else [paths]
    df = pd.concat([pd.read_csv(resolve(p), low_memory=False) for p in paths], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    cat = df["attack_cat"].fillna(NORMAL).astype(str).str.strip()
    # Raw CSVs spell some families differently from the partition CSVs.
    cat = cat.replace({"Backdoors": "Backdoor", "": NORMAL, "normal": NORMAL})
    y_bin = df["label"].astype(int)

    drop = ["attack_cat", "label"] + ([] if keep_id else UNSW_DROP)
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for c in UNSW_CATEGORICAL:
        X[c] = X[c].astype(str).str.strip()
    return Dataset("unsw", X, y_bin, cat, [c for c in UNSW_CATEGORICAL if c in X.columns])


# ---------------------------------------------------------- WUSTL-IIoT-2021
WUSTL_LABEL = "Traffic"
# Identifiers and the second label column. 'Target' is the binary label itself:
# the base paper's WUSTL notebook left it in the inputs (label leakage).
WUSTL_DROP = ["StartTime", "LastTime", "SrcAddr", "DstAddr", "Sport", "Dport",
              "sIpId", "dIpId", "Target"]


def load_wustl(path: str | dict, keep_leaky: bool = False) -> Dataset:
    df = pd.read_csv(resolve(path), low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    raw = df[WUSTL_LABEL].astype(str).str.strip()
    y_multi = raw.where(raw.str.lower() != "normal", NORMAL)
    y_bin = (y_multi != NORMAL).astype(int)
    time = pd.to_datetime(df["StartTime"], errors="coerce") if "StartTime" in df.columns else None

    drop = [WUSTL_LABEL] + ([] if keep_leaky else WUSTL_DROP)
    X = df.drop(columns=[c for c in drop if c in df.columns])
    if keep_leaky:   # base notebook kept only numeric columns, timestamps dropped as strings
        X = X.select_dtypes("number")
    categorical = [c for c in X.columns if X[c].dtype == object]
    for c in categorical:
        X[c] = X[c].astype(str).str.strip()
    return Dataset("wustl", X, y_bin, y_multi, categorical, time)


# --------------------------------------------------------------- CICIoT2023
def load_ciciot(paths: list[str], n_rows: int | None = 500_000, seed: int = 0) -> Dataset:
    """Load and stratified-subsample a few CICIoT2023 CSV parts.

    Labels are mapped to the 7 attack categories + Normal used in the paper.
    """
    df = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    label_col = next(c for c in df.columns if c.lower() == "label")
    fine = df[label_col].astype(str).str.strip()
    y_multi = fine.map(_ciciot_category)
    if n_rows is not None and len(df) > n_rows:
        frac = n_rows / len(df)
        idx = (df.groupby(y_multi, group_keys=False)
                 .apply(lambda g: g.sample(max(1, int(round(len(g) * frac))), random_state=seed))
                 .index)
        df, y_multi = df.loc[idx], y_multi.loc[idx]
    X = df.drop(columns=[label_col]).reset_index(drop=True)
    y_multi = y_multi.reset_index(drop=True)
    y_bin = (y_multi != NORMAL).astype(int)
    return Dataset("ciciot", X, y_bin, y_multi, [])


def _ciciot_category(label: str) -> str:
    l = label.upper()
    if l.startswith("BENIGN"):
        return NORMAL
    for prefix, cat in [("DDOS", "DDoS"), ("DOS", "DoS"), ("MIRAI", "Mirai"),
                        ("RECON", "Recon"), ("VULNERABILITYSCAN", "Recon"),
                        ("DICTIONARYBRUTEFORCE", "BruteForce"),
                        ("DNS_SPOOFING", "Spoofing"), ("MITM", "Spoofing")]:
        if l.startswith(prefix):
            return cat
    return "Web"   # SqlInjection, XSS, CommandInjection, Backdoor_Malware, Uploading_Attack, BrowserHijacking


def class_counts(ds: Dataset) -> pd.DataFrame:
    counts = ds.y_multi.value_counts()
    return pd.DataFrame({"count": counts, "share_%": (100 * counts / counts.sum()).round(3)})
