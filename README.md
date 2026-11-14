# XDA-IDS: Explainable, Drift-Aware Lightweight IDS for IoT/IIoT

Phase 3 implementation. It extends the base paper (Udurume, Shakhov & Koo, *Scientific Reports* 16:32576, 2026, https://doi.org/10.1038/s41598-025-34334-3), whose code is cloned in `baseline_repo/`.

## Layout (mapped to the Phase 2 methodology steps)
```
configs/default.yaml       all hyperparameters and dataset sources                 (all steps)
src/xda/data.py            loaders: UNSW-NB15, WUSTL-IIoT-2021, CICIoT2023, Kaggle   Step 2
src/xda/preprocess.py      cleaning, frequency encoding, scaling, Split A / Split B  Step 3
src/xda/models.py          1D-CNN (base-paper stack) and LSTM, PyTorch                Step 5 Module A
src/xda/losses.py          class-balanced focal loss                                  Step 5 Module A
src/xda/train.py           training (early stop on Macro-F1), head expansion,
                           fine-tuning for the retraining loop                        Step 5 Module A / 5.5
src/xda/explain.py         SHAP explainer, reference profile, fidelity, stability     Step 4, Module B, Step 8
src/xda/monitor.py         SHAP-drift monitor (KS + Wasserstein + Jaccard, per class,
                           diagnosis) + input-KS, confidence-KS, KSWIN, ADWIN, DDM    Module C, Step 6
src/xda/baselines.py       RF, XGBoost, RF + Isolation Forest hybrid, online SAM-KNN  Step 6
src/xda/streams.py         unseen-family, 4 drift types, class-mix, chronological     Step 1, Step 7
src/xda/metrics.py         Macro-F1, per-class recall, FPR, MCC, PR-AUC, latency      Step 8
scripts/check_data.py      dataset sanity and leakage checks                          Step 3
scripts/run_static.py      E1/E2, ablation, detector baselines                        Steps 6, 7, 10
scripts/run_explain.py     fidelity, stability, Top-k sweep, error explanations       Steps 4, 8, 10, 11
scripts/run_drift.py       E3, E4, chronological, retraining loop, window sweep, E6   Steps 5, 7, 8, 10
scripts/run_transfer.py    E5 cross-dataset transfer (UNSW <-> CICIoT2023)            Step 7
scripts/stats.py           mean +/- std, bootstrap CI, Wilcoxon/t-test, McNemar,
                           Friedman + Nemenyi                                         Step 9
scripts/analyze_errors.py  confusions, monitor misses, false-alarm classification     Step 11
scripts/smoke_test.py      end-to-end test on synthetic data
colab_run.ipynb            runs everything on Colab, section by section
```

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py          # checks everything runs, no data needed
```
TensorFlow has no Python 3.14 build, so the models are PyTorch ports of the paper's Keras layers.

## Datasets
By default `configs/default.yaml` fetches the datasets from Kaggle with `kagglehub`, so nothing has to be uploaded:
UNSW-NB15 from `mrwellsdavid/unsw-nb15`, and WUSTL-IIoT-2021 from `annaamalaiu/wustl-iiot-2021-dataset` (fallback `mostafa2372/wustl-iiot-2021`).
The base-paper notebooks in `baseline_repo/` have a first cell that does the same.
To use local copies instead, replace a config entry with a file path and place the files as below.

| Dataset | Needed for | Download | Files to keep |
|---|---|---|---|
| UNSW-NB15 | E1–E4 (start here) | https://research.unsw.edu.au/projects/unsw-nb15-dataset (or Kaggle mirror `mrwellsdavid/unsw-nb15`) | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` (about 45 MB total) |
| WUSTL-IIoT-2021 | E1, E2, E4 | https://www.cse.wustl.edu/~jain/iiot2/index.html | `wustl_iiot_2021.csv` |
| CICIoT2023 | E3, E5 (later) | https://www.unb.ca/cic/datasets/iotdataset-2023.html | 3–5 CSV parts only |

You do **not** need the UNSW pcaps or the raw `UNSW-NB15_1..4.csv` files to start.

Place files as:
```
data/raw/unsw/UNSW_NB15_training-set.csv
data/raw/unsw/UNSW_NB15_testing-set.csv
data/raw/wustl/wustl_iiot_2021.csv
data/raw/ciciot2023/*.csv
```

## Running
```bash
python scripts/check_data.py
python scripts/run_static.py  --dataset unsw --task multiclass                  # ablation, 5 seeds
python scripts/run_static.py  --dataset unsw --task multiclass --variants lstm rf xgb hybrid_rf_if
python scripts/run_explain.py --dataset unsw --task multiclass --seed 1         # fidelity, stability, Top-k
python scripts/run_drift.py   --family Exploits DoS Reconnaissance Fuzzers Backdoor --seed 1
python scripts/run_drift.py   --drift sudden gradual incremental recurring --class-mix DoS --seed 1
python scripts/run_drift.py   --dataset wustl --drift sudden --chrono --seed 1
python scripts/run_drift.py   --family Exploits --adapt --seed 1                # retraining loop
python scripts/run_transfer.py --ciciot "data/raw/ciciot2023/*.csv" --seed 1
python scripts/stats.py && python scripts/analyze_errors.py
```
For a quick first run, add `--seeds 1 --max-epochs 10` (or `--max-epochs 10` for the other scripts).

Notes on the design:
- The monitor raises its alarm from SHAP signals (KS count, max Wasserstein, top-k Jaccard) or from input KS. Diagnosis compares the relative importance share and the top-k set, not raw SHAP values, since raw SHAP values move whenever inputs move.
- All thresholds are calibrated on drift-free validation windows that are disjoint from the reference profile, to a target false-alarm rate (`monitor.target_far`).
- With 5 seeds, a two-sided Wilcoxon test cannot reach p < 0.05 (minimum 0.0625). `stats.py` reports one-sided tests plus McNemar; use more seeds if needed.

## Findings from the base paper's own code (baseline_repo/)
- `Enhancing_IDS_UNSW_Binary_Multi_Manuscript.ipynb` prints exactly the paper's Fig. 4 and Fig. 6 confusion matrices.
- That run trained for **5 epochs with one seed**; the 5-seed loop is commented out.
- In the same output, UNSW multiclass CNN scores **accuracy 0.81, macro-F1 0.41, weighted-F1 0.78**. The paper's Table 3 reports 94.1% / 94.0%.
- UNSW binary CNN: accuracy 0.968, macro-F1 0.96 (paper: 97.5%).
- The data was `UNSW_NB15.csv` with 175,341 rows (the training partition), not 257,673. The numeric inputs included the `id` row index (40 features rather than 39).
- The MinMax scaler was fitted on all rows before the split.
- The WUSTL notebook uses a different CNN (GlobalMaxPooling, Dense 128, no BN/Dropout), StandardScaler, a 20% test split and 10 epochs. Its inputs also kept `Target`, which is the binary label (leakage). `check_data.py` verifies this.
- `src/train.py` in the repo cannot run, because it references an undefined `vf1`.
