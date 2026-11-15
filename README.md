# XDA-IDS: Explainable, Drift-Aware Lightweight IDS for IoT/IIoT

Phase 3 implementation of the methodology. It extends the base paper: Udurume, Shakhov & Koo, *Scientific Reports* 16:32576, 2026 (https://doi.org/10.1038/s41598-025-34334-3; code at github.com/Mi12Mi12Cu/Explainable-IoT-IDS).

## Quick start (local NVIDIA GPU, e.g. GTX 1650 Super or GTX 1660 Ti)
Requirements: Python 3.11 (on Windows tick "Add python.exe to PATH"), a recent NVIDIA driver (`nvidia-smi` works) and internet access for the first run. `requirements.txt` pins pandas below 3; the reported runs used pandas 2.3.3.

Windows (Command Prompt), inside this folder:
```
setup_gpu.bat
.venv\Scripts\activate
python scripts\smoke_test.py
python run_all.py --quick
python run_all.py
```
Linux:
```
bash setup_gpu.sh
source .venv/bin/activate
python scripts/smoke_test.py
python run_all.py --quick
python run_all.py
```
1. `setup_gpu` creates `.venv`, installs the CUDA build of PyTorch and `requirements.txt`, then runs `scripts/check_gpu.py`.
2. `smoke_test.py` runs every script on synthetic data in a few minutes, as a check before the long runs.
3. `run_all.py --quick` does a first pass on the real data (1 seed, 10 epochs) and writes to `results_quick/`.
4. `run_all.py` runs the full protocol (5 seeds, early stopping) and writes to `results/`.

Measured on a GTX 1660 Ti (Ryzen 5 3600, 16 GB RAM): the smoke test takes about 11 minutes and `run_all.py --quick` about 2.7 hours. The full protocol is estimated at more than 12 hours.

Datasets are never downloaded by hand: `kagglehub` fetches them on first use and caches them (about 750 MB).
Finished runs are skipped, so an interrupted run resumes when you run the same command again.
`python run_all.py --list` shows the stages; `--stages static e3` runs only some of them.

## Phase 3 results (quick run)
`results_quick/` holds the run used in the Phase 3 report (`docs/`): 1 seed, at most 10 epochs, batch size 512, learning rate 2e-3 (`configs/default.yaml`; the base paper used 128 and 1e-3). Headline numbers, from `results_quick/comparison/comparison.md`:

| | base_repro (with leaks) | base_clean (leak-free) | xda_full |
|---|---|---|---|
| UNSW binary, Macro-F1 | 0.985 | 0.899 | 0.897 |
| UNSW multiclass, Macro-F1 | 0.433 | 0.356 | 0.409 |
| UNSW multiclass, FPR | 0.021 | 0.145 | 0.264 |
| WUSTL multiclass, Macro-F1 | 0.934 | 0.947 | 0.940 |

- Focal loss raises UNSW recall for Worms, Analysis and Shellcode from 0 to 0.88, 0.80 and 0.45, but raises FPR.
- XGBoost is the strongest detector baseline (UNSW multiclass Macro-F1 0.584); XDA-IDS is 12x faster per flow (0.24 ms) and 22x smaller.
- The XDA monitor detects all 11 injected drift / unseen-family scenarios without labels (mean delay 0.27 windows, false-alarm rate 0.164); the SHAP-KS-only variant reaches a 0.009 false-alarm rate.
- Known weaknesses: the diagnosis rule labels scale drift as concept drift, and UNSW-to-CICIoT2023 transfer fails in both directions.

## Datasets (from Kaggle, see `configs/default.yaml`)
| Dataset | Used for | Kaggle source |
|---|---|---|
| UNSW-NB15 (training + testing set) | base reproduction, E1-E4 | `mrwellsdavid/unsw-nb15` |
| WUSTL-IIoT-2021 | base reproduction, E1, E2, E4 | `annaamalaiu/wustl-iiot-2021-dataset` (fallback `mostafa2372/wustl-iiot-2021`) |
| CICIoT2023 (1M-row sample) | E5 cross-dataset transfer | `ioannispan/ciciot2023-1m` (fallback `himadri07/ciciot2023`) |

**UNSW file names in the Kaggle mirror.** In `mrwellsdavid/unsw-nb15`, `UNSW_NB15_training-set.csv` has 82,332 rows and `UNSW_NB15_testing-set.csv` has 175,341, the reverse of the official release. The XDA protocol concatenates both, so it is unaffected, but `base_repro` (which loads `unsw_train` only) trains on 82,332 rows rather than the 175,341 the base paper used. To match the paper exactly, point `data.unsw_train.file` at `UNSW_NB15_testing-set.csv`.

If Kaggle asks for a login: `python -c "import kagglehub; kagglehub.login()"`.

## Stages of `run_all.py` (methodology steps)
| Stage | What it does | Output |
|---|---|---|
| `env` | libraries, GPU, versions | `logs/env.log` |
| `check` | class counts, duplicates, `id`/`Target` leakage checks (Step 3) | `data_check.txt` |
| `static` | XDA-IDS vs base pipeline + ablation, UNSW and WUSTL, binary and multiclass (E1, E2, Step 10) | `static/` |
| `baselines` | LSTM, Random Forest, XGBoost, RF + Isolation Forest (Step 6) | `static/` |
| `explain` | SHAP fidelity, stability, Top-k sweep, error explanations (Steps 4, 8, 11) | `explain/` |
| `e3` | unseen attack families, all monitors (E3) | `drift/` |
| `e4` | 4 drift types + class mixture (UNSW), drift + natural order (WUSTL) (E4) | `drift/` |
| `adapt` | retraining loop after an alarm (Step 5.5) | `drift_adapt/` |
| `sensitivity` | window and SHAP-sample size (Step 10) | `drift_sensitivity/` |
| `transfer` | UNSW <-> CICIoT2023 transfer (E5) | `transfer/` |
| `stats` | mean ± std, bootstrap CI, Wilcoxon / t-test, McNemar, Friedman + Nemenyi (Step 9) | `stats_report.txt` |
| `analyze` | confusions, monitor misses, false-alarm classes (Step 11) | `failure_analysis.txt` |
| `figures` | confusion matrices, recall, drift timelines, monitors, Top-k, fidelity (Step 12) | `figures/` |
| `compare` | side-by-side comparison with the base paper | `comparison/comparison.md` |

## What is saved (`results/`, or `results_quick/` for `--quick`)
- `static/<dataset>_<task>_<variant>_s<seed>.json`: all metrics, per-class scores, confusion matrix, latency
- `static/..._pred.npz`: per-flow predictions (used by McNemar's test)
- `static/models/*.pt`: trained CNN/LSTM weights
- `static/<dataset>_<task>_summary.csv`: mean ± std per variant
- `explain/`, `drift/`, `drift_adapt/`, `drift_sensitivity/`, `transfer/`: JSON per run (drift runs include per-window logs)
- `stats_report.txt`, `failure_analysis.txt`, `data_check.txt`, `figures/*.png`, `comparison/`
- `config_used.yaml` and `logs/<stage>.log` (complete terminal output)

## Code layout
```
docs/                      Phase 3 report (DOCX and PDF)
results_quick/             outputs of the Phase 3 quick run
run_all.py                 runs every stage in order
setup_gpu.bat / .sh        one-time install (CUDA PyTorch + requirements.txt)
configs/default.yaml       hyperparameters and dataset sources
reference/base_paper.yaml  base paper's reported numbers (used by compare)
reference/base_notebook_outputs.txt   what the base paper's own notebooks print
src/xda/data.py            loaders, Kaggle download                                  Step 2
src/xda/preprocess.py      cleaning, frequency encoding, scaling, Split A / Split B  Step 3
src/xda/models.py          1D-CNN (base-paper layer stack) and LSTM                  Step 5 Module A
src/xda/losses.py          class-balanced focal loss                                 Step 5 Module A
src/xda/train.py           training (early stop on Macro-F1), retraining loop        Step 5
src/xda/explain.py         SHAP explainer, reference profile, fidelity, stability    Steps 4, 5B, 8
src/xda/monitor.py         SHAP-drift monitor + KS, KSWIN, ADWIN, DDM baselines      Steps 5C, 6
src/xda/baselines.py       RF, XGBoost, RF + Isolation Forest, online KNN            Step 6
src/xda/streams.py         unseen-family, drift, class-mix, chronological streams    Steps 1, 7
src/xda/metrics.py         detection metrics and latency                             Step 8
src/xda/report.py          saves printed reports to files
scripts/                   one script per stage (see the table above)
```

## Notes on the design
- The models are PyTorch ports of the base paper's Keras layers.
- The base pipeline is reproduced twice:
  - `base_repro` copies the published notebooks. It uses the 175,341-row UNSW training CSV with the `id` column kept as a feature, a scaler fitted before the split, and WUSTL's `Target` label left in the inputs.
  - `base_clean` uses the same model with a leak-free protocol.
- The monitor alarms on SHAP signals (KS count, max Wasserstein, top-k Jaccard, overall and per predicted class) or on input KS. The diagnosis compares relative importance shares, because raw SHAP values move whenever inputs move. All thresholds are calibrated on drift-free validation windows, disjoint from the reference profile, to `monitor.target_far`.
- With 5 seeds, a two-sided Wilcoxon test cannot reach p < 0.05 (the minimum is 0.0625). `stats.py` therefore reports one-sided tests and McNemar.
- The installed `river` version may have no SAM-KNN; the online learner then falls back to a sliding-window KNN, and the results record which one was used.

## Findings from the base paper's own code
Its notebooks were run unchanged except for loading the data from Kaggle; the outputs are in `reference/base_notebook_outputs.txt`.
- The manuscript notebook prints exactly the paper's Fig. 4 and Fig. 6 confusion matrices, after **5 epochs and a single seed** (the 5-seed loop is commented out).
- **UNSW multiclass:** the same output gives **accuracy 0.81, macro-F1 0.41, weighted-F1 0.78**. The paper's Table 3 reports 94.1% / 94.0%.
- **UNSW binary:** accuracy 0.968 and macro-F1 0.96. The paper reports 97.5%.
- **Data handling:**
  - The data is the 175,341-row training partition, not 257,673 rows.
  - The `id` row index is used as an input (40 features instead of 39).
  - The MinMax scaler is fitted on all rows before the split.
- **WUSTL notebook:** a different CNN (GlobalMaxPooling, Dense 128, no BN/Dropout), StandardScaler, a 20% test split and 10 epochs. Its inputs keep `Target`, which is the binary label itself. It prints **accuracy 1.00**, while the paper reports 95.9%.
- The repo's `src/train.py` cannot run, because it references an undefined `vf1`.
