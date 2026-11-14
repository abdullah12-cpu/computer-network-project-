"""Run the whole XDA-IDS methodology locally on the real datasets (Kaggle), stage by stage.

    python run_all.py --quick                  # first pass: 1 seed, 10 epochs, fewer scenarios
    python run_all.py                          # full protocol (5 seeds, early stopping)
    python run_all.py --stages static e3       # only some stages
    python run_all.py --list                   # show the stages
    python run_all.py --stages compare         # base-paper comparison from saved results

Finished runs are skipped, so the command can be re-run after an interruption.
Results, logs and figures: results/ (quick runs: results_quick/)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FAMILIES = ["Exploits", "DoS", "Reconnaissance", "Fuzzers", "Backdoor"]
DRIFTS = ["sudden", "gradual", "incremental", "recurring"]


def stages(a) -> dict[str, list[list[str]]]:
    seeds = [str(s) for s in a.seeds]
    ep = ["--max-epochs", str(a.max_epochs)] if a.max_epochs else []
    fams = FAMILIES[:2] if a.quick else FAMILIES
    R = a.results
    S = {
        # Libraries, GPU and versions (saved in the log)
        "env": [["scripts/check_gpu.py"]],
        # Step 3: data sanity and leakage checks (downloads the datasets on first run)
        "check": [["scripts/check_data.py", "--report", f"{R}/data_check.txt"]],
        # Steps 5-7, 10: XDA-IDS detector vs base paper + ablation (E1, E2)
        "static": [["scripts/run_static.py", "--dataset", ds, "--task", task, "--seeds", *seeds, *ep,
                    "--out", f"{R}/static"]
                   for ds in ("unsw", "wustl") for task in ("multiclass", "binary")],
        # Step 6: detector baselines
        "baselines": [["scripts/run_static.py", "--dataset", ds, "--task", "multiclass",
                       "--variants", "lstm", "rf", "xgb", "hybrid_rf_if", "--seeds", *seeds, *ep,
                       "--out", f"{R}/static"]
                      for ds in ("unsw", "wustl")],
        # Steps 4, 8, 10, 11: explanation fidelity/stability, Top-k sweep, error explanations
        "explain": [["scripts/run_explain.py", "--dataset", ds, "--task", "multiclass", "--seed", seeds[0], *ep,
                     "--out", f"{R}/explain"]
                    for ds in ("unsw", "wustl")],
        # Step 7 E3: unseen attack families
        "e3": [["scripts/run_drift.py", "--family", *fams, "--seed", s, *ep, "--out", f"{R}/drift"] for s in seeds],
        # Step 7 E4: injected drift (4 types + class mixture) and WUSTL natural order
        "e4": ([["scripts/run_drift.py", "--dataset", "unsw", "--drift", *DRIFTS, "--class-mix", "DoS",
                 "--seed", s, *ep, "--out", f"{R}/drift"] for s in seeds] +
               [["scripts/run_drift.py", "--dataset", "wustl", "--drift", *DRIFTS, "--chrono",
                 "--seed", s, *ep, "--out", f"{R}/drift"] for s in seeds[:3]]),
        # Step 5.5: retraining loop
        "adapt": [["scripts/run_drift.py", "--family", "Exploits", "DoS", "Backdoor", "--adapt",
                   "--seed", seeds[0], "--out", f"{R}/drift_adapt", *ep]],
        # Step 10: window / SHAP-sample sensitivity
        "sensitivity": [["scripts/run_drift.py", "--drift", "sudden", "gradual", "--window", w, "--sample", n,
                         "--seed", seeds[0], "--out", f"{R}/drift_sensitivity", *ep]
                        for w, n in [("1000", "300"), ("10000", "300"), ("5000", "100"), ("5000", "1000")]],
        # Step 7 E5: cross-dataset transfer (CICIoT2023 from Kaggle, or --ciciot for local files)
        "transfer": [["scripts/run_transfer.py", "--seed", seeds[0], *ep, "--out", f"{R}/transfer",
                      *(["--ciciot", a.ciciot] if a.ciciot else [])]],
        # Steps 9, 11, 12: statistics, failure analysis, figures
        "stats": [["scripts/stats.py", "--static", f"{R}/static", "--drift", f"{R}/drift"]],
        "analyze": [["scripts/analyze_errors.py", "--static", f"{R}/static", "--drift", f"{R}/drift"]],
        "figures": [["scripts/make_figures.py", "--results", R, "--out", f"{R}/figures"]],
        # Comparison with the base paper (reference/base_paper.yaml)
        "compare": [["scripts/compare_base.py", "--results", R]],
    }
    if a.quick:   # keep the quick pass short
        S["sensitivity"] = []
    return S


def run(cmd: list[str], log) -> int:
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1")
    line = f"\n$ python {' '.join(cmd)}\n"
    print(line, end="", flush=True)
    log.write(line)
    proc = subprocess.Popen([sys.executable, *cmd], cwd=ROOT, env=env, text=True, encoding="utf-8",
                            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for out in proc.stdout:
        print(out, end="", flush=True)
        log.write(out)
    return proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=None, help="subset of stages to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="1 seed, 10 epochs, fewer scenarios")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="default 1 2 3 4 5 (quick: 1)")
    ap.add_argument("--max-epochs", type=int, default=None, help="cap training epochs (quick: 10)")
    ap.add_argument("--ciciot", default=None, help='local CICIoT2023 CSV glob for E5 (default: download from Kaggle)')
    ap.add_argument("--keep-going", action="store_true", help="continue with the next stage after a failure")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = [1] if a.quick else [1, 2, 3, 4, 5]
    if a.max_epochs is None and a.quick:
        a.max_epochs = 10
    # Quick runs go to their own folder so the full run never reuses 10-epoch results.
    a.results = "results_quick" if a.quick else "results"

    S = stages(a)
    if a.list:
        for name, cmds in S.items():
            print(f"{name:12s} {len(cmds)} command(s)")
        return
    chosen = a.stages or list(S)
    unknown = set(chosen) - set(S)
    if unknown:
        sys.exit(f"unknown stage(s): {sorted(unknown)}; choose from {list(S)}")

    os.makedirs(os.path.join(ROOT, a.results, "logs"), exist_ok=True)
    # Keep the exact configuration next to the results.
    shutil.copy(os.path.join(ROOT, "configs", "default.yaml"), os.path.join(ROOT, a.results, "config_used.yaml"))
    t_all = time.time()
    for name in chosen:
        if not S[name]:
            print(f"\n=== {name}: nothing to run")
            continue
        print(f"\n=================== stage: {name} ===================")
        t0 = time.time()
        with open(os.path.join(ROOT, a.results, "logs", f"{name}.log"), "a", encoding="utf-8") as log:
            for cmd in S[name]:
                code = run(cmd, log)
                if code != 0:
                    print(f"\n!!! stage '{name}' failed (exit {code}); see {a.results}/logs/{name}.log")
                    if not a.keep_going:
                        sys.exit(code)
                    break
        print(f"=== {name} done in {(time.time() - t0) / 60:.1f} min")
    print(f"\nAll requested stages finished in {(time.time() - t_all) / 3600:.2f} h. "
          f"Results and figures: {a.results}/")


if __name__ == "__main__":
    main()
