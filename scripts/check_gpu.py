"""Verify the local install: libraries import, the GPU is visible, and a CUDA op runs.

    python scripts/check_gpu.py
"""
from __future__ import annotations

import importlib
import sys

print(f"Python {sys.version.split()[0]}")
missing = []
for name in ["numpy", "pandas", "scipy", "sklearn", "yaml", "shap", "xgboost", "river",
             "statsmodels", "scikit_posthocs", "matplotlib", "seaborn", "kagglehub"]:
    try:
        mod = importlib.import_module(name)
        print(f"  ok  {name:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:   # report every missing library, not just the first
        missing.append(name)
        print(f"  MISSING {name}: {e}")

import torch  # noqa: E402

print(f"\ntorch {torch.__version__}, built for CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, compute capability {p.major}.{p.minor}")
    x = torch.rand(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    print("CUDA test op:", float((x @ x).mean()) > 0)
else:
    print("WARNING: CUDA not available. Training will run on the CPU.")
    print("  Check that 'nvidia-smi' works and that torch was installed from the CUDA index,")
    print("  e.g. pip install torch --index-url https://download.pytorch.org/whl/cu128")

if missing:
    sys.exit(f"\nMissing libraries: {missing}. Run: pip install -r requirements.txt")
print("\nAll good.")
