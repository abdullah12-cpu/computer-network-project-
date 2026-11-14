#!/usr/bin/env bash
# One-time setup of XDA-IDS on a Linux PC with an NVIDIA GPU (e.g. GTX 1650 Super).
# Needs: Python 3.11 or 3.12 and a recent NVIDIA driver (check with nvidia-smi).
# Usage: bash setup_gpu.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    PY=$(command -v python3.11 || command -v python3.12 || command -v python3)
    echo "Creating virtual environment with $PY ..."
    "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

echo "Installing PyTorch with CUDA support ..."
pip install torch --index-url https://download.pytorch.org/whl/cu128 \
  || pip install torch --index-url https://download.pytorch.org/whl/cu126

echo "Installing the other libraries ..."
pip install -r requirements.txt

python scripts/check_gpu.py
echo
echo "Setup finished. Next:  source .venv/bin/activate  then  python run_all.py --quick"
