@echo off
REM One-time setup of XDA-IDS on a Windows PC with an NVIDIA GPU (e.g. GTX 1650 Super).
REM Needs: Python 3.11 or 3.12 from python.org, and a recent NVIDIA driver (check with nvidia-smi).
REM Usage: double-click, or run  setup_gpu.bat  from this folder.

cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment .venv ...
    py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Could not create a virtual environment. Install Python 3.11 from python.org first.
        exit /b 1
    )
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support ...
pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo CUDA 12.8 build not available, trying CUDA 12.6 ...
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    if errorlevel 1 exit /b 1
)

echo Installing the other libraries ...
pip install -r requirements.txt
if errorlevel 1 exit /b 1

python scripts\check_gpu.py
echo.
echo Setup finished. Next:  .venv\Scripts\activate  then  python run_all.py --quick
