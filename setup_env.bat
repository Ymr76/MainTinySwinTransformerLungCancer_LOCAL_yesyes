@echo off
:: ============================================================
::  PORTABLE INSTALLER
::  Run this once on any new device to set up the environment
:: ============================================================
echo Setting up environment in %~dp0

:: Create python venv if it doesn't exist
if not exist "%~dp0venv" (
    echo [1/4] Creating new virtual environment...
    py -3 -m venv "%~dp0venv"
)

set VENV_PIP="%~dp0venv\Scripts\pip.exe"
set VENV_PY="%~dp0venv\Scripts\python.exe"

echo.
echo [2/4] Installing standard requirements...
%VENV_PY% -m pip install --upgrade pip --quiet
%VENV_PIP% install -r "%~dp0requirements.txt" --quiet

echo.
echo [3/4] Installing PyTorch 2.5.1 + CUDA 12.1...
echo       (This is a massive download, ensure VPN is off if it gets stuck)
echo.
%VENV_PIP% install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/4] Registering Jupyter Kernel...
%VENV_PY% -m ipykernel install --user --name swin-lung --display-name "Python (swin-lung RTX3050)"

echo.
echo ============================================================
echo DONE! You can now open the notebook and run it.
:: Remind the user about the Kaggle Token
echo Note: If this is a new PC, make sure you edit .kaggle/kaggle.json 
echo       to include your Kaggle API key!
echo ============================================================
pause
