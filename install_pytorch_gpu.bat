@echo off
:: ============================================================
::  install_pytorch_gpu.bat
::  Run this ONCE with NordVPN DISABLED
::  Uses E:\ as pip cache so C:\ (full) is not touched
:: ============================================================
set PIP_CACHE_DIR=E:\Tiny swin transformer\pip_cache
set VENV_PIP=E:\Tiny swin transformer\venv\Scripts\pip.exe
set VENV_PY=E:\Tiny swin transformer\venv\Scripts\python.exe

echo.
echo  [1/3] Upgrading pip...
"%VENV_PY%" -m pip install --upgrade pip --cache-dir "%PIP_CACHE_DIR%"

echo.
echo  [2/3] Installing PyTorch 2.5.1 + CUDA 12.1  (~2.4 GB download)
echo        NordVPN must be DISABLED for this step!
echo.
"%VENV_PIP%" install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    --cache-dir "%PIP_CACHE_DIR%"

echo.
echo  [3/3] Verifying GPU...
"%VENV_PY%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo ============================================================
echo  Done! Select kernel "Python (swin-lung RTX3050)" in Jupyter.
echo ============================================================
pause
