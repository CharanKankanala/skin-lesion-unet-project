@echo off
REM ============================================================
REM Skin Lesion Segmentation - One-Command Launcher (Windows)
REM
REM This script:
REM   1) Creates a Python virtual environment (only on first run)
REM   2) Installs all dependencies from requirements.txt (only on first run)
REM   3) Downloads the trained model from Hugging Face (only on first run)
REM   4) Runs inference on a sample dermoscopic image and saves the result
REM
REM Usage:
REM     run.bat
REM         default: U-Net model, bundled sample image
REM
REM     run.bat --model attention_unet
REM         use the Attention U-Net checkpoint
REM
REM     run.bat --image path\to\your\image.jpg
REM         run on your own image
REM ============================================================

setlocal enabledelayedexpansion

echo ==========================================================
echo   Skin Lesion Segmentation - One-Command Launcher
echo ==========================================================

REM ---- Step 1: virtual environment ----
if not exist "venv\" (
    echo [1/3] Creating virtual environment in .\venv\ ...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.10+ is installed and on your PATH.
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists, reusing it.
)

REM Activate the venv
call venv\Scripts\activate.bat

REM ---- Step 2: dependencies ----
if not exist "venv\.deps_installed" (
    echo [2/3] Installing dependencies (this can take a few minutes the first time)...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        exit /b 1
    )
    echo. > venv\.deps_installed
) else (
    echo [2/3] Dependencies already installed, skipping.
)

REM ---- Step 3: inference ----
echo [3/3] Running inference...
echo.
python inference.py %*

endlocal
