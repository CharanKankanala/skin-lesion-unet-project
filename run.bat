@echo off
REM Skin Lesion Segmentation - Windows One-Command Launcher

echo.
echo Skin Lesion Segmentation - One-Command Launcher
echo.

if not exist "venv\" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python 3.10 or higher is installed and on your PATH.
        exit /b 1
    )
) else (
    echo [1/3] Virtual environment already exists, reusing it.
)

call venv\Scripts\activate.bat

if not exist "venv\.deps_installed" (
    echo [2/3] Installing dependencies. This can take a few minutes the first time...
    python -m pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        exit /b 1
    )
    echo done > venv\.deps_installed
) else (
    echo [2/3] Dependencies already installed, skipping.
)

echo [3/3] Running inference...
echo.
python inference.py %*
