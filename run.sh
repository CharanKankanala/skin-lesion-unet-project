#!/usr/bin/env bash
#
# One-command launcher for the skin lesion segmentation project.
#
# This script:
#   1) Creates a Python virtual environment (only on first run)
#   2) Installs all dependencies from requirements.txt (only on first run)
#   3) Downloads the trained model from Hugging Face (only on first run)
#   4) Runs inference on a sample dermoscopic image and saves the result
#
# Usage:
#     bash run.sh
#         # default: U-Net model, bundled sample image
#
#     bash run.sh --model attention_unet
#         # use the Attention U-Net checkpoint
#
#     bash run.sh --image path/to/your/image.jpg
#         # run on your own image

set -e  # exit on any error

VENV_DIR="venv"
PYTHON_BIN="python3"

echo "=========================================================="
echo "  Skin Lesion Segmentation - One-Command Launcher"
echo "=========================================================="

# ---- Step 1: virtual environment ----
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/3] Creating virtual environment in ./$VENV_DIR/ ..."
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "[1/3] Virtual environment already exists, reusing it."
fi

# Activate the venv (works on macOS/Linux)
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---- Step 2: dependencies ----
INSTALL_MARKER="$VENV_DIR/.deps_installed"
if [ ! -f "$INSTALL_MARKER" ]; then
    echo "[2/3] Installing dependencies (this can take a few minutes the first time)..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    touch "$INSTALL_MARKER"
else
    echo "[2/3] Dependencies already installed, skipping."
fi

# ---- Step 3: inference ----
echo "[3/3] Running inference..."
echo
python inference.py "$@"
