# Skin Lesion Segmentation with U-Net Variants

Group 18 — MSML640 Spring 2026

## Overview

This project implements and compares two U-Net-style architectures for
binary skin lesion segmentation on the ISIC 2018 dataset:

- **Baseline U-Net** — classical encoder–decoder with skip connections.
- **Attention U-Net** — same backbone with attention gates on every skip
  connection, allowing the decoder to focus on lesion regions and suppress
  irrelevant background features.

Both models are implemented from scratch in PyTorch and trained under the
same pipeline so that comparisons are controlled.

The four experiments described in the project proposal are all wired into
`main.py`:

1. **Architecture comparison** — U-Net vs Attention U-Net.
2. **Loss function study** — Dice, BCE, and Combined Dice + BCE.
3. **Data efficiency study** — full training set vs 50 % subset.
4. **Augmentation study** — with augmentation vs without.

Each run reports four metrics on the validation set: Dice, IoU, Precision,
and Recall.

## Project Structure

```
skin-lesion-unet-project/
├── main.py                          # Runs all experiments + visualizations
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── dataset.py                   # SkinLesionDataset (ISIC images + masks)
│   ├── transforms.py                # Albumentations train/val transforms
│   ├── losses.py                    # Dice, BCE, Combined Dice+BCE
│   ├── metrics.py                   # Dice, IoU, Precision, Recall
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                  # Baseline U-Net (custom implementation)
│   │   └── attention_unet.py        # Attention U-Net (custom implementation)
│   ├── train_unet.py                # Baseline U-Net training (explicit loop)
│   ├── train_attention_unet.py      # Attention U-Net training (explicit loop)
│   └── visualize.py                 # Training curves + prediction comparisons
└── outputs/
    ├── checkpoints/<run_tag>/       # epoch01.pth ... epochN.pth + best_model.pth
    ├── results/<run_tag>.csv        # per-epoch metrics
    └── figures/                     # training curves, summary table, comparison
```

## Dataset

Download the ISIC 2018 Task 1 segmentation dataset and place it as:

```
data/raw/images/   # *.jpg dermoscopic images
data/raw/masks/    # *_segmentation.png ground-truth masks
```

The dataset is split 80 % train / 20 % validation with a fixed seed (42).

## Setup

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full set of proposal experiments (10 runs total) and generate
all figures:

```bash
python main.py
```

Useful flags:

```bash
python main.py --epochs 10                  # change epoch count
python main.py --skip-data-efficiency       # skip the 50% study
python main.py --skip-aug-study             # skip the no-aug study
python main.py --skip-train                 # only re-generate figures
python main.py --skip-viz                   # only run training
```

Run a single experiment from Python:

```python
from src.train_unet import train_unet
from src.train_attention_unet import train_attention_unet

train_unet(loss_name="dice", num_epochs=10)
train_attention_unet(loss_name="combined", num_epochs=10, train_fraction=0.5)
```

## Training Configuration

| Setting | Value |
| --- | --- |
| Image size | 256 × 256 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 8 |
| Epochs (per run) | 10 |
| Augmentation | horizontal flip, vertical flip, rotation (training only) |
| Train/val split | 80 / 20 (seed 42) |

## Run Tags

Every run is given a tag that uniquely identifies its configuration:

| Tag | Architecture | Loss | Data | Augmentation |
| --- | --- | --- | --- | --- |
| `unet_dice` | U-Net | Dice | 100 % | yes |
| `unet_bce` | U-Net | BCE | 100 % | yes |
| `unet_combined` | U-Net | Dice + BCE | 100 % | yes |
| `attention_unet_dice` | Attention U-Net | Dice | 100 % | yes |
| `attention_unet_bce` | Attention U-Net | BCE | 100 % | yes |
| `attention_unet_combined` | Attention U-Net | Dice + BCE | 100 % | yes |
| `unet_combined_frac50` | U-Net | Dice + BCE | 50 % | yes |
| `attention_unet_combined_frac50` | Attention U-Net | Dice + BCE | 50 % | yes |
| `unet_combined_noaug` | U-Net | Dice + BCE | 100 % | no |
| `attention_unet_combined_noaug` | Attention U-Net | Dice + BCE | 100 % | no |

## Outputs

Each call to `train_unet(...)` or `train_attention_unet(...)` produces:

- `outputs/checkpoints/<run_tag>/epoch01.pth` … `epoch10.pth`
- `outputs/checkpoints/<run_tag>/best_model.pth` (highest validation Dice)
- `outputs/results/<run_tag>.csv` with columns
  `epoch, train_loss, val_loss, dice, iou, precision, recall`

Visualizations:

- `outputs/figures/curves_<run_tag>.png` — per-run training curves
  (loss / Dice / IoU / Precision+Recall).
- `outputs/figures/summary_table.csv` — best-Dice epoch from every run.
- `outputs/figures/model_comparison.png` — input | ground-truth |
  U-Net prediction | Attention U-Net prediction.

## Evaluation Metrics

- **Dice coefficient** — primary overlap metric.
- **Intersection over Union (IoU)** — secondary overlap metric.
- **Precision** — fraction of predicted lesion pixels that are correct
  (low = over-segmenting).
- **Recall** — fraction of true lesion pixels that are recovered
  (low = under-segmenting).

All four are computed on the validation split after every epoch.
