# Skin Lesion Segmentation with U-Net Variants

**Group 18 — MSML640 Spring 2026**
*Siva Akash Ramini · Harshini Karella · Charan Kankanala*

This project compares a baseline **U-Net** with an **Attention U-Net** for binary
skin lesion segmentation on the **ISIC 2018 Task 1** dataset. Both models are
implemented from scratch in PyTorch and trained under the same pipeline so that
all comparisons are controlled.

The trained models live on Hugging Face and are downloaded automatically the
first time you run inference, so the project can be reproduced end-to-end on a
fresh machine without any retraining.

**Models:** https://huggingface.co/CharanKankanala2003/skin-lesion-unet-models

---

## Quick start (one command)

After unzipping or cloning, the entire project runs with **a single command**:

```bash
bash run.sh
```

This will (on first run):
1. Create a Python virtual environment in `./venv/`
2. Install all dependencies from `requirements.txt`
3. Download the trained U-Net checkpoint (~120 MB) from Hugging Face into `./models_cache/`
4. Run inference on the bundled `sample_image.jpg`
5. Save the predicted lesion mask and a visualization in `./inference_outputs/`

On every subsequent run, steps 1–3 are skipped (cached), so inference takes
just a few seconds.

### Other inference modes

```bash
# Run the Attention U-Net checkpoint instead of the default U-Net
bash run.sh --model attention_unet

# Run on your own dermoscopic image
bash run.sh --image path/to/your/image.jpg

# Combine both
bash run.sh --model attention_unet --image path/to/your/image.jpg
```

---

## Requirements

- **Python 3.10+** (tested on Python 3.12)
- **macOS / Linux** (the `run.sh` launcher is a bash script)
- **Internet** for the first run (to download the model from Hugging Face)
- *Optional:* CUDA GPU or Apple Silicon MPS — auto-detected. CPU works too,
  inference takes ~2 seconds per image either way.

If you are on Windows, the inference script itself works; just call it directly:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python inference.py
```

---

## Project structure

```
skin-lesion-unet-project/
├── run.sh                           # One-command launcher (recommended entry point)
├── inference.py                     # Single-image inference (HF model download + prediction)
├── main.py                          # Full training pipeline for all 10 experiments
├── sample_image.jpg                 # Bundled sample for the demo run
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset.py                   # SkinLesionDataset (ISIC images + masks)
│   ├── transforms.py                # Albumentations train/val transforms
│   ├── losses.py                    # Dice, BCE, Combined Dice + BCE
│   ├── metrics.py                   # Dice, IoU, Precision, Recall
│   ├── device.py                    # CUDA / MPS / CPU auto-selection
│   ├── models/
│   │   ├── unet.py                  # Baseline U-Net (custom implementation)
│   │   └── attention_unet.py        # Attention U-Net (custom implementation)
│   ├── train_unet.py                # Baseline training (explicit loop)
│   ├── train_attention_unet.py      # Attention training (explicit loop)
│   └── visualize.py                 # Training curves, summary table, model comparison
│
└── outputs/
    ├── results/<run_tag>.csv        # Per-epoch metrics for each of the 10 experiments
    └── figures/                     # Training-curve plots, summary, model comparison
```

---

## Reproducing the training (optional — takes ~13 hours)

> **You do NOT need to retrain to grade this project.** The trained checkpoints
> are on Hugging Face and `bash run.sh` downloads them automatically.

If you want to retrain everything from scratch:

1. Download the **ISIC 2018 Task 1** dataset and place it as:
   ```
   data/raw/images/   # ISIC_*.jpg dermoscopic images (~2594 files)
   data/raw/masks/    # ISIC_*_segmentation.png masks  (~2594 files)
   ```
2. Activate the venv (`source venv/bin/activate`)
3. Run:
   ```bash
   python main.py
   ```

This will run all 10 experiments described in the proposal:
- 6 main runs: 2 architectures × 3 losses (Dice / BCE / Combined Dice+BCE)
- 2 data-efficiency runs: 50% training subset, both architectures
- 2 augmentation-ablation runs: no augmentation, both architectures

After training, every `outputs/results/<run_tag>.csv` and
`outputs/figures/curves_<run_tag>.png` is regenerated, plus
`outputs/figures/summary_table.csv` and `outputs/figures/model_comparison.png`.

### Training configuration

| Setting | Value |
| --- | --- |
| Image size | 256 × 256 |
| Optimizer | Adam |
| Learning rate | 1e-3 (1e-4 for Attention U-Net + Dice loss) |
| Batch size | 8 |
| Epochs (per run) | 10 |
| Augmentation | horizontal flip, vertical flip, rotation (training only) |
| Train/val split | 80 / 20, fixed seed (42) |

---

## Headline results

The full per-run summary is in `outputs/figures/summary_table.csv`. Best
validation-Dice highlights:

| Run | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| **U-Net + Combined loss** (best overall) | **0.8334** | 0.7220 | 0.8611 | 0.8229 |
| U-Net + Dice loss | 0.8298 | 0.7182 | 0.8860 | 0.7969 |
| U-Net + BCE loss | 0.8166 | 0.6995 | 0.8890 | 0.7698 |
| **Attention U-Net + Dice loss** (best Attention) | **0.8153** | 0.6926 | 0.8025 | 0.8460 |
| U-Net (no augmentation) | 0.8151 | 0.6992 | 0.9016 | 0.7610 |
| U-Net (50 % training data) | 0.8035 | 0.6820 | 0.8470 | 0.7840 |

---

## Evaluation metrics

- **Dice coefficient** — primary overlap metric.
- **Intersection over Union (IoU)** — secondary overlap metric.
- **Precision** — fraction of predicted lesion pixels that are correct.
- **Recall** — fraction of true lesion pixels that are recovered.

All four are computed on the validation split after every epoch and saved per
run as `outputs/results/<run_tag>.csv`.
