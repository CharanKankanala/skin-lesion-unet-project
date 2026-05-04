"""
Visualization utilities for the project.

Three figure types are produced under ``outputs/figures/``:
  1) Per-run training curves: loss / Dice / IoU / Precision+Recall vs epoch
     for every results CSV in ``outputs/results/``.
  2) ``model_comparison.png`` -- input | ground-truth | U-Net pred | Attn pred
     for a few validation samples.
  3) ``summary_table.csv`` -- best-Dice row from every run, useful as the
     headline table for the report.
"""

import os
import csv
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import SkinLesionDataset
from src.transforms import get_val_transforms
from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.device import get_device


IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"
CHECKPOINT_ROOT = "outputs/checkpoints"
RESULTS_ROOT = "outputs/results"
FIGURES_ROOT = "outputs/figures"


def get_image_mask_paths(image_dir, mask_dir):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(".jpg")
    ])
    mask_paths = []
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        image_id = image_name.replace(".jpg", "")
        mask_name = image_id + "_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            raise FileNotFoundError(f"Mask not found for {image_name}")
    return image_paths, mask_paths


# ----------------------------------------------------------------------------
# Reading metrics CSVs
# ----------------------------------------------------------------------------
def _read_history(csv_path):
    history = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "dice": float(row["dice"]),
                "iou": float(row["iou"]),
                "precision": float(row.get("precision", 0.0)),
                "recall": float(row.get("recall", 0.0)),
            })
    return history


# ----------------------------------------------------------------------------
# Training curves (one figure per run)
# ----------------------------------------------------------------------------
def plot_training_curves():
    os.makedirs(FIGURES_ROOT, exist_ok=True)

    if not os.path.isdir(RESULTS_ROOT):
        print(f"  No results dir at {RESULTS_ROOT}, skipping curves.")
        return

    csv_files = sorted(f for f in os.listdir(RESULTS_ROOT) if f.endswith(".csv"))
    if not csv_files:
        print("  No results CSVs found, skipping curves.")
        return

    for csv_file in csv_files:
        run_name = csv_file.replace(".csv", "")
        history = _read_history(os.path.join(RESULTS_ROOT, csv_file))
        if not history:
            continue

        epochs = [h["epoch"] for h in history]

        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss"); axes[0].legend()

        axes[1].plot(epochs, [h["dice"] for h in history], color="tab:green")
        axes[1].set_title("Validation Dice"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice")

        axes[2].plot(epochs, [h["iou"] for h in history], color="tab:orange")
        axes[2].set_title("Validation IoU"); axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("IoU")

        axes[3].plot(epochs, [h["precision"] for h in history], label="precision")
        axes[3].plot(epochs, [h["recall"] for h in history], label="recall")
        axes[3].set_title("Precision / Recall"); axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Score"); axes[3].legend()

        fig.suptitle(run_name, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(FIGURES_ROOT, f"curves_{run_name}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


# ----------------------------------------------------------------------------
# Headline summary table: best-Dice epoch per run
# ----------------------------------------------------------------------------
def write_summary_table():
    os.makedirs(FIGURES_ROOT, exist_ok=True)

    if not os.path.isdir(RESULTS_ROOT):
        print(f"  No results dir at {RESULTS_ROOT}, skipping summary.")
        return

    csv_files = sorted(f for f in os.listdir(RESULTS_ROOT) if f.endswith(".csv"))
    if not csv_files:
        print("  No results CSVs found, skipping summary.")
        return

    rows = []
    for csv_file in csv_files:
        run_name = csv_file.replace(".csv", "")
        history = _read_history(os.path.join(RESULTS_ROOT, csv_file))
        if not history:
            continue
        best = max(history, key=lambda h: h["dice"])
        rows.append({
            "run": run_name,
            "best_epoch": best["epoch"],
            "dice": round(best["dice"], 4),
            "iou": round(best["iou"], 4),
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
        })

    summary_path = os.path.join(FIGURES_ROOT, "summary_table.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run", "best_epoch", "dice", "iou", "precision", "recall"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {summary_path}")


# ----------------------------------------------------------------------------
# Side-by-side prediction comparison: input | GT | U-Net | Attention U-Net
# ----------------------------------------------------------------------------
def _load_model(model_class, checkpoint_path, device):
    model = model_class(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def plot_model_comparison(
    unet_ckpt="outputs/checkpoints/unet_combined/best_model.pth",
    attn_ckpt="outputs/checkpoints/attention_unet_combined/best_model.pth",
    n_samples=3,
    image_size=256,
    seed=42,
):
    """Compare U-Net and Attention U-Net predictions on validation samples.

    By default uses the ``combined`` (Dice + BCE) checkpoints because that
    loss is generally the strongest under foreground-background imbalance.
    Falls back gracefully to ``dice`` checkpoints if the combined ones do
    not exist.
    """
    os.makedirs(FIGURES_ROOT, exist_ok=True)

    # Fallback to dice checkpoints if combined runs were skipped.
    if not (os.path.exists(unet_ckpt) and os.path.exists(attn_ckpt)):
        unet_ckpt = "outputs/checkpoints/unet_dice/best_model.pth"
        attn_ckpt = "outputs/checkpoints/attention_unet_dice/best_model.pth"

    if not (os.path.exists(unet_ckpt) and os.path.exists(attn_ckpt)):
        print("  One or both checkpoints missing, skipping model comparison.")
        print(f"    U-Net:           {unet_ckpt}")
        print(f"    Attention U-Net: {attn_ckpt}")
        return

    device = get_device()

    image_paths, mask_paths = get_image_mask_paths(IMAGE_DIR, MASK_DIR)
    _, val_imgs, _, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=seed,
    )

    val_dataset = SkinLesionDataset(
        val_imgs, val_masks, transform=get_val_transforms(image_size)
    )
    val_loader = DataLoader(val_dataset, batch_size=n_samples, shuffle=True)

    images, masks = next(iter(val_loader))
    images_dev = images.to(device)

    unet = _load_model(UNet, unet_ckpt, device)
    attn = _load_model(AttentionUNet, attn_ckpt, device)

    with torch.no_grad():
        unet_pred = (torch.sigmoid(unet(images_dev)) > 0.5).float().cpu()
        attn_pred = (torch.sigmoid(attn(images_dev)) > 0.5).float().cpu()

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    titles = ["Input Image", "Ground Truth", "U-Net Prediction", "Attention U-Net Prediction"]

    for i in range(n_samples):
        # Roughly de-normalize for display (Albumentations.Normalize uses ImageNet stats).
        img = images[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        gt = masks[i][0].numpy()
        u_p = unet_pred[i][0].numpy()
        a_p = attn_pred[i][0].numpy()
        panels = [img, gt, u_p, a_p]

        for j, panel in enumerate(panels):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(panel)
            else:
                ax.imshow(panel, cmap="gray")
            if i == 0:
                ax.set_title(titles[j], fontsize=13, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_ROOT, "model_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def run_all_visualizations():
    print("\n=== Visualizations ===")
    print("Training curves:")
    plot_training_curves()
    print("Summary table:")
    write_summary_table()
    print("Model comparison:")
    plot_model_comparison()


if __name__ == "__main__":
    run_all_visualizations()
