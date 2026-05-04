"""
Training script for the baseline U-Net.

Trains the model for a fixed number of epochs with a chosen loss function
(Dice / BCE / Combined Dice+BCE). After every epoch the model is evaluated
on the validation set with four metrics (Dice, IoU, Precision, Recall),
and a checkpoint is saved per epoch under
``outputs/checkpoints/<run_tag>/``. The best-scoring checkpoint by
validation Dice is also saved as ``best_model.pth`` in that folder, and a
per-epoch metrics CSV is written to ``outputs/results/<run_tag>.csv``.
"""

import os
import csv
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.dataset import SkinLesionDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.models.unet import UNet
from src.losses import get_loss
from src.metrics import dice_score, iou_score, precision_score, recall_score
from src.device import get_device


IMAGE_DIR = "data/raw/images"
MASK_DIR = "data/raw/masks"
CHECKPOINT_ROOT = "outputs/checkpoints"
RESULTS_ROOT = "outputs/results"

CSV_FIELDS = ["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall"]


def get_image_mask_paths(image_dir, mask_dir):
    """Pair every .jpg image with its <image_id>_segmentation.png mask."""
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


def train_unet(
    loss_name="dice",
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-3,
    image_size=256,
    use_augmentation=True,
    train_fraction=1.0,
    seed=42,
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
):
    """Train the baseline U-Net.

    Args:
        loss_name: 'dice', 'bce', or 'combined'.
        num_epochs: number of training epochs.
        batch_size: training/validation batch size.
        learning_rate: Adam learning rate.
        image_size: square resize size in pixels.
        use_augmentation: if False, training uses only resize+normalize
            (used for the augmentation ablation study).
        train_fraction: fraction of the training set to use (data-efficiency study).
        seed: random seed for the train/val split.
        image_dir, mask_dir: dataset paths (defaulted to data/raw/...).

    Returns:
        List of per-epoch dicts with train_loss, val_loss, dice, iou,
        precision, recall.
    """

    # ---- Build a unique run tag so each experiment has its own folder ----
    run_tag = f"unet_{loss_name}"
    if not use_augmentation:
        run_tag += "_noaug"
    if train_fraction < 1.0:
        run_tag += f"_frac{int(train_fraction * 100)}"

    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, run_tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # ---- Device ----
    device = get_device()
    print(f"\n[U-Net | {loss_name} | aug={use_augmentation} | "
          f"frac={train_fraction:.2f}] device={device}")

    # ---- Data ----
    image_paths, mask_paths = get_image_mask_paths(image_dir, mask_dir)

    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths,
        mask_paths,
        test_size=0.2,
        random_state=seed,
    )

    train_tf = get_train_transforms(image_size) if use_augmentation else get_val_transforms(image_size)
    val_tf = get_val_transforms(image_size)

    train_dataset = SkinLesionDataset(train_imgs, train_masks, transform=train_tf)
    val_dataset = SkinLesionDataset(val_imgs, val_masks, transform=val_tf)

    if train_fraction < 1.0:
        n_keep = max(1, int(len(train_dataset) * train_fraction))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_dataset), generator=generator)[:n_keep].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"  Reduced training set: {n_keep} samples ({train_fraction * 100:.0f}%)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Model / optimizer / loss ----
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss(loss_name)

    # ---- Training loop (written out explicitly) ----
    history = []
    best_dice = -1.0

    for epoch in range(1, num_epochs + 1):
        # ---------------- Train one epoch ----------------
        model.train()
        running_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            preds = model(images)
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # ---------------- Validate ----------------
        model.eval()
        val_running_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_prec = 0.0
        val_rec = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                preds = model(images)

                val_running_loss += loss_fn(preds, masks).item()
                val_dice += dice_score(preds, masks)
                val_iou += iou_score(preds, masks)
                val_prec += precision_score(preds, masks)
                val_rec += recall_score(preds, masks)

        n_batches = max(1, len(val_loader))
        val_loss = val_running_loss / n_batches
        val_dice /= n_batches
        val_iou /= n_batches
        val_prec /= n_batches
        val_rec /= n_batches

        print(
            f"  Epoch {epoch:02d}/{num_epochs} | "
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"dice={val_dice:.4f} | iou={val_iou:.4f} | "
            f"prec={val_prec:.4f} | rec={val_rec:.4f}"
        )

        # ---------------- Save checkpoints ----------------
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_dir, f"epoch{epoch:02d}.pth"),
        )
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dice": val_dice,
            "iou": val_iou,
            "precision": val_prec,
            "recall": val_rec,
        })

    # ---- Save metrics CSV ----
    results_csv = os.path.join(RESULTS_ROOT, f"{run_tag}.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    print(f"  Best Dice: {best_dice:.4f} | Results -> {results_csv}")
    return history


if __name__ == "__main__":
    train_unet(loss_name="dice", num_epochs=10)
