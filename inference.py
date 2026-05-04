"""
Single-image inference script for the skin lesion segmentation project.

Downloads the trained model from Hugging Face on first run (cached after that),
runs it on an input dermoscopic image, and saves the predicted lesion mask
plus a side-by-side visualization next to the input.

Usage:
    python inference.py
        # uses the bundled sample image and the U-Net model

    python inference.py --model attention_unet
        # uses the Attention U-Net checkpoint instead

    python inference.py --image path/to/your/image.jpg
        # runs on your own image

All outputs land in ./inference_outputs/.
"""

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.transforms import get_val_transforms
from src.device import get_device


# ----------------------------------------------------------------------------
# Hugging Face model registry
# ----------------------------------------------------------------------------
HF_REPO = "CharanKankanala2003/skin-lesion-unet-models"

MODELS = {
    "unet": {
        "class": UNet,
        "filename": "unet_combined.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/unet_combined.pth",
        "description": "Baseline U-Net trained with combined Dice + BCE loss (Dice=0.8334)",
    },
    "attention_unet": {
        "class": AttentionUNet,
        "filename": "attention_unet_dice.pth",
        "url": f"https://huggingface.co/{HF_REPO}/resolve/main/attention_unet_dice.pth",
        "description": "Attention U-Net trained with Dice loss, lr=1e-4 (Dice=0.8153)",
    },
}

CACHE_DIR = "models_cache"
OUTPUT_DIR = "inference_outputs"
SAMPLE_IMAGE = "sample_image.jpg"
IMAGE_SIZE = 256


# ----------------------------------------------------------------------------
# Model loading (with HF download + local cache)
# ----------------------------------------------------------------------------
def download_with_progress(url, dest_path):
    """Download a file from URL with a simple progress bar."""
    def _hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\r  Downloading: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, _hook)
    print()  # newline after progress bar


def load_model(model_key, device):
    """Load the requested model, downloading the checkpoint from HF if needed."""
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from: {list(MODELS.keys())}"
        )

    info = MODELS[model_key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    ckpt_path = os.path.join(CACHE_DIR, info["filename"])

    if not os.path.exists(ckpt_path):
        print(f"Model not found locally. Downloading from Hugging Face...")
        print(f"  Source: {info['url']}")
        download_with_progress(info["url"], ckpt_path)
        print(f"  Saved to: {ckpt_path}")
    else:
        print(f"Using cached model: {ckpt_path}")

    model = info["class"](in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ----------------------------------------------------------------------------
# Inference + visualization
# ----------------------------------------------------------------------------
def run_inference(model, image_path, device, image_size=IMAGE_SIZE):
    """Run the model on a single image and return (input_rgb, pred_mask)."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transform = get_val_transforms(image_size)
    augmented = transform(image=image_rgb, mask=np.zeros(image_rgb.shape[:2], dtype="float32"))
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits)
        pred_mask = (prob > 0.5).float().cpu().numpy()[0, 0]

    image_resized = cv2.resize(image_rgb, (image_size, image_size))
    return image_resized, pred_mask


def save_visualization(image_rgb, pred_mask, model_key, save_dir=OUTPUT_DIR):
    """Save the predicted mask and a side-by-side comparison figure."""
    os.makedirs(save_dir, exist_ok=True)

    mask_path = os.path.join(save_dir, f"{model_key}_predicted_mask.png")
    cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Image", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(pred_mask, cmap="gray")
    axes[1].set_title(f"{model_key} - Predicted Lesion Mask", fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    fig_path = os.path.join(save_dir, f"{model_key}_visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mask_path, fig_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run skin lesion segmentation on a single image."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="unet",
        help="Which trained model to use (default: unet).",
    )
    parser.add_argument(
        "--image",
        default=SAMPLE_IMAGE,
        help=f"Path to input image (default: {SAMPLE_IMAGE}).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Input image not found at '{args.image}'")
        print(f"       Either provide --image PATH or place a sample at '{SAMPLE_IMAGE}'.")
        sys.exit(1)

    device = get_device()
    print("=" * 70)
    print("Skin Lesion Segmentation - Inference")
    print("=" * 70)
    print(f"Model:  {args.model}  ({MODELS[args.model]['description']})")
    print(f"Image:  {args.image}")
    print(f"Device: {device}")
    print()

    model = load_model(args.model, device)
    print("Running inference...")
    image_rgb, pred_mask = run_inference(model, args.image, device)

    coverage = float(pred_mask.mean()) * 100
    print(f"Predicted lesion coverage: {coverage:.2f}% of image area")

    mask_path, fig_path = save_visualization(image_rgb, pred_mask, args.model)
    print()
    print("Saved outputs:")
    print(f"  Mask:           {mask_path}")
    print(f"  Visualization:  {fig_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
