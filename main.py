"""
Top-level entry point for the skin lesion segmentation project.

This script reproduces every experiment described in the project proposal:

  1. Architecture comparison        - U-Net  vs Attention U-Net
  2. Loss function study            - Dice / BCE / Combined Dice+BCE
  3. Data efficiency study          - Full dataset vs 50% subset
  4. Augmentation study             - With augmentation vs without

After training, all visualizations (training curves, summary table,
prediction comparison) are generated.

Per-run outputs are saved under:
    outputs/checkpoints/<run_tag>/      # epoch01.pth ... epochN.pth + best_model.pth
    outputs/results/<run_tag>.csv       # per-epoch metrics
    outputs/figures/                    # plots and summary table
"""

import argparse

from src.train_unet import train_unet
from src.train_attention_unet import train_attention_unet
from src.visualize import run_all_visualizations


# Loss functions covered by the loss-comparison experiment.
LOSS_FUNCTIONS = ["dice", "bce", "combined"]

# Loss used for the data-efficiency and augmentation ablation studies.
# Combined Dice + BCE is the standard strong baseline for imbalanced
# segmentation, so we keep it as the loss while changing the other factor.
ABLATION_LOSS = "combined"


def run_main_experiments(num_epochs, batch_size, learning_rate):
    """Architecture + loss study (full data, with augmentation)."""
    print("\n" + "=" * 70)
    print(" 1) ARCHITECTURE + LOSS STUDY")
    print("=" * 70)

    for loss_name in LOSS_FUNCTIONS:
        train_unet(
            loss_name=loss_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_augmentation=True,
            train_fraction=1.0,
        )

    for loss_name in LOSS_FUNCTIONS:
        train_attention_unet(
            loss_name=loss_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            use_augmentation=True,
            train_fraction=1.0,
        )


def run_data_efficiency_experiments(num_epochs, batch_size, learning_rate, fraction=0.5):
    """Both models with reduced training data, all other settings fixed."""
    print("\n" + "=" * 70)
    print(f" 2) DATA EFFICIENCY STUDY ({int(fraction * 100)}% training data)")
    print("=" * 70)

    train_unet(
        loss_name=ABLATION_LOSS,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentation=True,
        train_fraction=fraction,
    )
    train_attention_unet(
        loss_name=ABLATION_LOSS,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentation=True,
        train_fraction=fraction,
    )


def run_augmentation_experiments(num_epochs, batch_size, learning_rate):
    """Both models trained without augmentation."""
    print("\n" + "=" * 70)
    print(" 3) AUGMENTATION STUDY (no augmentation)")
    print("=" * 70)

    train_unet(
        loss_name=ABLATION_LOSS,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentation=False,
        train_fraction=1.0,
    )
    train_attention_unet(
        loss_name=ABLATION_LOSS,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_augmentation=False,
        train_fraction=1.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Skin lesion U-Net experiments")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs per run (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-fraction", type=float, default=0.5,
                        help="Fraction used in the data-efficiency study (default: 0.5)")

    # Skip flags so the user can run any subset independently.
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip all training and only generate visualizations.")
    parser.add_argument("--skip-main", action="store_true",
                        help="Skip the architecture + loss study.")
    parser.add_argument("--skip-data-efficiency", action="store_true",
                        help="Skip the 50%% training-set study.")
    parser.add_argument("--skip-aug-study", action="store_true",
                        help="Skip the no-augmentation study.")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization at the end.")
    args = parser.parse_args()

    if not args.skip_train:
        if not args.skip_main:
            run_main_experiments(args.epochs, args.batch_size, args.lr)

        if not args.skip_data_efficiency:
            run_data_efficiency_experiments(
                args.epochs, args.batch_size, args.lr, fraction=args.data_fraction
            )

        if not args.skip_aug_study:
            run_augmentation_experiments(args.epochs, args.batch_size, args.lr)

    if not args.skip_viz:
        run_all_visualizations()


if __name__ == "__main__":
    main()
