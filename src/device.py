"""Device selection helper. Picks CUDA when present, then Apple Silicon MPS,
then CPU. Used by training and visualization scripts."""

import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
