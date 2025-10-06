# unlearun/utils/helpers.py
"""Helper utility functions."""

import random
import numpy as np
import torch
import time
from datetime import timedelta


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def format_time(seconds: float) -> str:
    """Format seconds to human-readable string."""
    return str(timedelta(seconds=int(seconds)))