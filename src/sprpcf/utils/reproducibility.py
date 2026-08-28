from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, include_tensorflow: bool = False) -> None:
    """Seed supported RNGs without making optional dependencies mandatory."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if include_tensorflow:
        try:
            import tensorflow as tf
        except ImportError:
            return
        tf.keras.utils.set_random_seed(seed)
