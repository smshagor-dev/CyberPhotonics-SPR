from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


_SCALER_KEYS = {
    "geometry_mean",
    "geometry_scale",
    "condition_mean",
    "condition_scale",
    "metric_mean",
    "metric_scale",
}


def _safe_value(value: Any) -> Any:
    """Convert checkpoint metadata to values accepted by PyTorch weights-only loading."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    return value


def save_tandem_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    """Persist a checkpoint that can be read without arbitrary pickle execution."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_safe_value(checkpoint), path)


def load_tandem_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted project checkpoint using PyTorch's restricted weights-only unpickler."""
    path = Path(path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Refusing to deserialize unsafe or legacy checkpoint: {path}. "
            "Regenerate the checkpoint with the current CyberPhotonics-SPR training pipeline."
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    for key in _SCALER_KEYS:
        value = checkpoint.get(key)
        if isinstance(value, torch.Tensor):
            checkpoint[key] = value.detach().cpu().numpy()
    return checkpoint
