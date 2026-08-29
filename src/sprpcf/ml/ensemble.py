from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint, save_tandem_checkpoint

from sprpcf.ml.dataset import DesignDataModule
from sprpcf.ml.tandem import ForwardNetwork
from sprpcf.ml.train_tandem import _select_device, evaluate_forward, train_forward
from sprpcf.utils.reproducibility import seed_everything


def _check_scalers(checkpoint: dict[str, Any], data: DesignDataModule) -> None:
    pairs = [
        ("geometry_mean", data.geometry_scaler.mean_),
        ("geometry_scale", data.geometry_scaler.scale_),
        ("condition_mean", data.condition_scaler.mean_),
        ("condition_scale", data.condition_scaler.scale_),
        ("metric_mean", data.metric_scaler.mean_),
        ("metric_scale", data.metric_scaler.scale_),
    ]
    for key, current in pairs:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint is missing scaler field {key!r}.")
        stored = np.asarray(checkpoint[key], dtype=float)
        if stored.shape != np.asarray(current).shape or not np.allclose(stored, current, rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"Reference data split/scaling does not match checkpoint field {key!r}; "
                "use the dataset and seed that produced the checkpoint."
            )


def build_forward_ensemble_checkpoint(
    checkpoint_path: Path,
    data_path: Path,
    output_path: Path,
    *,
    members: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Attach independently initialized forward surrogates to an existing tandem checkpoint."""
    if members < 2:
        raise ValueError("members must be >= 2 for a forward ensemble.")
    if epochs < 1:
        raise ValueError("epochs must be >= 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    checkpoint = load_tandem_checkpoint(checkpoint_path)
    if "forward_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing forward_state_dict.")
    seed = int(checkpoint.get("seed", 7))
    data = DesignDataModule(data_path, batch_size=batch_size, seed=seed)
    data.setup()
    _check_scalers(checkpoint, data)
    device = _select_device(device_name)

    primary = ForwardNetwork().to(device)
    primary.load_state_dict(checkpoint["forward_state_dict"])
    primary.eval()
    models = [primary]
    metrics: list[dict[str, float]] = [evaluate_forward(primary, data.val_loader(), data, device)]

    for member_index in range(1, members):
        seed_everything(seed + 1009 * member_index)
        model = train_forward(data, epochs, lr, device)
        models.append(model)
        metrics.append(evaluate_forward(model, data.val_loader(), data, device))

    checkpoint["forward_ensemble_state_dicts"] = [model.state_dict() for model in models]
    checkpoint["forward_ensemble_members"] = int(members)
    checkpoint["forward_ensemble_validation_metrics_physical"] = metrics
    checkpoint["forward_ensemble_training"] = {
        "members": int(members),
        "additional_member_epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "base_seed": int(seed),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_tandem_checkpoint(checkpoint, output_path)
    return {
        "members": int(members),
        "validation_metrics": metrics,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a deep forward ensemble to a tandem PCF-SPR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = build_forward_ensemble_checkpoint(
        args.checkpoint,
        args.data,
        args.out,
        members=args.members,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
