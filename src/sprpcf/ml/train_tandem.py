from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import optim
from torch.utils.data import DataLoader
from tqdm import trange

from sprpcf.ml.checkpoint_io import save_tandem_checkpoint
from sprpcf.ml.dataset import CONDITION_COLUMNS, DesignDataModule, GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.ml.losses import clamp_physical_geometry, geometry_constraint_loss
from sprpcf.ml.onnx_export import export_inverse_generator_onnx
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator, TandemNetwork
from sprpcf.utils.reproducibility import seed_everything


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_forward(model: ForwardNetwork, loader: DataLoader, data: DesignDataModule, device: torch.device) -> dict[str, float]:
    model.eval()
    truth: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    with torch.no_grad():
        for geometry, conditions, metrics in loader:
            forward_input = torch.cat([geometry, conditions], dim=-1).to(device)
            output = model(forward_input).cpu().numpy()
            pred.append(data.metric_scaler.inverse_transform(output))
            truth.append(data.metric_scaler.inverse_transform(metrics.numpy()))
    return _regression_metrics(np.concatenate(truth), np.concatenate(pred))


def evaluate_inverse_geometry(
    inverse: InverseGenerator,
    loader: DataLoader,
    data: DesignDataModule,
    device: torch.device,
) -> dict[str, float]:
    inverse.eval()
    truth: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    with torch.no_grad():
        for geometry, conditions, metrics in loader:
            generated = inverse(metrics.to(device), conditions.to(device)).cpu().numpy()
            generated_physical = data.geometry_scaler.inverse_transform(generated)
            generated_tensor = clamp_physical_geometry(torch.tensor(generated_physical, dtype=torch.float32))
            pred.append(generated_tensor.numpy())
            truth.append(data.geometry_scaler.inverse_transform(geometry.numpy()))
    return _regression_metrics(np.concatenate(truth), np.concatenate(pred))


def train_forward(data: DesignDataModule, epochs: int, lr: float, device: torch.device) -> ForwardNetwork:
    model = ForwardNetwork().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in trange(epochs, desc="forward"):
        model.train()
        for geometry, conditions, metrics in data.train_loader():
            forward_input = torch.cat([geometry, conditions], dim=-1).to(device)
            metrics = metrics.to(device)
            loss = F.mse_loss(model(forward_input), metrics)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def train_inverse(
    data: DesignDataModule,
    forward_model: ForwardNetwork,
    epochs: int,
    lr: float,
    device: torch.device,
    alpha: float,
    beta: float,
    dispersion_weight: float = 0.0,
) -> InverseGenerator:
    inverse = InverseGenerator().to(device)
    forward_model.eval()
    for parameter in forward_model.parameters():
        parameter.requires_grad = False

    optimizer = optim.AdamW(inverse.parameters(), lr=lr, weight_decay=1e-4)
    geometry_mean = torch.tensor(data.geometry_scaler.mean_, dtype=torch.float32, device=device)
    geometry_scale = torch.tensor(data.geometry_scaler.scale_, dtype=torch.float32, device=device)
    lambda_mean = torch.tensor(data.metric_scaler.mean_[2], dtype=torch.float32, device=device)
    lambda_scale = torch.tensor(data.metric_scaler.scale_[2], dtype=torch.float32, device=device)

    for _ in trange(epochs, desc="inverse"):
        inverse.train()
        for _, conditions, target_metrics in data.train_loader():
            conditions = conditions.to(device)
            target_metrics = target_metrics.to(device)
            latent = torch.randn(
                target_metrics.shape[0], inverse.latent_dim, dtype=target_metrics.dtype, device=device
            ) * 0.05
            generated_geometry = inverse(target_metrics, conditions, latent)
            predicted_metrics = forward_model(torch.cat([generated_geometry, conditions], dim=-1))
            physical_geometry = generated_geometry * geometry_scale + geometry_mean
            loss = F.mse_loss(predicted_metrics, target_metrics) + geometry_constraint_loss(
                physical_geometry,
                overlap_weight=alpha,
                boundary_weight=beta,
                resonance_wavelength_nm=predicted_metrics[:, 2] * lambda_scale + lambda_mean,
                dispersion_weight=dispersion_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return inverse


def train_tandem_pipeline(
    data_path: Path,
    checkpoint_out: Path,
    onnx_out: Path,
    epochs: int = 50,
    forward_epochs: int | None = None,
    inverse_epochs: int | None = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_name: str = "auto",
    alpha: float = 1.0,
    beta: float = 1.0,
    dispersion_weight: float = 0.0,
    seed: int = 7,
) -> dict[str, dict[str, float]]:
    """Train forward and inverse tandem networks and export the inverse generator."""
    seed_everything(seed)
    device = _select_device(device_name)
    resolved_forward_epochs = forward_epochs if forward_epochs is not None else epochs
    resolved_inverse_epochs = inverse_epochs if inverse_epochs is not None else epochs

    data = DesignDataModule(data_path, batch_size=batch_size, seed=seed)
    data.setup()
    forward = train_forward(data, resolved_forward_epochs, lr, device)
    forward_metrics = evaluate_forward(forward, data.val_loader(), data, device)
    inverse = train_inverse(data, forward, resolved_inverse_epochs, lr, device, alpha, beta, dispersion_weight)
    inverse_metrics = evaluate_inverse_geometry(inverse, data.val_loader(), data, device)
    tandem = TandemNetwork(forward, inverse)

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": tandem.state_dict(),
        "forward_state_dict": forward.state_dict(),
        "inverse_state_dict": inverse.state_dict(),
        "forward_metrics_physical": forward_metrics,
        "inverse_geometry_metrics_physical": inverse_metrics,
        "geometry_columns": GEOMETRY_COLUMNS,
        "condition_columns": CONDITION_COLUMNS,
        "metric_columns": METRIC_COLUMNS,
        "geometry_mean": data.geometry_scaler.mean_,
        "geometry_scale": data.geometry_scaler.scale_,
        "condition_mean": data.condition_scaler.mean_,
        "condition_scale": data.condition_scaler.scale_,
        "metric_mean": data.metric_scaler.mean_,
        "metric_scale": data.metric_scaler.scale_,
        "seed": seed,
    }
    save_tandem_checkpoint(checkpoint, checkpoint_out)
    export_inverse_generator_onnx(
        inverse,
        onnx_out,
        data.metric_scaler.mean_,
        data.metric_scaler.scale_,
        data.condition_scaler.mean_,
        data.condition_scaler.scale_,
        data.geometry_scaler.mean_,
        data.geometry_scaler.scale_,
    )
    return {"forward": forward_metrics, "inverse_geometry": inverse_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tandem network for PCF-SPR inverse design.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forward-epochs", type=int, default=None)
    parser.add_argument("--inverse-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--alpha", type=float, default=1.0, help="Air-hole overlap penalty weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Fabrication-boundary penalty weight.")
    parser.add_argument("--dispersion-weight", type=float, default=0.0, help="Sellmeier/Drude dispersion penalty weight.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("models/tandem.pt"))
    parser.add_argument("--onnx-out", type=Path, default=Path("models/inverse_pcf_spr.onnx"))
    args = parser.parse_args()

    metrics = train_tandem_pipeline(
        data_path=args.data,
        checkpoint_out=args.out,
        onnx_out=args.onnx_out,
        epochs=args.epochs,
        forward_epochs=args.forward_epochs,
        inverse_epochs=args.inverse_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        alpha=args.alpha,
        beta=args.beta,
        dispersion_weight=args.dispersion_weight,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
