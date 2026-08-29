from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.dataset import FORWARD_INPUT_COLUMNS, METRIC_COLUMNS, read_table
from sprpcf.ml.tandem import ForwardNetwork


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_index: int = 0,
    baseline: torch.Tensor | None = None,
    steps: int = 64,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be >= 2.")
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    scaled_inputs = [baseline + (float(step) / steps) * (inputs - baseline) for step in range(1, steps + 1)]
    gradients: list[torch.Tensor] = []
    for scaled in scaled_inputs:
        scaled = scaled.detach().requires_grad_(True)
        output = model(scaled)[:, target_index].sum()
        gradient = torch.autograd.grad(output, scaled)[0]
        gradients.append(gradient)
    average_gradient = torch.stack(gradients, dim=0).mean(dim=0)
    return (inputs - baseline) * average_gradient


def shap_feature_attribution(
    model: torch.nn.Module,
    background: np.ndarray,
    samples: np.ndarray,
    target_index: int = 0,
) -> np.ndarray:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install 'shap' to use SHAP attributions, or use Integrated Gradients.") from exc

    def predict(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            output = model(torch.tensor(batch, dtype=torch.float32)).cpu().numpy()
        return output[:, target_index]

    explainer = shap.KernelExplainer(predict, background)
    values = explainer.shap_values(samples)
    return np.asarray(values, dtype=np.float32)


def attribution_matrix(
    model: torch.nn.Module,
    standardized_inputs: np.ndarray,
    method: str = "integrated-gradients",
    target_index: int = 0,
    steps: int = 64,
) -> pd.DataFrame:
    """Return attributions in the same standardized input space used for training."""
    if standardized_inputs.ndim != 2 or standardized_inputs.shape[1] != len(FORWARD_INPUT_COLUMNS):
        raise ValueError(f"standardized_inputs must have shape [N, {len(FORWARD_INPUT_COLUMNS)}].")
    if method == "shap":
        background = standardized_inputs[: min(20, standardized_inputs.shape[0])]
        values = shap_feature_attribution(model, background, standardized_inputs, target_index=target_index)
    elif method == "integrated-gradients":
        tensor = torch.tensor(standardized_inputs, dtype=torch.float32)
        values = integrated_gradients(model, tensor, target_index=target_index, steps=steps).detach().cpu().numpy()
    else:
        raise ValueError("method must be 'shap' or 'integrated-gradients'.")
    return pd.DataFrame(values, columns=FORWARD_INPUT_COLUMNS)


def save_attribution_outputs(matrix: pd.DataFrame, output_csv: Path, heatmap_path: Path | None = None) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_csv, index=False)
    if heatmap_path is not None:
        import matplotlib.pyplot as plt

        heatmap_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        image = ax.imshow(matrix.to_numpy().T, aspect="auto", cmap="coolwarm")
        ax.set_yticks(range(len(matrix.columns)), labels=matrix.columns)
        ax.set_xlabel("Sample")
        ax.set_title("Forward-Model Feature Attribution (standardized input space)")
        fig.colorbar(image, ax=ax, label="Attribution")
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=160)
        plt.close(fig)


def load_forward_model_and_scalers(checkpoint_path: Path) -> tuple[ForwardNetwork, dict[str, np.ndarray]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ForwardNetwork()
    model.load_state_dict(checkpoint["forward_state_dict"])
    model.eval()
    scalers = {
        "geometry_mean": np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        "geometry_scale": np.asarray(checkpoint["geometry_scale"], dtype=np.float32),
        "condition_mean": np.asarray(checkpoint["condition_mean"], dtype=np.float32),
        "condition_scale": np.asarray(checkpoint["condition_scale"], dtype=np.float32),
    }
    return model, scalers


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PCF-SPR XAI feature attribution matrices.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", choices=["integrated-gradients", "shap"], default="integrated-gradients")
    parser.add_argument("--target", choices=METRIC_COLUMNS, default="sensitivity_nm_per_riu")
    parser.add_argument("--out", type=Path, default=Path("outputs/feature_attribution.csv"))
    parser.add_argument("--heatmap", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()

    frame = read_table(args.data).dropna(subset=FORWARD_INPUT_COLUMNS).iloc[: args.limit]
    model, scalers = load_forward_model_and_scalers(args.checkpoint)
    geometry = frame[FORWARD_INPUT_COLUMNS[:-1]].to_numpy(np.float32)
    condition = frame[FORWARD_INPUT_COLUMNS[-1:]].to_numpy(np.float32)
    geometry = (geometry - scalers["geometry_mean"]) / scalers["geometry_scale"]
    condition = (condition - scalers["condition_mean"]) / scalers["condition_scale"]
    standardized_inputs = np.concatenate([geometry, condition], axis=1)
    matrix = attribution_matrix(
        model,
        standardized_inputs,
        method=args.method,
        target_index=METRIC_COLUMNS.index(args.target),
    )
    save_attribution_outputs(matrix, args.out, args.heatmap)
    print(f"Wrote attribution matrix to {args.out}")


if __name__ == "__main__":
    main()
