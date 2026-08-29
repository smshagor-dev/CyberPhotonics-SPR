from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint

from sprpcf.ml.losses import clamp_physical_geometry
from sprpcf.ml.onnx_export import export_inverse_generator_onnx
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator, TandemNetwork


class PhysicalTandemWrapper(torch.nn.Module):
    """Expose physical metrics/RI inputs and physical geometry/metric outputs."""

    def __init__(self, tandem: TandemNetwork, checkpoint: dict) -> None:
        super().__init__()
        self.tandem = tandem
        for name in ("metric_mean", "metric_scale", "condition_mean", "condition_scale", "geometry_mean", "geometry_scale"):
            self.register_buffer(name, torch.tensor(checkpoint[name], dtype=torch.float32))

    def forward(self, target_metrics: torch.Tensor, analyte_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        standardized_metrics = (target_metrics - self.metric_mean) / self.metric_scale
        standardized_condition = (analyte_ri - self.condition_mean) / self.condition_scale
        standardized_geometry, predicted_metrics = self.tandem(standardized_metrics, standardized_condition)
        physical_geometry = clamp_physical_geometry(standardized_geometry * self.geometry_scale + self.geometry_mean)
        physical_metrics = predicted_metrics * self.metric_scale + self.metric_mean
        return physical_geometry, physical_metrics


def export_tandem_onnx(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = load_tandem_checkpoint(checkpoint_path)
    model = TandemNetwork(ForwardNetwork(), InverseGenerator())
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wrapper = PhysicalTandemWrapper(model, checkpoint).eval()
    dummy_metrics = torch.tensor([checkpoint["metric_mean"]], dtype=torch.float32)
    dummy_condition = torch.tensor([checkpoint["condition_mean"]], dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_metrics, dummy_condition),
        output_path,
        input_names=["target_metrics", "analyte_ri"],
        output_names=["geometry", "predicted_metrics"],
        dynamic_axes={
            "target_metrics": {0: "batch"},
            "analyte_ri": {0: "batch"},
            "geometry": {0: "batch"},
            "predicted_metrics": {0: "batch"},
        },
        opset_version=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained tandem model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inverse-only", action="store_true")
    args = parser.parse_args()
    checkpoint = load_tandem_checkpoint(args.checkpoint)
    if args.inverse_only:
        inverse = InverseGenerator()
        inverse.load_state_dict(checkpoint["inverse_state_dict"])
        export_inverse_generator_onnx(
            inverse,
            args.out,
            checkpoint["metric_mean"],
            checkpoint["metric_scale"],
            checkpoint["condition_mean"],
            checkpoint["condition_scale"],
            checkpoint["geometry_mean"],
            checkpoint["geometry_scale"],
        )
    else:
        export_tandem_onnx(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
