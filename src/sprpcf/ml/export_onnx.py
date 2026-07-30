from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator, TandemNetwork
from sprpcf.ml.onnx_export import export_inverse_generator_onnx


class _OnnxTandemWrapper(torch.nn.Module):
    def __init__(self, tandem: TandemNetwork) -> None:
        super().__init__()
        self.tandem = tandem

    def forward(self, target_metrics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tandem(target_metrics)


def export_tandem_onnx(checkpoint_path: Path, output_path: Path) -> None:
    """Export the tandem inverse-design model to ONNX for runtime inference."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TandemNetwork(ForwardNetwork(), InverseGenerator())
    model.load_state_dict(checkpoint["model"])
    model.eval()
    wrapper = _OnnxTandemWrapper(model)
    dummy_metrics = torch.randn(1, 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_metrics,
        output_path,
        input_names=["target_metrics"],
        output_names=["geometry", "predicted_metrics"],
        dynamic_axes={"target_metrics": {0: "batch"}, "geometry": {0: "batch"}, "predicted_metrics": {0: "batch"}},
        opset_version=17,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained tandem model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inverse-only", action="store_true")
    args = parser.parse_args()
    if args.inverse_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        inverse = InverseGenerator()
        inverse.load_state_dict(checkpoint["inverse_state_dict"])
        export_inverse_generator_onnx(
            inverse,
            args.out,
            checkpoint["metric_mean"],
            checkpoint["metric_scale"],
            checkpoint["geometry_mean"],
            checkpoint["geometry_scale"],
        )
    else:
        export_tandem_onnx(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
