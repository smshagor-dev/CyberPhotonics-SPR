from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from sprpcf.ml.tandem import InverseGenerator


class PhysicalInverseGenerator(nn.Module):
    """ONNX wrapper: physical target metrics -> physical geometry."""

    def __init__(
        self,
        inverse: InverseGenerator,
        metric_mean: Iterable[float],
        metric_scale: Iterable[float],
        geometry_mean: Iterable[float],
        geometry_scale: Iterable[float],
    ) -> None:
        super().__init__()
        self.inverse = inverse
        self.register_buffer("metric_mean", torch.tensor(list(metric_mean), dtype=torch.float32))
        self.register_buffer("metric_scale", torch.tensor(list(metric_scale), dtype=torch.float32))
        self.register_buffer("geometry_mean", torch.tensor(list(geometry_mean), dtype=torch.float32))
        self.register_buffer("geometry_scale", torch.tensor(list(geometry_scale), dtype=torch.float32))

    def forward(self, physical_metrics: torch.Tensor) -> torch.Tensor:
        standardized_metrics = (physical_metrics - self.metric_mean) / self.metric_scale
        standardized_geometry = self.inverse(standardized_metrics)
        return standardized_geometry * self.geometry_scale + self.geometry_mean


def export_inverse_generator_onnx(
    inverse: InverseGenerator,
    output_path: Path,
    metric_mean: Iterable[float],
    metric_scale: Iterable[float],
    geometry_mean: Iterable[float],
    geometry_scale: Iterable[float],
    opset: int = 17,
) -> None:
    """Export inverse generator as physical metrics -> physical geometry ONNX."""
    wrapper = PhysicalInverseGenerator(inverse.cpu().eval(), metric_mean, metric_scale, geometry_mean, geometry_scale).eval()
    dummy_metrics = torch.tensor([list(metric_mean)], dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_metrics,
        output_path,
        input_names=["target_metrics"],
        output_names=["geometry"],
        dynamic_axes={"target_metrics": {0: "batch"}, "geometry": {0: "batch"}},
        opset_version=opset,
    )
