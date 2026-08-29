from __future__ import annotations

from collections.abc import Iterable
import logging
from pathlib import Path
import warnings

import torch
from torch import nn

from sprpcf.ml.losses import clamp_physical_geometry
from sprpcf.ml.tandem import InverseGenerator


class PhysicalInverseGenerator(nn.Module):
    """ONNX wrapper: physical target metrics + analyte RI -> physical geometry."""

    def __init__(
        self,
        inverse: InverseGenerator,
        metric_mean: Iterable[float],
        metric_scale: Iterable[float],
        condition_mean: Iterable[float],
        condition_scale: Iterable[float],
        geometry_mean: Iterable[float],
        geometry_scale: Iterable[float],
    ) -> None:
        super().__init__()
        self.inverse = inverse
        self.register_buffer("metric_mean", torch.tensor(list(metric_mean), dtype=torch.float32))
        self.register_buffer("metric_scale", torch.tensor(list(metric_scale), dtype=torch.float32))
        self.register_buffer("condition_mean", torch.tensor(list(condition_mean), dtype=torch.float32))
        self.register_buffer("condition_scale", torch.tensor(list(condition_scale), dtype=torch.float32))
        self.register_buffer("geometry_mean", torch.tensor(list(geometry_mean), dtype=torch.float32))
        self.register_buffer("geometry_scale", torch.tensor(list(geometry_scale), dtype=torch.float32))

    def forward(self, physical_metrics: torch.Tensor, analyte_ri: torch.Tensor) -> torch.Tensor:
        standardized_metrics = (physical_metrics - self.metric_mean) / self.metric_scale
        standardized_condition = (analyte_ri - self.condition_mean) / self.condition_scale
        standardized_geometry = self.inverse(standardized_metrics, standardized_condition)
        physical_geometry = standardized_geometry * self.geometry_scale + self.geometry_mean
        return clamp_physical_geometry(physical_geometry)


def export_inverse_generator_onnx(
    inverse: InverseGenerator,
    output_path: Path,
    metric_mean: Iterable[float],
    metric_scale: Iterable[float],
    condition_mean: Iterable[float],
    condition_scale: Iterable[float],
    geometry_mean: Iterable[float],
    geometry_scale: Iterable[float],
    opset: int = 18,
) -> None:
    """Export inverse generator as physical metrics + RI -> bounded physical geometry."""
    metric_mean_values = list(metric_mean)
    condition_mean_values = list(condition_mean)
    wrapper = PhysicalInverseGenerator(
        inverse.cpu().eval(),
        metric_mean_values,
        metric_scale,
        condition_mean_values,
        condition_scale,
        geometry_mean,
        geometry_scale,
    ).eval()
    dummy_metrics = torch.tensor([metric_mean_values], dtype=torch.float32)
    dummy_condition = torch.tensor([condition_mean_values], dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Let torch.export infer the shared symbolic batch relationship. Named shared
    # dimensions currently cause PyTorch's ONNX renamer to warn even though the
    # inferred constraint is valid; AUTO keeps the dimension dynamic without
    # entering that deprecated/ambiguous rename path.
    dynamic_shapes = {
        "physical_metrics": {0: torch.export.Dim.AUTO},
        "analyte_ri": {0: torch.export.Dim.AUTO},
    }

    # The exporter probes optional torchvision registrations even when this model
    # does not use torchvision operators. Silence only that internal probe logger;
    # project warnings remain visible and CI promotes them to errors.
    registration_logger = logging.getLogger("torch.onnx._internal.exporter._registration")
    previous_registration_level = registration_logger.level
    registration_logger.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
                module=r"copyreg",
            )
            torch.onnx.export(
                wrapper,
                (dummy_metrics, dummy_condition),
                output_path,
                input_names=["target_metrics", "analyte_ri"],
                output_names=["geometry"],
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                opset_version=opset,
            )
    finally:
        registration_logger.setLevel(previous_registration_level)
