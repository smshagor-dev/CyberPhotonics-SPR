from __future__ import annotations

import importlib.util

import pytest

from sprpcf.ml.tandem import InverseGenerator
from sprpcf.ml.onnx_export import export_inverse_generator_onnx


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None, reason="onnx is not installed")
def test_inverse_generator_onnx_export_is_valid(tmp_path) -> None:
    import onnx

    output_path = tmp_path / "inverse_pcf_spr.onnx"
    inverse = InverseGenerator()
    export_inverse_generator_onnx(
        inverse=inverse,
        output_path=output_path,
        metric_mean=[800.0, 20.0, 650.0],
        metric_scale=[100.0, 5.0, 50.0],
        geometry_mean=[2.0, 0.5, 45.0],
        geometry_scale=[0.5, 0.1, 10.0],
    )

    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "target_metrics"
    assert model.graph.output[0].name == "geometry"
