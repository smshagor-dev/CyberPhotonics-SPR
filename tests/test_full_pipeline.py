from __future__ import annotations

import importlib.util

import pytest

from main import main


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None or importlib.util.find_spec("onnx") is None,
    reason="full pipeline requires TensorFlow and ONNX",
)


def test_end_to_end_mock_pipeline_runs(tmp_path) -> None:
    data_path = tmp_path / "dataset.parquet"
    model_dir = tmp_path / "models"

    main(["generate-data", "--samples", "12", "--wavelengths", "32", "--out", data_path.as_posix()])
    main(
        [
            "train-inverse",
            "--data",
            data_path.as_posix(),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--checkpoint",
            (model_dir / "tandem.pt").as_posix(),
            "--export-onnx",
            (model_dir / "inverse_pcf_spr.onnx").as_posix(),
        ]
    )
    main(
        [
            "train-edge",
            "--data",
            data_path.as_posix(),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--quantize",
            "--export-dir",
            model_dir.as_posix(),
        ]
    )
    main(
        [
            "simulate-stream",
            "--data",
            data_path.as_posix(),
            "--tflite-dir",
            model_dir.as_posix(),
            "--duration-sec",
            "0.01",
        ]
    )

    assert (model_dir / "inverse_pcf_spr.onnx").exists()
    assert (model_dir / "edge_denoiser_quantized.tflite").exists()
    assert (model_dir / "edge_ri_predictor_quantized.tflite").exists()
