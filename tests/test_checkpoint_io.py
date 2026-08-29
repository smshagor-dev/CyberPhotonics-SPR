from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint, save_tandem_checkpoint


def test_safe_checkpoint_round_trip_converts_numpy_scalers(tmp_path: Path) -> None:
    path = tmp_path / "safe.pt"
    payload = {
        "inverse_state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "geometry_mean": np.array([1.0, 2.0], dtype=np.float64),
        "metric_scale": np.array([3.0, 4.0], dtype=np.float32),
        "seed": 7,
    }
    save_tandem_checkpoint(payload, path)
    loaded = load_tandem_checkpoint(path)
    np.testing.assert_allclose(loaded["geometry_mean"], [1.0, 2.0])
    np.testing.assert_allclose(loaded["metric_scale"], [3.0, 4.0])
    assert loaded["seed"] == 7


def test_safe_loader_refuses_legacy_numpy_pickle_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save({"geometry_mean": np.array([1.0, 2.0])}, path)
    with pytest.raises(RuntimeError, match="Refusing to deserialize unsafe or legacy checkpoint"):
        load_tandem_checkpoint(path)
