from __future__ import annotations

import numpy as np

from sprpcf.ml.dataset import DesignDataModule
from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.synthetic import build_synthetic_dataset


def test_grouped_train_validation_split_has_no_geometry_leakage(tmp_path) -> None:
    frame = build_synthetic_dataset(samples=6, wavelengths=64, seed=3)
    path = tmp_path / "data.csv"
    write_dataset(frame, path)
    data = DesignDataModule(path, batch_size=8, seed=3)
    data.setup()

    train_groups = set(data.groups[data.train_indices].tolist())
    val_groups = set(data.groups[data.val_indices].tolist())
    assert train_groups.isdisjoint(val_groups)
    assert np.isfinite(data.metric_scaler.scale_).all()
