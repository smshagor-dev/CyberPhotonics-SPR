from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

GEOMETRY_COLUMNS = ["pitch_um", "d_over_lambda", "metal_thickness_nm", "channel_radius_um"]
CONDITION_COLUMNS = ["analyte_ri"]
FORWARD_INPUT_COLUMNS = GEOMETRY_COLUMNS + CONDITION_COLUMNS
METRIC_COLUMNS = ["sensitivity_nm_per_riu", "fom_per_riu", "lambda_res_nm"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def geometry_group_labels(frame: pd.DataFrame) -> np.ndarray:
    """Create stable group labels so RI sweeps of one geometry stay in one split."""
    missing = [column for column in GEOMETRY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing geometry columns for grouped split: {missing}")
    return pd.util.hash_pandas_object(frame[GEOMETRY_COLUMNS], index=False).to_numpy()


class DesignDataModule:
    """Prepare leakage-resistant standardized tensors for tandem learning."""

    def __init__(self, path: Path, batch_size: int = 64, test_size: float = 0.2, seed: int = 7) -> None:
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.geometry_scaler = StandardScaler()
        self.condition_scaler = StandardScaler()
        self.metric_scaler = StandardScaler()

    def setup(self) -> None:
        required = GEOMETRY_COLUMNS + CONDITION_COLUMNS + METRIC_COLUMNS
        frame = read_table(self.path).dropna(subset=required).reset_index(drop=True)
        if len(frame) < 4:
            raise ValueError("At least four valid rows are required for train/validation splitting.")

        geometry = frame[GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
        conditions = frame[CONDITION_COLUMNS].to_numpy(dtype=np.float32)
        metrics = frame[METRIC_COLUMNS].to_numpy(dtype=np.float32)
        groups = geometry_group_labels(frame)
        if np.unique(groups).size < 2:
            raise ValueError("At least two unique base geometries are required for leakage-resistant validation.")

        splitter = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.seed)
        train_idx, val_idx = next(splitter.split(geometry, metrics, groups=groups))

        self.geometry_scaler.fit(geometry[train_idx])
        self.condition_scaler.fit(conditions[train_idx])
        self.metric_scaler.fit(metrics[train_idx])

        def build_dataset(indices: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.tensor(self.geometry_scaler.transform(geometry[indices]), dtype=torch.float32),
                torch.tensor(self.condition_scaler.transform(conditions[indices]), dtype=torch.float32),
                torch.tensor(self.metric_scaler.transform(metrics[indices]), dtype=torch.float32),
            )

        self.train_dataset = build_dataset(train_idx)
        self.val_dataset = build_dataset(val_idx)
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.groups = groups

    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
