from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

GEOMETRY_COLUMNS = ["pitch_um", "d_over_lambda", "metal_thickness_nm"]
METRIC_COLUMNS = ["sensitivity_nm_per_riu", "fom_per_riu", "lambda_res_nm"]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


class DesignDataModule:
    """Prepare standardized geometry and metric tensors for tandem learning."""

    def __init__(self, path: Path, batch_size: int = 64, test_size: float = 0.2, seed: int = 7) -> None:
        self.path = path
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed
        self.geometry_scaler = StandardScaler()
        self.metric_scaler = StandardScaler()

    def setup(self) -> None:
        frame = read_table(self.path).dropna(subset=GEOMETRY_COLUMNS + METRIC_COLUMNS)
        geometry = frame[GEOMETRY_COLUMNS].to_numpy(dtype=np.float32)
        metrics = frame[METRIC_COLUMNS].to_numpy(dtype=np.float32)

        x_train, x_val, y_train, y_val = train_test_split(
            geometry,
            metrics,
            test_size=self.test_size,
            random_state=self.seed,
        )
        self.geometry_scaler.fit(x_train)
        self.metric_scaler.fit(y_train)
        self.train_dataset = TensorDataset(
            torch.tensor(self.geometry_scaler.transform(x_train), dtype=torch.float32),
            torch.tensor(self.metric_scaler.transform(y_train), dtype=torch.float32),
        )
        self.val_dataset = TensorDataset(
            torch.tensor(self.geometry_scaler.transform(x_val), dtype=torch.float32),
            torch.tensor(self.metric_scaler.transform(y_val), dtype=torch.float32),
        )

    def train_loader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_loader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
