from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

DESIGN_COLUMNS = ["pitch_um", "d_over_lambda", "metal_thickness_nm", "channel_radius_um"]
CONDITION_COLUMNS = ["analyte_ri"]
FORWARD_INPUT_COLUMNS = DESIGN_COLUMNS + CONDITION_COLUMNS
GEOMETRY_COLUMNS = DESIGN_COLUMNS
METRIC_COLUMNS = ["sensitivity_nm_per_riu", "fom_per_riu", "lambda_res_nm"]

def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)

class DesignDataModule:
    def __init__(self, path: Path, batch_size: int = 64, test_size: float = 0.2, seed: int = 7):
        self.path=path; self.batch_size=batch_size; self.test_size=test_size; self.seed=seed
        self.forward_input_scaler=StandardScaler(); self.design_scaler=StandardScaler(); self.metric_scaler=StandardScaler()
    def setup(self):
        frame=read_table(self.path).dropna(subset=FORWARD_INPUT_COLUMNS+METRIC_COLUMNS).copy()
        if len(frame)<5: raise ValueError("At least five complete samples are required for tandem training.")
        fwd=frame[FORWARD_INPUT_COLUMNS].to_numpy(np.float32); design=frame[DESIGN_COLUMNS].to_numpy(np.float32); cond=frame[CONDITION_COLUMNS].to_numpy(np.float32); metrics=frame[METRIC_COLUMNS].to_numpy(np.float32); idx=np.arange(len(frame))
        groups=frame["geometry_id"].to_numpy() if "geometry_id" in frame.columns else None
        if groups is not None and np.unique(groups).size>=2:
            train_groups,val_groups=train_test_split(np.unique(groups),test_size=self.test_size,random_state=self.seed); train_idx=idx[np.isin(groups,train_groups)]; val_idx=idx[np.isin(groups,val_groups)]
        else:
            train_idx,val_idx=train_test_split(idx,test_size=self.test_size,random_state=self.seed)
        self.forward_input_scaler.fit(fwd[train_idx]); self.design_scaler.fit(design[train_idx]); self.metric_scaler.fit(metrics[train_idx])
        def build(s):
            return TensorDataset(torch.tensor(self.forward_input_scaler.transform(fwd[s]),dtype=torch.float32),torch.tensor(self.design_scaler.transform(design[s]),dtype=torch.float32),torch.tensor(cond[s],dtype=torch.float32),torch.tensor(self.metric_scaler.transform(metrics[s]),dtype=torch.float32))
        self.train_dataset=build(np.asarray(train_idx)); self.val_dataset=build(np.asarray(val_idx))
    def train_loader(self): return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
    def val_loader(self): return DataLoader(self.val_dataset,batch_size=self.batch_size)
