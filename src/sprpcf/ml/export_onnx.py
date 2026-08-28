from __future__ import annotations
import argparse
from pathlib import Path
import torch
from sprpcf.ml.onnx_export import export_inverse_generator_onnx
from sprpcf.ml.tandem import InverseGenerator
def export_checkpoint_inverse(checkpoint_path:Path,output_path:Path):
    cp=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if int(cp.get("schema_version",1))<2: raise ValueError("Legacy checkpoint detected. Retrain with corrected schema.")
    inv=InverseGenerator(design_dim=len(cp["design_columns"])); inv.load_state_dict(cp["inverse_state_dict"]); export_inverse_generator_onnx(inv,output_path,cp["metric_mean"],cp["metric_scale"],cp["design_mean"],cp["design_scale"])
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); export_checkpoint_inverse(a.checkpoint,a.out)
if __name__=="__main__":main()
