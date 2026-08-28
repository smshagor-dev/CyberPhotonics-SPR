from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from sprpcf.ml.dataset import FORWARD_INPUT_COLUMNS,METRIC_COLUMNS,read_table
from sprpcf.ml.tandem import ForwardNetwork
def integrated_gradients(model,inputs,target_index=0,baseline=None,steps=64):
    if steps<2: raise ValueError("steps must be >= 2")
    model.eval(); baseline=torch.zeros_like(inputs) if baseline is None else baseline; grads=[]
    for step in range(1,steps+1):
        x=(baseline+(step/steps)*(inputs-baseline)).detach().requires_grad_(True); grads.append(torch.autograd.grad(model(x)[:,target_index].sum(),x)[0])
    return (inputs-baseline)*torch.stack(grads).mean(0)
def attribution_matrix(model,standardized_inputs,target_index=0,steps=64): return pd.DataFrame(integrated_gradients(model,torch.tensor(standardized_inputs,dtype=torch.float32),target_index,steps=steps).detach().numpy(),columns=FORWARD_INPUT_COLUMNS)
def load_forward_model(checkpoint_path):
    cp=torch.load(checkpoint_path,map_location="cpu",weights_only=False); model=ForwardNetwork(input_dim=len(cp["forward_input_columns"])); model.load_state_dict(cp["forward_state_dict"]); model.eval(); return model,np.asarray(cp["forward_input_mean"],np.float32),np.asarray(cp["forward_input_scale"],np.float32)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--target",choices=METRIC_COLUMNS,default=METRIC_COLUMNS[0]); p.add_argument("--out",type=Path,default=Path("outputs/feature_attribution.csv")); p.add_argument("--limit",type=int,default=128); a=p.parse_args(); model,mean,scale=load_forward_model(a.checkpoint); frame=read_table(a.data).dropna(subset=FORWARD_INPUT_COLUMNS); x=frame[FORWARD_INPUT_COLUMNS].to_numpy(np.float32)[:a.limit]; matrix=attribution_matrix(model,(x-mean)/scale,METRIC_COLUMNS.index(a.target)); a.out.parent.mkdir(parents=True,exist_ok=True); matrix.to_csv(a.out,index=False)
if __name__=="__main__":main()
