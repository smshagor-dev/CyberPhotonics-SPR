from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np,pandas as pd,torch
from sprpcf.ml.dataset import DESIGN_COLUMNS,METRIC_COLUMNS,read_table
from sprpcf.ml.tandem import InverseGenerator
from sprpcf.simulation.comsol_sweep import run_comsol_candidates,write_dataset
@dataclass(frozen=True)
class ActiveLearningResult: candidate_metrics:pd.DataFrame; uncertainty:np.ndarray; selected:pd.DataFrame; comsol_results:pd.DataFrame|None=None
def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m,torch.nn.Dropout): m.train()
def mc_dropout_inverse_uncertainty(inverse,target_metrics,condition,passes=32):
    if passes<2: raise ValueError("passes must be >= 2")
    enable_mc_dropout(inverse); preds=[]
    with torch.no_grad():
        for _ in range(passes): preds.append(inverse(target_metrics,condition))
    s=torch.stack(preds); return s.mean(0),s.std(0)
def select_uncertain_candidates(inverse,candidate_metrics,metric_mean,metric_scale,design_mean,design_scale,uncertainty_threshold,passes=32,device="cpu"):
    required=METRIC_COLUMNS+["analyte_ri"]; missing=[c for c in required if c not in candidate_metrics]
    if missing: raise ValueError(f"Missing candidate columns: {missing}")
    dev=torch.device(device); inverse=inverse.to(dev); metrics=candidate_metrics[METRIC_COLUMNS].to_numpy(np.float32); standardized=(metrics-np.asarray(metric_mean,np.float32))/np.asarray(metric_scale,np.float32); cond=candidate_metrics[["analyte_ri"]].to_numpy(np.float32); mean,std=mc_dropout_inverse_uncertainty(inverse,torch.tensor(standardized,device=dev),torch.tensor(cond,device=dev),passes); uncertainty=std.norm(dim=1).cpu().numpy(); physical=mean.cpu().numpy()*np.asarray(design_scale)+np.asarray(design_mean); enriched=candidate_metrics.copy()
    for i,c in enumerate(DESIGN_COLUMNS): enriched[c]=physical[:,i]
    selected=enriched.loc[uncertainty>uncertainty_threshold].copy(); selected["uncertainty"]=uncertainty[uncertainty>uncertainty_threshold]; return ActiveLearningResult(enriched,uncertainty,selected)
def load_checkpoint_inverse(path):
    cp=torch.load(path,map_location="cpu",weights_only=False)
    if int(cp.get("schema_version",1))<2: raise ValueError("Legacy checkpoint; retrain before active learning.")
    inv=InverseGenerator(design_dim=len(cp["design_columns"])); inv.load_state_dict(cp["inverse_state_dict"]); return inv,np.asarray(cp["metric_mean"],np.float32),np.asarray(cp["metric_scale"],np.float32),np.asarray(cp["design_mean"],np.float32),np.asarray(cp["design_scale"],np.float32)
def run_active_learning_iteration(checkpoint_path,candidate_path,uncertainty_threshold,passes=32):
    inv,mm,ms,dm,ds=load_checkpoint_inverse(checkpoint_path); return select_uncertain_candidates(inv,read_table(candidate_path),mm,ms,dm,ds,uncertainty_threshold,passes)
def trigger_comsol_for_uncertain_candidates(result,model_path,config_path,output_path):
    if result.selected.empty:return result
    cr=run_comsol_candidates(model_path,config_path,result.selected); write_dataset(cr,output_path); return ActiveLearningResult(result.candidate_metrics,result.uncertainty,result.selected,cr)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--candidates",type=Path,required=True); p.add_argument("--threshold",type=float,default=.05); p.add_argument("--passes",type=int,default=32); p.add_argument("--out",type=Path,default=Path("outputs/uncertain_candidates.csv")); a=p.parse_args(); r=run_active_learning_iteration(a.checkpoint,a.candidates,a.threshold,a.passes); a.out.parent.mkdir(parents=True,exist_ok=True); r.selected.to_csv(a.out,index=False)
if __name__=="__main__":main()
