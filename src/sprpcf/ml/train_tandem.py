from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from torch import optim
from tqdm import trange
from sprpcf.ml.dataset import CONDITION_COLUMNS,DESIGN_COLUMNS,FORWARD_INPUT_COLUMNS,METRIC_COLUMNS,DesignDataModule
from sprpcf.ml.losses import geometry_constraint_loss
from sprpcf.ml.onnx_export import export_inverse_generator_onnx
from sprpcf.ml.tandem import ForwardNetwork,InverseGenerator,TandemNetwork

def set_reproducible_seed(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
def _select_device(name):
    if name=="auto": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=torch.device(name)
    if d.type=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA was requested but is not available.")
    return d
def _metrics(a,b): return {"r2":float(r2_score(a,b,multioutput="uniform_average")),"rmse":float(np.sqrt(mean_squared_error(a,b))),"mae":float(mean_absolute_error(a,b))}
def evaluate_forward(model,loader,device):
    model.eval(); t=[];p=[]
    with torch.no_grad():
        for x,_,_,y in loader: p.append(model(x.to(device)).cpu().numpy()); t.append(y.numpy())
    return _metrics(np.concatenate(t),np.concatenate(p))
def evaluate_inverse_geometry(inv,loader,data,device):
    inv.eval();t=[];p=[]
    with torch.no_grad():
        for _,d,c,m in loader: p.append(data.design_scaler.inverse_transform(inv(m.to(device),c.to(device)).cpu().numpy())); t.append(data.design_scaler.inverse_transform(d.numpy()))
    return _metrics(np.concatenate(t),np.concatenate(p))
def train_forward(data,epochs,lr,device):
    model=ForwardNetwork(len(FORWARD_INPUT_COLUMNS)).to(device); opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    for _ in trange(epochs,desc="forward"):
        model.train()
        for x,_,_,y in data.train_loader():
            loss=F.mse_loss(model(x.to(device)),y.to(device)); opt.zero_grad(); loss.backward(); opt.step()
    return model
def train_inverse(data,forward,epochs,lr,device,alpha,beta,dispersion_weight=0.0):
    inv=InverseGenerator(design_dim=len(DESIGN_COLUMNS)).to(device); forward.eval(); [setattr(p,"requires_grad",False) for p in forward.parameters()]; opt=optim.AdamW(inv.parameters(),lr=lr,weight_decay=1e-4)
    dm=torch.tensor(data.design_scaler.mean_,dtype=torch.float32,device=device); ds=torch.tensor(data.design_scaler.scale_,dtype=torch.float32,device=device); fm=torch.tensor(data.forward_input_scaler.mean_,dtype=torch.float32,device=device); fs=torch.tensor(data.forward_input_scaler.scale_,dtype=torch.float32,device=device); lm=torch.tensor(data.metric_scaler.mean_[2],dtype=torch.float32,device=device); ls=torch.tensor(data.metric_scaler.scale_[2],dtype=torch.float32,device=device)
    for _ in trange(epochs,desc="inverse"):
        inv.train()
        for _,_,c,target in data.train_loader():
            c,target=c.to(device),target.to(device); sd=inv(target,c); pd=sd*ds+dm; pred=forward((torch.cat([pd,c],1)-fm)/fs); loss=F.mse_loss(pred,target)+geometry_constraint_loss(pd,alpha,beta,pred[:,2]*ls+lm,dispersion_weight); opt.zero_grad(); loss.backward(); opt.step()
    return inv
def train_tandem_pipeline(data_path,checkpoint_out,onnx_out,epochs=50,forward_epochs=None,inverse_epochs=None,batch_size=64,lr=1e-3,device_name="auto",alpha=1.0,beta=1e-3,dispersion_weight=0.0,seed=7):
    set_reproducible_seed(seed); device=_select_device(device_name); data=DesignDataModule(data_path,batch_size=batch_size,seed=seed); data.setup(); forward=train_forward(data,forward_epochs or epochs,lr,device); fm=evaluate_forward(forward,data.val_loader(),device); inv=train_inverse(data,forward,inverse_epochs or epochs,lr,device,alpha,beta,dispersion_weight); im=evaluate_inverse_geometry(inv,data.val_loader(),data,device)
    tandem=TandemNetwork(forward,inv,torch.tensor(data.forward_input_scaler.mean_,dtype=torch.float32),torch.tensor(data.forward_input_scaler.scale_,dtype=torch.float32),torch.tensor(data.design_scaler.mean_,dtype=torch.float32),torch.tensor(data.design_scaler.scale_,dtype=torch.float32)); checkpoint_out.parent.mkdir(parents=True,exist_ok=True)
    cp={"schema_version":2,"seed":seed,"model":tandem.state_dict(),"forward_state_dict":forward.state_dict(),"inverse_state_dict":inv.state_dict(),"forward_metrics_standardized":fm,"inverse_geometry_metrics_physical":im,"design_columns":DESIGN_COLUMNS,"condition_columns":CONDITION_COLUMNS,"forward_input_columns":FORWARD_INPUT_COLUMNS,"metric_columns":METRIC_COLUMNS,"design_mean":data.design_scaler.mean_,"design_scale":data.design_scaler.scale_,"forward_input_mean":data.forward_input_scaler.mean_,"forward_input_scale":data.forward_input_scaler.scale_,"metric_mean":data.metric_scaler.mean_,"metric_scale":data.metric_scaler.scale_}; torch.save(cp,checkpoint_out); export_inverse_generator_onnx(inv,onnx_out,cp["metric_mean"],cp["metric_scale"],cp["design_mean"],cp["design_scale"]); return {"forward":fm,"inverse_geometry":im}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--epochs",type=int,default=50); p.add_argument("--forward-epochs",type=int); p.add_argument("--inverse-epochs",type=int); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--device",default="auto"); p.add_argument("--alpha",type=float,default=1.0); p.add_argument("--beta",type=float,default=1e-3); p.add_argument("--dispersion-weight",type=float,default=0.0); p.add_argument("--seed",type=int,default=7); p.add_argument("--out",type=Path,default=Path("models/tandem.pt")); p.add_argument("--onnx-out",type=Path,default=Path("models/inverse_pcf_spr.onnx")); a=p.parse_args(); print(json.dumps(train_tandem_pipeline(a.data,a.out,a.onnx_out,a.epochs,a.forward_epochs,a.inverse_epochs,a.batch_size,a.lr,a.device,a.alpha,a.beta,a.dispersion_weight,a.seed),indent=2))
if __name__=="__main__": main()
