from __future__ import annotations
import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self,in_features,out_features,hidden,dropout=0.0):
        super().__init__(); layers=[]; previous=in_features
        for width in hidden:
            layers.extend([nn.Linear(previous,width),nn.SiLU(),nn.LayerNorm(width)])
            if dropout>0: layers.append(nn.Dropout(dropout))
            previous=width
        layers.append(nn.Linear(previous,out_features)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)
class ForwardNetwork(nn.Module):
    def __init__(self,input_dim=5,metric_dim=3): super().__init__(); self.model=MLP(input_dim,metric_dim,(128,128,64))
    def forward(self,x): return self.model(x)
class InverseGenerator(nn.Module):
    def __init__(self,metric_dim=3,condition_dim=1,design_dim=4,latent_dim=4): super().__init__(); self.latent_dim=latent_dim; self.model=MLP(metric_dim+condition_dim+latent_dim,design_dim,(128,128,64),0.10)
    def forward(self,metrics,condition,latent=None):
        if condition.ndim==1: condition=condition[:,None]
        if latent is None: latent=torch.zeros(metrics.shape[0],self.latent_dim,dtype=metrics.dtype,device=metrics.device)
        return self.model(torch.cat([metrics,condition,latent],dim=-1))
class TandemNetwork(nn.Module):
    def __init__(self,forward_model,inverse_model,forward_input_mean,forward_input_scale,design_mean,design_scale):
        super().__init__(); self.forward_model=forward_model; self.inverse_model=inverse_model; self.register_buffer("forward_input_mean",forward_input_mean); self.register_buffer("forward_input_scale",forward_input_scale); self.register_buffer("design_mean",design_mean); self.register_buffer("design_scale",design_scale)
    def forward(self,target_metrics,analyte_ri,latent=None):
        sd=self.inverse_model(target_metrics,analyte_ri,latent); pd=sd*self.design_scale+self.design_mean
        if analyte_ri.ndim==1: analyte_ri=analyte_ri[:,None]
        pf=torch.cat([pd,analyte_ri],dim=-1); pm=self.forward_model((pf-self.forward_input_mean)/self.forward_input_scale); return sd,pm
