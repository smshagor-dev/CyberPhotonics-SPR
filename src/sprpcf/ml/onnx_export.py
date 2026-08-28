from __future__ import annotations
from pathlib import Path
from typing import Iterable
import torch
from torch import nn
from sprpcf.ml.tandem import InverseGenerator
class PhysicalInverseGenerator(nn.Module):
    def __init__(self,inverse,metric_mean,metric_scale,design_mean,design_scale): super().__init__(); self.inverse=inverse; self.register_buffer("metric_mean",torch.tensor(list(metric_mean),dtype=torch.float32)); self.register_buffer("metric_scale",torch.tensor(list(metric_scale),dtype=torch.float32)); self.register_buffer("design_mean",torch.tensor(list(design_mean),dtype=torch.float32)); self.register_buffer("design_scale",torch.tensor(list(design_scale),dtype=torch.float32))
    def forward(self,physical_metrics,analyte_ri): return self.inverse((physical_metrics-self.metric_mean)/self.metric_scale,analyte_ri)*self.design_scale+self.design_mean
def export_inverse_generator_onnx(inverse:InverseGenerator,output_path:Path,metric_mean:Iterable[float],metric_scale:Iterable[float],design_mean:Iterable[float],design_scale:Iterable[float],opset:int=17):
    wrapper=PhysicalInverseGenerator(inverse.cpu().eval(),metric_mean,metric_scale,design_mean,design_scale).eval(); output_path.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,(torch.tensor([list(metric_mean)],dtype=torch.float32),torch.tensor([[1.35]],dtype=torch.float32)),output_path,input_names=["target_metrics","analyte_ri"],output_names=["geometry"],dynamic_axes={"target_metrics":{0:"batch"},"analyte_ri":{0:"batch"},"geometry":{0:"batch"}},opset_version=opset)
