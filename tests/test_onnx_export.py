from __future__ import annotations
import importlib.util,pytest
from sprpcf.ml.onnx_export import export_inverse_generator_onnx
from sprpcf.ml.tandem import InverseGenerator
pytestmark=pytest.mark.skipif(importlib.util.find_spec("onnx") is None,reason="onnx is not installed")
def test_inverse_generator_onnx_has_condition_input(tmp_path):
    import onnx
    out=tmp_path/"inverse.onnx";export_inverse_generator_onnx(InverseGenerator(),out,[800,20,650],[100,5,50],[2,.5,45,.6],[.5,.1,10,.1]);model=onnx.load(out);onnx.checker.check_model(model);assert [i.name for i in model.graph.input]==["target_metrics","analyte_ri"];assert model.graph.output[0].name=="geometry"
