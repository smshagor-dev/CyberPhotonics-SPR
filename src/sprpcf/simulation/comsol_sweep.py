from __future__ import annotations
import argparse,logging
from pathlib import Path
import numpy as np,pandas as pd,yaml
from sprpcf.simulation.metrics import assign_grouped_sensitivity,extract_metrics
from sprpcf.simulation.schema import Geometry
LOGGER=logging.getLogger(__name__)
def _load_mph():
    try: import mph
    except ImportError as exc: raise RuntimeError("The 'mph' package is required for COMSOL automation.") from exc
    return mph
def _grid(v): return [Geometry(float(d),float(p),float(m),float(n),float(c)) for d in v["d_over_lambda"] for p in v["pitch_um"] for m in v["metal_thickness_nm"] for n in v["analyte_ri"] for c in v.get("channel_radius_um",[0.6])]
def _config(path):
    cfg=yaml.safe_load(path.read_text()) or {}; cfg.setdefault("study","std1"); cfg.setdefault("wavelength_expression","lambda"); cfg.setdefault("loss_expression","loss"); cfg.setdefault("wavelength_scale_to_nm",1.0); cfg.setdefault("loss_scale_to_db_per_cm",1.0); return cfg
def run_comsol_geometries(model_path,config_path,geometries):
    cfg=_config(config_path); client=_load_mph().start(); model=client.load(str(model_path)); rows=[]
    for i,g in enumerate(geometries):
        try:
            model.parameter("d_over_lambda",g.d_over_lambda); model.parameter("pitch_um",f"{g.pitch_um}[um]"); model.parameter("metal_thickness_nm",f"{g.metal_thickness_nm}[nm]"); model.parameter("analyte_ri",g.analyte_ri); model.parameter("channel_radius_um",f"{g.channel_radius_um}[um]"); model.solve(cfg["study"]); wavelength=np.asarray(model.evaluate(cfg["wavelength_expression"]),dtype=float).ravel()*float(cfg["wavelength_scale_to_nm"]); loss=np.asarray(model.evaluate(cfg["loss_expression"]),dtype=float).ravel()*float(cfg["loss_scale_to_db_per_cm"]); met=extract_metrics(wavelength,loss); rows.append({"sample_id":i,"status":"ok",**g.__dict__,**met.__dict__,"wavelength_nm":",".join(f"{x:.6f}" for x in wavelength),"loss_db_per_cm":",".join(f"{x:.6f}" for x in loss),"source":"comsol"})
        except Exception as exc: LOGGER.exception("COMSOL sample %s failed",i); rows.append({"sample_id":i,"status":f"failed: {exc}",**g.__dict__})
    frame=pd.DataFrame(rows)
    if not frame.empty:
        ok=frame["status"].eq("ok")
        if ok.any():
            corrected=assign_grouped_sensitivity(frame.loc[ok].copy()); frame.loc[corrected.index,corrected.columns]=corrected
    return frame
def run_comsol_sweep(model_path,config_path): return run_comsol_geometries(model_path,config_path,_grid(_config(config_path)["sweep"]))
def run_comsol_candidates(model_path,config_path,candidates):
    cols=["d_over_lambda","pitch_um","metal_thickness_nm","analyte_ri","channel_radius_um"]; missing=[c for c in cols if c not in candidates]
    if missing: raise ValueError(f"Missing candidate columns: {missing}")
    return run_comsol_geometries(model_path,config_path,[Geometry(*(float(row[c]) for c in cols)) for _,row in candidates.iterrows()])
def write_dataset(frame,output): output.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(output,index=False) if output.suffix.lower()==".parquet" else frame.to_csv(output,index=False)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); write_dataset(run_comsol_sweep(a.model,a.config),a.out)
if __name__=="__main__":main()
