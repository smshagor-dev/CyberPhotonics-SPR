from __future__ import annotations
import numpy as np,pandas as pd,pytest
from sprpcf.simulation.metrics import assign_grouped_sensitivity,extract_metrics,finite_difference_sensitivity,resonance_wavelength
def test_resonance_wavelength_finds_peak():
    w=np.linspace(500,700,201);l=1/(1+((w-612)/10)**2);lam,peak=resonance_wavelength(w,l);assert abs(lam-612)<=1;assert peak>.99
def test_extract_metrics_returns_positive_fwhm():
    w=np.linspace(500,700,201);l=1/(1+((w-612)/10)**2);m=extract_metrics(w,l,1000);assert m.fwhm_nm>0;assert m.fom_per_riu is not None
def test_sensitivity_rejects_duplicate_ri():
    with pytest.raises(ValueError): finite_difference_sensitivity(np.array([600.,610.]),np.array([1.33,1.33]))
def test_grouped_sensitivity_does_not_mix_geometries():
    rows=[]
    for pitch,offset in [(1.5,0.),(2.5,100.)]:
        for ri in [1.33,1.35,1.37]: rows.append({"d_over_lambda":.5,"pitch_um":pitch,"metal_thickness_nm":45.,"channel_radius_um":.6,"analyte_ri":ri,"lambda_res_nm":600+offset+1000*(ri-1.33),"fwhm_nm":20.})
    out=assign_grouped_sensitivity(pd.DataFrame(rows));assert np.allclose(out["sensitivity_nm_per_riu"],1000);assert np.allclose(out["fom_per_riu"],50)
