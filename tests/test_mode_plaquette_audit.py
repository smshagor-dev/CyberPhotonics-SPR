"""Synthetic assignment permutations only; not Maxwell evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pytest
from mode_plaquette_audit import audit_mode_plaquettes

def edges(n=2):
    out=[]
    for ri in range(n):
        for wl in range(n-1):
            for mode in range(2):
                out.append(dict(varying="wavelength",ri_index=ri,previous_wl_index=wl,current_wl_index=wl+1,previous_mode_index=mode,current_mode_index=mode))
    for wl in range(n):
        for ri in range(n-1):
            for mode in range(2):
                out.append(dict(varying="analyte_ri",wavelength_index=wl,previous_ri_index=ri,current_ri_index=ri+1,previous_mode_index=mode,current_mode_index=mode))
    return out

def test_consistent_3x3():
    r=audit_mode_plaquettes(edges(3),n_ri=3,n_wavelength=3,n_modes=2)
    assert r["plaquettes_checked"]==4 and r["path_independent_assignment"]
    assert r["physics_qualified"] is False

def test_cycle_inconsistent_but_bijective():
    e=edges()
    for x in e:
        if x["varying"]=="wavelength" and x["ri_index"]==1:
            x["current_mode_index"]=1-x["previous_mode_index"]
    r=audit_mode_plaquettes(e,n_ri=2,n_wavelength=2,n_modes=2)
    assert r["holonomy_defects"]==[{"ri_index":0,"wavelength_index":0,"mismatched_start_modes":[0,1]}]

def test_missing_edge_rejected():
    with pytest.raises(ValueError,match="non-bijective"):
        audit_mode_plaquettes(edges()[:-1],n_ri=2,n_wavelength=2,n_modes=2)

def test_duplicate_rejected():
    with pytest.raises(ValueError,match="duplicate"):
        audit_mode_plaquettes(edges()+[edges()[0]],n_ri=2,n_wavelength=2,n_modes=2)

def test_no_plaquettes_one_dimensional():
    e=[dict(varying="wavelength",ri_index=0,previous_wl_index=0,current_wl_index=1,previous_mode_index=i,current_mode_index=i) for i in range(2)]
    r=audit_mode_plaquettes(e,n_ri=1,n_wavelength=2,n_modes=2)
    assert r["plaquettes_checked"]==0 and not r["physics_qualified"]
