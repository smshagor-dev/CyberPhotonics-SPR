"""Synthetic fixtures test integrity only, never Maxwell physics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import pytest
from overlap_graph_audit import audit_overlap_graph

def fixture_edges():
    edges=[]
    for axis,fixed in (("wavelength",0),("wavelength",1),("analyte_ri",0),("analyte_ri",1)):
        for mode in range(2):
            e={"varying":axis,"previous_mode_index":mode,"current_mode_index":mode,
               "vector_eh_poynting_overlap_abs":0.9,"unselected_overlap_abs":0.2,
               "reverse_competing_overlap_abs":0.3,"forward_overlap_margin":0.7,
               "reverse_overlap_margin":0.6,"bidirectional_min_overlap_margin":0.6,
               "assignment_ambiguous_nonpositive_margin":False,"mode_identity_physically_verified":False}
            if axis=="wavelength":
                e.update(ri_index=fixed,previous_wl_index=0,current_wl_index=1)
            else:
                e.update(wavelength_index=fixed,previous_ri_index=0,current_ri_index=1)
            edges.append(e)
    return edges

def test_fixture_does_not_qualify_physics():
    report=audit_overlap_graph(fixture_edges(),n_ri=2,n_wavelength=2,n_modes=2)
    assert report["matched_edges"]==8 and report["physics_qualified"] is False

@pytest.mark.parametrize("change",["duplicate","missing","margin","flag","nonadjacent","identity"])
def test_tampering_fails_closed(change):
    edges=fixture_edges()
    if change=="duplicate":edges[1]["current_mode_index"]=0
    if change=="missing":edges.pop()
    if change=="margin":edges[0]["forward_overlap_margin"]=0.5
    if change=="flag":edges[0]["assignment_ambiguous_nonpositive_margin"]=True
    if change=="nonadjacent":edges[0]["current_wl_index"]=3
    if change=="identity":edges[0]["mode_identity_physically_verified"]=True
    with pytest.raises(ValueError):
        audit_overlap_graph(edges,n_ri=2,n_wavelength=2,n_modes=2)


def test_full_three_by_three_grid_matches_native_campaign_cardinality():
    """Exercise the real campaign's 3x3x2 graph shape; data remain synthetic."""
    edges=[]
    for ri in range(3):
        for wl in range(1,3):
            for mode in range(2):
                edge=dict(fixture_edges()[0])
                edge.update(ri_index=ri, previous_wl_index=wl-1,
                            current_wl_index=wl, previous_mode_index=mode,
                            current_mode_index=mode)
                edges.append(edge)
    for wl in range(3):
        for ri in range(1,3):
            for mode in range(2):
                edge=dict(fixture_edges()[4])
                edge.update(wavelength_index=wl, previous_ri_index=ri-1,
                            current_ri_index=ri, previous_mode_index=mode,
                            current_mode_index=mode)
                edges.append(edge)
    report=audit_overlap_graph(edges,n_ri=3,n_wavelength=3,n_modes=2)
    assert report["adjacency_pairs"]==12
    assert report["matched_edges"]==24
    assert report["ambiguous_edges"]==0
    assert report["physics_qualified"] is False
    with pytest.raises(ValueError, match="missing adjacency"):
        audit_overlap_graph(edges[:-2],n_ri=3,n_wavelength=3,n_modes=2)
