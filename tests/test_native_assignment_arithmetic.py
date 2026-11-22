"""Software-only adversarial tests; these fixtures are never physics evidence."""
from copy import deepcopy
import pytest
from research.native_assignment_arithmetic import verify

def fixture():
    edges=[]
    for i in range(3):
        for j in range(2):
            edges.append(edge([i,j],[i,j+1],"wavelength"))
    for i in range(2):
        for j in range(3):
            edges.append(edge([i,j],[i+1,j],"analyte_ri"))
    return {"source_sha256":"a"*64,"edges":edges}

def edge(a,b,axis):
    return {"from":a,"to":b,"axis":axis,"similarity":[[1.,0.],[0.,1.]],
            "permutation":[0,1],"assignment_margin":2.,
            "matched_min":1.,"matched_mean":1.}

def test_complete_grid_is_unqualified():
    r=verify(fixture())
    assert r["edges_checked"]==12 and r["consistent_plaquettes"]==4
    assert r["physics_qualified"] is False and r["publication_ready"] is False
    assert r["comsol_verified"] is False

@pytest.mark.parametrize("mutation",[
    lambda d:d["edges"][0].update(assignment_margin=float("nan")),
    lambda d:d["edges"][0].update(permutation=[1,0]),
    lambda d:d["edges"][0].update(similarity=[[1.1,0],[0,1]]),
    lambda d:d["edges"][0].update(matched_min=0),
    lambda d:d["edges"].append(deepcopy(d["edges"][0])),
    lambda d:d.update(source_sha256="not-a-sha"),
    lambda d:d["edges"][0].update(axis="analyte_ri"),
    lambda d:d["edges"].pop(),
])
def test_corrupt_grid_rejected(mutation):
    d=fixture();mutation(d)
    with pytest.raises(ValueError): verify(d)
