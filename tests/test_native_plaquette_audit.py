"""Regression tests for exploratory plaquette assignment consistency."""
from copy import deepcopy
import pytest
from research.native_plaquette_audit import compose, plaquettes, evaluate

def grid():
    edges=[]
    for i in range(3):
        for j in range(2):
            edges.append({"from":[i,j],"to":[i,j+1],"permutation":[0,1],"assignment_margin":0.1})
    for j in range(3):
        for i in range(2):
            edges.append({"from":[i,j],"to":[i+1,j],"permutation":[0,1],"assignment_margin":0.1})
    return {"source_sha256":"test-only", "edges":edges}

def test_identity():
    result=evaluate(grid())
    assert result["consistent_count"]==4
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False

def test_inconsistent_path():
    data=grid()
    data["edges"][0]["permutation"]=[1,0]
    assert sum(not x["consistent"] for x in plaquettes(data))==1

def test_missing_edge():
    data=grid()
    data["edges"].pop()
    with pytest.raises(ValueError,match="Missing"):
        plaquettes(data)

def test_invalid_permutation():
    with pytest.raises(ValueError,match="Invalid"):
        compose([0,0],[0,1])

def test_duplicate_edge():
    data=grid()
    data["edges"].append(deepcopy(data["edges"][0]))
    with pytest.raises(ValueError,match="Duplicate"):
        plaquettes(data)
