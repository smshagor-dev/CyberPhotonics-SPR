import pytest
from scripts.spectral_resolution_audit import audit_spectrum

def test_resolved():
    x=[600+i for i in range(101)]
    y=[1+10/(1+((v-650)/12)**2) for v in x]
    out=audit_spectrum(x,y)
    assert out["spectral_resolution_gate"]
    assert out["sampled_peak_nm"]==650
    assert not out["physics_qualified"]

def test_underresolved():
    x=[600+10*i for i in range(11)]
    y=[1+10/(1+((v-650)/12)**2) for v in x]
    assert not audit_spectrum(x,y)["spectral_resolution_gate"]

@pytest.mark.parametrize("x,y", [
    ([1,2,3],[1,2,1]),
    ([1,2,2,3,4,5,6],[1,2,3,4,3,2,1]),
    ([1,2,3,4,5,6,7],[1,2,3,float("nan"),3,2,1]),
    ([1,2,3,4,5,6,7],[1,2,3,4,5,6,7])
])
def test_reject(x,y):
    with pytest.raises(ValueError):
        audit_spectrum(x,y)
