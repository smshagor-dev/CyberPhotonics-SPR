"""Synthetic-only software regression tests, not optical evidence."""
import pytest
from sprpcf.simulation.resonance_gate import bracketed_resonance

def test_symmetric_peak():
    r = bracketed_resonance([.70,.72,.74,.76,.78,.80,.82], [0,1,2,4,2,1,0])
    assert r["fwhm_diagnostic_um"] == pytest.approx(.04)
    assert r["physics_qualified"] is False
    assert r["spectral_resolution_verified"] is False

def test_three_point_grid_not_resolved():
    with pytest.raises(ValueError, match="seven"):
        bracketed_resonance([.75,.80,.85], [1,2,1])

def test_flat_peak_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        bracketed_resonance([1,2,3,4,5,6,7], [0,1,3,3,3,1,0])

def test_multiple_peaks_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        bracketed_resonance([1,2,3,4,5,6,7], [0,2,0,1,0,2,0])

def test_nan_rejected():
    with pytest.raises(ValueError, match="finite"):
        bracketed_resonance([1,2,3,4,5,6,7], [0,1,2,float("nan"),2,1,0])
