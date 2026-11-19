"""Software-only fixtures: not Maxwell physics evidence."""
import pytest
from sprpcf.simulation.spectral_refinement_audit import audit_spectral_refinement

def spectrum(step, **changes):
    xs=[600+i*step for i in range(int(100/step)+1)]
    ys=[1+9/(1+((x-650)/10)**2) for x in xs]
    row=dict(wavelength_nm=xs,loss_db_cm=ys,geometry_sha256="geometry",
             material_sha256="material",mode_branch_id="core0",ri=1.33,
             pml_sha256="pml",solver_name="femwell")
    row.update(changes)
    return row

def run(rows):
    return audit_spectral_refinement(rows,peak_tol_nm=2,fwhm_relative_tol=.1)

def test_stable_sampling_does_not_qualify_physics():
    result=run([spectrum(5),spectrum(2),spectrum(1)])
    assert result["diagnostic_pass"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False

def test_changed_ri_fails_closed():
    with pytest.raises(ValueError,match="mismatch"):
        run([spectrum(5),spectrum(2),spectrum(1,ri=1.34)])

def test_nonrefining_grid_rejected():
    with pytest.raises(ValueError,match="strictly refined"):
        run([spectrum(5),spectrum(1),spectrum(2)])

def test_nonfinite_loss_rejected():
    bad=spectrum(1)
    bad["loss_db_cm"][10]=float("nan")
    with pytest.raises(ValueError,match="nonfinite"):
        run([spectrum(5),spectrum(2),bad])

def test_ambiguous_peak_rejected():
    bad=spectrum(1)
    bad["loss_db_cm"][49]=bad["loss_db_cm"][50]
    with pytest.raises(ValueError):
        run([spectrum(5),spectrum(2),bad])

def test_nonuniform_sampling_recovers_subgrid_peak():
    # Exact downward parabola, vertex deliberately between samples.
    xs0=[600,610,620,630,640,648,651,655,660,670,680,690,700]
    def row(xs):
        ys=[20-0.002*(x-649.3)**2 for x in xs]
        return spectrum(1,wavelength_nm=xs,loss_db_cm=ys)
    rows=[row(list(range(600,701,5))), row(list(range(600,701,2))),
          row(list(range(600,701,1)))]
    result=run(rows)
    assert result["diagnostic_pass"]
    assert abs(result["spectra"][-1]["peak_nm"]-649.3)<1e-8
    assert result["spectra"][-1]["sample_peak_nm"] != result["spectra"][-1]["peak_nm"]
    assert result["physics_qualified"] is False

def test_peak_vertex_rejects_unresolved_flat_curvature():
    bad=spectrum(1)
    bad["loss_db_cm"][49:52]=[10,10,10]
    with pytest.raises(ValueError):
        run([spectrum(5),spectrum(2),bad])
