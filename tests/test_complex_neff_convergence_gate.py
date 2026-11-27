from sprpcf.simulation.complex_neff_convergence_gate import audit_complex_neff_convergence as audit


def sample():
    return [dict(mesh_size_um=h, neff_real=n, neff_imag=0.0002,
                 geometry_sha256="g", material_sha256="m", wavelength_um=.8,
                 analyte_ri=1.36, mode_id="core_candidate", pml_config_sha256="p")
            for h, n in ((.03, 1.450), (.025, 1.4501), (.02, 1.45012))]


def test_diagnostic_never_qualifies_physics():
    result = audit(sample())
    assert result["numerical_diagnostic_pass"]
    assert not result["physics_qualified"]
    assert not result["pml_convergence_verified"]


def test_changed_material_rejected():
    rows = sample()
    rows[1]["material_sha256"] = "different"
    assert "changed_material_sha256" in audit(rows)["failures"]


def test_mesh_order_rejected():
    rows = sample()
    rows.reverse()
    assert "mesh_not_strictly_refined" in audit(rows)["failures"]


def test_divergent_refinement_rejected():
    rows = sample()
    rows[2]["neff_real"] = 1.47
    assert "nondecreasing_refinement_drift" in audit(rows)["failures"]


def test_nonfinite_rejected():
    rows = sample()
    rows[1]["neff_imag"] = float("nan")
    assert not audit(rows)["numerical_diagnostic_pass"]


def test_short_study_rejected():
    assert "at_least_three_mesh_levels_required" in audit(sample()[:2])["failures"]
