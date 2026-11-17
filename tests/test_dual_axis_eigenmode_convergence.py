"""Constructed records test gate logic, never Maxwell physics."""
from research.dual_axis_eigenmode_convergence import audit


def records():
    return [
        {"mesh_um": m, "half_domain_um": d,
         "beta_per_um": [10 + 0.01 * m + 0.0001 / d, 0.2],
         "relative_matrix_residual": 1e-12,
         "material_sha256": "a" * 64, "topology_sha256": "b" * 64,
         "wavelength_um": 0.8, "analyte_ri": 1.36,
         "boundary_condition": "dirichlet"}
        for m in (0.01, 0.02, 0.04) for d in (2., 3., 4.)
    ]


def test_diagnostic_only():
    result = audit(records())
    assert result["diagnostic_pass"]
    assert not result["physics_qualified"]
    assert not result["pml_verified"]


def test_mesh_drift_not_masked_by_domain_stability():
    rows = records()
    for row in rows:
        row["beta_per_um"][0] += 100 * row["mesh_um"]
    assert "mesh drift exceeds tolerance" in audit(rows)["errors"]


def test_material_change_fails_closed():
    rows = records()
    rows[0]["material_sha256"] = "c" * 64
    assert not audit(rows)["diagnostic_pass"]


def test_duplicate_mesh_domain_rejected():
    rows = records()
    rows.append(dict(rows[0]))
    assert "duplicate mesh/domain combinations" in audit(rows)["errors"]


def test_nonfinite_beta_rejected():
    rows = records()
    rows[0]["beta_per_um"] = [float("nan"), 0.2]
    assert not audit(rows)["diagnostic_pass"]


def test_bad_residual_rejected():
    rows = records()
    rows[0]["relative_matrix_residual"] = 0.1
    assert not audit(rows)["diagnostic_pass"]


def test_domain_only_sweep_rejected():
    rows = [r for r in records() if r["half_domain_um"] == 4.]
    assert not audit(rows)["diagnostic_pass"]
