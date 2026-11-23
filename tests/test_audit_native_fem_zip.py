"""Adversarial structural checks; fixtures are never physics evidence."""
import io
import json
import zipfile

from scripts.audit_native_fem_zip import audit_zip

FLAGS = ("scientific_submission_ready", "pml_validated", "mode_tracking_physically_verified", "resonance_fwhm_verified", "physical_sensor_sensitivity_verified")

def make_zip(tmp_path, promoted=False, missing=False):
    path = tmp_path / "pilot.zip"
    status = {"completed": True, "solver_calls_completed": 2, "raw_eigenmode_rows": 4,
              **{flag: False for flag in FLAGS}}
    if promoted:
        status["scientific_submission_ready"] = True
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as npz:
        npz.writestr("fields.npy", b"fixture-only")
    with zipfile.ZipFile(path, "w") as outer:
        outer.writestr("GRID_READINESS.json", json.dumps(status))
        outer.writestr("raw_eigenmode_grid.csv", "mode\\n0\\n1\\n0\\n1\\n".replace("\\n", "\n"))
        outer.writestr("raw_vector_eh_ri0_wl0.npz", inner.getvalue())
        if not missing:
            outer.writestr("raw_vector_eh_ri0_wl1.npz", inner.getvalue())
    return path

def test_valid_structure(tmp_path):
    assert audit_zip(make_zip(tmp_path), 1, 2, 2)["passed"]

def test_missing_eh_archive(tmp_path):
    assert not audit_zip(make_zip(tmp_path, missing=True), 1, 2, 2)["passed"]

def test_reject_publication_promotion(tmp_path):
    assert not audit_zip(make_zip(tmp_path, promoted=True), 1, 2, 2)["passed"]

def test_reject_incorrect_grid(tmp_path):
    assert not audit_zip(make_zip(tmp_path), 2, 2, 2)["passed"]
