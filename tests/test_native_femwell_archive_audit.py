"""Software-only adversarial tests; fixtures are never physical evidence."""
import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from scripts.audit_native_femwell_archive import audit


def make_archive(path, *, corrupt_loss=False, missing_field=False):
    readiness = {
        "completed": True, "analyte_ris": [1.36], "wavelengths_um": [0.8],
        "num_modes_requested": 1, "solver_calls_completed": 1,
        "raw_eigenmode_rows": 1, "vector_overlap_matched_edges": 0,
        **{key: False for key in (
            "scientific_submission_ready", "mode_tracking_physically_verified",
            "mesh_convergence_verified", "pml_validated",
            "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
            "Au_reference_transcription_independently_verified")},
    }
    loss = (20 / np.log(10)) * (2 * np.pi / 0.8) * 1e-7 * 1e4
    if corrupt_loss:
        loss += 0.01
    record = {
        "ri_index": 0, "wavelength_index": 0, "mode_index_native_untracked": 0,
        "wavelength_um": 0.8, "n_eff_real": 1.45, "n_eff_imag": 1e-7,
        "raw_attenuation_db_per_cm_unqualified": loss,
    }
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(record))
    writer.writeheader()
    writer.writerow(record)
    vectors = io.BytesIO()
    arrays = {"mode_0_E": np.ones(3, dtype=complex)}
    if not missing_field:
        arrays["mode_0_H"] = np.ones(3, dtype=complex)
    np.savez(vectors, **arrays)
    members = {
        "experiment_protocol.json": b"{}",
        "topology.json": b"{}",
        "raw_eigenmode_grid.csv": output.getvalue().encode(),
        "raw_vector_eh_ri0_wl0.npz": vectors.getvalue(),
        "vector_mode_overlap_diagnostics.json": b"[]",
    }
    readiness["artifacts_sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in members.items()
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("GRID_READINESS.json", json.dumps(readiness))


def test_fixture_integrity_does_not_qualify_physics(tmp_path):
    path = tmp_path / "fixture.zip"
    make_archive(path)
    result = audit(path)
    assert result["native_integrity_verified"]
    assert not result["physics_qualified"]
    assert not result["publication_ready"]
    assert not result["comsol_verified"]


@pytest.mark.parametrize("mutation", ["corrupt_loss", "missing_field"])
def test_fail_closed_on_corrupted_native_data(tmp_path, mutation):
    path = tmp_path / "fixture.zip"
    make_archive(path, **{mutation: True})
    with pytest.raises(ValueError):
        audit(path)
