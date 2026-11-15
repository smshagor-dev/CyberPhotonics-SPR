"""Software-only negative tests for ZIP integrity auditor; never physics evidence."""
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "audit_native_grid_zip.py"
spec = importlib.util.spec_from_file_location("native_zip_audit", MODULE)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def make_zip(path, *, duplicate=False, nonfinite=False, publication=False, missing=False):
    import csv
    import hashlib
    import math
    protocol = {"analyte_ris": [1.35], "wavelengths_um": [0.8], "num_modes": 1}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["ri_index", "wavelength_index",
        "mode_index_native_untracked", "analyte_ri", "wavelength_um",
        "n_eff_real", "n_eff_imag", "gold_epsilon_real", "gold_epsilon_imag"])
    writer.writeheader()
    row = dict(ri_index=0, wavelength_index=0, mode_index_native_untracked=0,
               analyte_ri=1.35, wavelength_um=0.8, n_eff_real=float("nan") if nonfinite else 1.4,
               n_eff_imag=0.01, gold_epsilon_real=-20, gold_epsilon_imag=1.5)
    writer.writerow(row)
    if duplicate:
        writer.writerow(row)
    fields = io.BytesIO()
    np.savez(fields, electric=np.array([1 + 2j]), magnetic=np.array([3 + 4j]))
    files = {"experiment_protocol.json": json.dumps(protocol).encode(),
             "raw_eigenmode_grid.csv": buffer.getvalue().encode()}
    if not missing:
        files["raw_vector_eh_ri0_wl0.npz"] = fields.getvalue()
    report = {"artifacts_sha256": {name: hashlib.sha256(data).hexdigest()
                                    for name, data in files.items()},
              "solver_calls_completed": 1,
              "mesh_convergence_verified": False,
              "mode_tracking_physically_verified": False,
              "physical_sensor_sensitivity_verified": False,
              "pml_validated": False,
              "resonance_fwhm_verified": False,
              "scientific_submission_ready": publication,
              "sensor_geometry_peer_reviewed_and_frozen": False}
    files["GRID_READINESS.json"] = json.dumps(report).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


@pytest.mark.parametrize("variant", ["valid", "duplicate", "nonfinite", "publication", "missing"])
def test_zip_audit_fail_closed(tmp_path, variant):
    path = tmp_path / "archive.zip"
    make_zip(path, **({} if variant == "valid" else {variant: True}))
    result = audit_module.audit(path)
    assert result["pass"] is (variant == "valid")
    assert result["evidence_class"] == "archive_integrity_only_not_physics_validation"
