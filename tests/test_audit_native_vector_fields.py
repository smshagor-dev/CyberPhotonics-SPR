"""Adversarial software-only fixtures; not numerical physics evidence."""
import hashlib
import io
import json
import zipfile
import warnings

import numpy as np
import pytest

from scripts.audit_native_vector_fields import audit_vector_fields


def fixture(path, fields=None, duplicate=False):
    if fields is None:
        fields = {"mode_0_E": np.array([1+1j]), "mode_0_H": np.array([2+1j])}
    stream = io.BytesIO()
    np.savez(stream, **fields)
    raw = stream.getvalue()
    protocol = json.dumps({"analyte_ris": [1.35], "wavelengths_um": [0.8], "num_modes": 1}).encode()
    hashes = {
        "experiment_protocol.json": hashlib.sha256(protocol).hexdigest(),
        "raw_vector_eh_ri0_wl0.npz": hashlib.sha256(raw).hexdigest(),
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("experiment_protocol.json", protocol)
        archive.writestr("GRID_READINESS.json", json.dumps({"artifacts_sha256": hashes}))
        archive.writestr("raw_vector_eh_ri0_wl0.npz", raw)
        if duplicate:
            # Deliberately construct a malformed archive; only the expected
            # writer warning is suppressed under pytest -W error.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Duplicate name: .*", category=UserWarning, module=r"zipfile")
                archive.writestr("raw_vector_eh_ri0_wl0.npz", raw)


def test_valid_fixture(tmp_path):
    path = tmp_path / "fields.zip"
    fixture(path)
    result = audit_vector_fields(path)
    assert result["native_field_arrays_verified"] == 2
    assert result["publication_ready"] is False


@pytest.mark.parametrize("fields", [
    {"mode_0_E": np.array([1+1j])},
    {"mode_0_E": np.array([complex("nan")]), "mode_0_H": np.array([1+0j])},
    {"mode_0_E": np.array([0j]), "mode_0_H": np.array([1+0j])},
    {"mode_0_E": np.array([1.0]), "mode_0_H": np.array([1+0j])},
])
def test_reject_bad_fields(tmp_path, fields):
    path = tmp_path / "fields.zip"
    fixture(path, fields)
    with pytest.raises(ValueError):
        audit_vector_fields(path)


def test_duplicate_member(tmp_path):
    path = tmp_path / "fields.zip"
    fixture(path, duplicate=True)
    with pytest.raises(ValueError):
        audit_vector_fields(path)
