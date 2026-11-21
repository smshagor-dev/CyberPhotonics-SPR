"""Software-only adversarial tests; optional genuine archive never replaced by a fixture."""
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import pytest

from scripts.native_spectral_gate import audit


def test_missing_readiness_flag_fails_closed():
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "bad.zip"
        protocol = {"analyte_ris": [1.35], "wavelengths_um": [0.75, 0.8, 0.85], "num_modes": 1}
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("experiment_protocol.json", json.dumps(protocol))
            z.writestr("GRID_READINESS.json", json.dumps({"artifacts_sha256": {
                "experiment_protocol.json": hashlib.sha256(json.dumps(protocol).encode()).hexdigest()}}))
        with pytest.raises(ValueError):
            audit(archive)


def test_duplicate_members_rejected():
    with tempfile.TemporaryDirectory() as d:
        archive = Path(d) / "bad.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("GRID_READINESS.json", "{}")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
                z.writestr("GRID_READINESS.json", "{}")
        with pytest.raises(ValueError, match="unsafe"):
            audit(archive)


def test_genuine_archive_when_explicitly_provided():
    path = os.environ.get("NATIVE_FEM_ARCHIVE")
    if not path:
        pytest.skip("No genuine solver archive provided; never substitute synthetic data")
    result = audit(path)
    assert result["eigenpairs"] == 18
    assert len(result["spectra"]) == 6
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False
