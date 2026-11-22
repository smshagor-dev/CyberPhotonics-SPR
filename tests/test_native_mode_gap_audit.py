"""Regression tests for native mode gap evidence gate (fixtures are not physics)."""
import importlib.util
from pathlib import Path
import warnings
import zipfile

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "native_mode_gap_audit.py"
spec = importlib.util.spec_from_file_location("native_mode_gap_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_duplicate_zip_members_fail_closed(tmp_path):
    p = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("duplicate", "a")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
            z.writestr("duplicate", "b")
    with pytest.raises(ValueError, match="unsafe ZIP"):
        module.audit_native_mode_gaps(p)


def test_missing_registry_fails(tmp_path):
    p = tmp_path / "missing.zip"
    with zipfile.ZipFile(p, "w"):
        pass
    with pytest.raises(KeyError):
        module.audit_native_mode_gaps(p)


def test_unregistered_input_fails(tmp_path):
    import json
    p = tmp_path / "unregistered.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("GRID_READINESS.json", json.dumps({"artifacts_sha256": {}}))
    with pytest.raises(ValueError, match="unregistered physics input"):
        module.audit_native_mode_gaps(p)
