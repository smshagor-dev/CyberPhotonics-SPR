"""Fail-closed native FEM archive checks; no physics qualification from fixtures."""
import importlib.util
import json
import warnings
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_native_fem_archive.py"
spec = importlib.util.spec_from_file_location("native_fem_archive_audit", SCRIPT)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def test_duplicate_zip_members_fail_closed(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("same", "first")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr("same", "second")
    with pytest.raises(ValueError, match="duplicate"):
        audit_module.audit(path)


def test_unregistered_artifacts_fail_closed(tmp_path):
    path = tmp_path / "unregistered.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("GRID_READINESS.json", json.dumps({"artifacts_sha256": {}}))
        archive.writestr("unregistered.txt", "not physics")
    with pytest.raises(ValueError, match="unregistered"):
        audit_module.audit(path)


def test_path_traversal_fails_closed(tmp_path):
    path = tmp_path / "traversal.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="unsafe"):
        audit_module.audit(path)
