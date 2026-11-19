"""Software-only native archive gate tests; no Maxwell evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

from test_vector_plaquette_audit import edges


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_native_vector_plaquettes.py"
    spec = importlib.util.spec_from_file_location("native_plaquette_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    protocol = {"analyte_ris": [1.35, 1.36], "wavelengths_um": [0.75, 0.80], "num_modes": 2}
    files = {"experiment_protocol.json": protocol, "vector_mode_overlap_diagnostics.json": edges()}
    hashes = {}
    for name, data in files.items():
        path = root / name
        path.write_text(json.dumps(data))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "GRID_READINESS.json").write_text(json.dumps({"artifacts_sha256": hashes}))


def test_read_only_native_graph_passes_without_physics_promotion(tmp_path):
    fixture(tmp_path)
    result = load_runner().audit_native_grid(tmp_path)
    assert result["graph_consistent"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False


def test_tampered_native_overlap_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "vector_mode_overlap_diagnostics.json").write_text("[]")
    result = load_runner().audit_native_grid(tmp_path)
    assert not result["graph_consistent"]
    assert any("digest mismatch" in e for e in result["errors"])


def test_native_graph_ambiguity_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "vector_mode_overlap_diagnostics.json"
    data = json.loads(path.read_text())
    data[0]["bidirectional_min_overlap_margin"] = -0.01
    path.write_text(json.dumps(data))
    readiness = tmp_path / "GRID_READINESS.json"
    report = json.loads(readiness.read_text())
    report["artifacts_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    readiness.write_text(json.dumps(report))
    result = load_runner().audit_native_grid(tmp_path)
    assert not result["graph_consistent"]
    assert result["ambiguous_plaquettes"]


def test_missing_native_protocol_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "experiment_protocol.json").unlink()
    assert not load_runner().audit_native_grid(tmp_path)["graph_consistent"]
