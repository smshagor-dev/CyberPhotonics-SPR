"""Software-only regression checks for native archive audit (not physics evidence)."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_native_vector_grid import audit_grid


def make_archive(root):
    protocol = {"wavelengths_um": [0.75, 0.8], "analyte_ris": [1.35, 1.36],
                "num_modes": 1}
    files = {"experiment_protocol.json": json.dumps(protocol),
             "pcf_frozen_exploratory_grid_mesh.msh": "mesh fixture",
             "topology.json": "{}",
             "vector_mode_overlap_diagnostics.json": "[]"}
    for i, ri in enumerate(protocol["analyte_ris"]):
        for j, wl in enumerate(protocol["wavelengths_um"]):
            files[f"raw_vector_eh_ri{i}_wl{j}.npz"] = "NOT A PHYSICAL FIELD"
    header = ("ri_index,wavelength_index,mode_index_native_untracked,"
              "analyte_ri,wavelength_um,n_eff_real,n_eff_imag,"
              "gold_epsilon_real,gold_epsilon_imag,silica_ri\n")
    rows = "".join(f"{i},{j},0,{ri},{wl},1.4,0.01,-20,1.5,1.45\n"
                   for i, ri in enumerate(protocol["analyte_ris"])
                   for j, wl in enumerate(protocol["wavelengths_um"]))
    files["raw_eigenmode_grid.csv"] = header + rows
    hashes = {}
    for name, value in files.items():
        data = value.encode()
        (root / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    report = {"evidence_class": "unqualified_solver_feasibility",
              "scientific_submission_ready": False, "completed": True,
              "failures": [], "solver_calls_completed": 4,
              "raw_eigenmode_rows": 4, "artifacts_sha256": hashes}
    (root / "GRID_READINESS.json").write_text(json.dumps(report))
    return report


def test_complete_fixture_only_passes_archive_integrity(tmp_path):
    make_archive(tmp_path)
    result = audit_grid(tmp_path)
    assert result["passed"] is True
    assert result["publication_qualified"] is False


def test_tampered_field_fails(tmp_path):
    make_archive(tmp_path)
    (tmp_path / "raw_vector_eh_ri0_wl0.npz").write_text("tampered")
    assert "SHA256 mismatch" in str(audit_grid(tmp_path)["errors"])


def test_missing_mode_fails_even_when_hashes_updated(tmp_path):
    report = make_archive(tmp_path)
    p = tmp_path / "raw_eigenmode_grid.csv"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    report["artifacts_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    assert "incomplete eigenmode grid" in str(audit_grid(tmp_path)["errors"])


def test_publication_flag_true_fails(tmp_path):
    report = make_archive(tmp_path)
    report["scientific_submission_ready"] = True
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    assert audit_grid(tmp_path)["passed"] is False


def test_missing_manifest_fails(tmp_path):
    assert audit_grid(tmp_path)["passed"] is False


def test_duplicate_mode_rejected_with_updated_manifest(tmp_path):
    report = make_archive(tmp_path)
    p = tmp_path / "raw_eigenmode_grid.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines + [lines[1]]) + "\n")
    report["artifacts_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    result = audit_grid(tmp_path)
    assert result["passed"] is False
    assert "duplicate or unexpected" in str(result["errors"])
    assert result["publication_qualified"] is False


def test_nonfinite_eigenvalue_rejected_with_updated_manifest(tmp_path):
    report = make_archive(tmp_path)
    p = tmp_path / "raw_eigenmode_grid.csv"
    lines = p.read_text().splitlines()
    lines[1] = lines[1].replace(",1.4,0.01,", ",nan,0.01,")
    p.write_text("\n".join(lines) + "\n")
    report["artifacts_sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    result = audit_grid(tmp_path)
    assert result["passed"] is False
    assert "nonfinite n_eff_real" in str(result["errors"])


def test_missing_native_field_rejected_even_with_updated_manifest(tmp_path):
    report = make_archive(tmp_path)
    name = "raw_vector_eh_ri0_wl0.npz"
    (tmp_path / name).unlink()
    report["artifacts_sha256"].pop(name)
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    result = audit_grid(tmp_path)
    assert result["passed"] is False
    assert "untracked native E/H field" in str(result["errors"])
