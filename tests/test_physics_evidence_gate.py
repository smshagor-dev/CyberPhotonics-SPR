from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_combined_physics_gate_fails_closed_for_missing_real_archives(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_physics_evidence.py"
    output = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--iteration", str(tmp_path / "missing_comsol"),
            "--refinement-archive", str(tmp_path / "missing_refinement"),
            "--out", str(output), "--strict",
        ],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["archived_records_consistent"] is False
    assert report["submission_physics_claim_ready"] is False
    assert report["independent_physics_correctness_verified"] is False
    assert report["spectrum_audit"]["consistency_ok"] is False
    assert report["mesh_and_spectral_audit"]["convergence_consistent"] is False
