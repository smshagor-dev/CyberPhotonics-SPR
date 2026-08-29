from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sprpcf.evidence.qualification import qualify_experimental_sensor, write_evidence_record


def test_registry_artifacts_enter_reviewer_package(tmp_path: Path) -> None:
    raw = tmp_path / "capture.csv"
    raw.write_text("x,y\n1,2\n", encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# Test fixture\n", encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text('{"fixture": true}\n', encoding="utf-8")
    registry = tmp_path / "evidence_registry.json"

    record = qualify_experimental_sensor(
        [raw],
        protocol_path=protocol,
        calibration_path=calibration,
        instrument_id="TEST-INSTRUMENT",
        acquired_at="2026-08-29T10:30:00+03:00",
        registry_path=registry,
        label="Test fixture only",
    )
    write_evidence_record(registry, record)

    out = tmp_path / "reviewer"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_reviewer_package.py",
            "--out",
            str(out),
            "--evidence-registry",
            str(registry),
            "--no-release-metadata",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "experimental_sensor" in manifest["evidence_classes"]
    assert "reproducibility" in manifest["evidence_classes"]
    assert (out / "evidence" / "qualified_evidence_registry" / "evidence_registry.json").is_file()
