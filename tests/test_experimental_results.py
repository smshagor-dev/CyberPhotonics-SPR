from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sprpcf.validation.experiment import analyze_experimental_measurements, validate_experiment_manifest


def _spectrum(path: Path, center_nm: float) -> Path:
    wavelength = np.linspace(600.0, 680.0, 321)
    loss = 2.0 + 8.0 * np.exp(-0.5 * ((wavelength - center_nm) / 4.0) ** 2)
    pd.DataFrame({"wavelength_nm": wavelength, "loss_db_per_cm": loss}).to_csv(path, index=False)
    return path


def test_experimental_analysis_computes_sensitivity_repeatability_and_provenance(tmp_path: Path) -> None:
    spectra = []
    for ri, center in ((1.33, 620.0), (1.35, 636.0), (1.37, 652.0)):
        for replicate, shift in ((1, -0.1), (2, 0.1)):
            raw = tmp_path / f"raw_{ri}_{replicate}.jsonl"
            raw.write_text('{"measured":true}\n', encoding="utf-8")
            calibrated = _spectrum(tmp_path / f"calibrated_{ri}_{replicate}.csv", center + shift)
            spectra.append(
                {
                    "analyte_ri": ri,
                    "replicate": str(replicate),
                    "raw_path": raw.name,
                    "path": calibrated.name,
                }
            )
    manifest = tmp_path / "measurements.yaml"
    manifest.write_text(
        yaml.safe_dump({"schema_version": 1, "experiment_id": "test", "spectra": spectra}, sort_keys=False),
        encoding="utf-8",
    )

    report = analyze_experimental_measurements(manifest, tmp_path / "analysis")

    assert report["replicate_count"] == 6
    assert report["unique_ri_count"] == 3
    assert report["sensitivity_nm_per_riu"] == pytest.approx(800.0, rel=1e-3)
    assert report["fit_r2"] > 0.999
    assert report["repeatability_max_sd_nm"] is not None
    assert all(len(row["raw_sha256"]) == 64 for row in report["source_spectra"])
    assert (tmp_path / "analysis" / "resonance_vs_ri.png").is_file()
    assert (tmp_path / "analysis" / "repeatability_by_ri.png").is_file()


def test_experimental_manifest_requires_raw_traceability() -> None:
    with pytest.raises(ValueError, match="raw_path"):
        validate_experiment_manifest(
            {
                "schema_version": 1,
                "spectra": [
                    {"analyte_ri": 1.33, "path": "a.csv"},
                    {"analyte_ri": 1.35, "path": "b.csv"},
                ],
            }
        )
