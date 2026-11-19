"""Fail-closed integrity audit for exploratory solver-native FEM archives.

Passing means archive consistency ONLY, never numerical physics qualification.
"""
import csv
import hashlib
import json
from pathlib import Path

FALSE_GATES = (
    "sensor_geometry_peer_reviewed_and_frozen",
    "pml_validated",
    "mesh_convergence_verified", "mode_tracking_physically_verified",
    "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
    "scientific_submission_ready",
)

def audit_grid(root):
    root = Path(root)
    errors = []
    try:
        report = json.loads((root / "GRID_READINESS.json").read_text())
    except (OSError, ValueError) as exc:
        return {"integrity_passed": False, "physics_qualified": False, "errors": [str(exc)]}
    if report.get("evidence_class") != "unqualified_solver_feasibility":
        errors.append("unexpected evidence class")
    for gate in FALSE_GATES:
        if report.get(gate) is not False:
            errors.append("missing/unsafe false gate: " + gate)
    if report.get("completed") is not True or report.get("failures"):
        errors.append("incomplete solver run")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict):
        hashes = {}
        errors.append("missing hashes")
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or name == "GRID_READINESS.json":
            errors.append("unsafe artifact path")
            continue
        try:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                errors.append("checksum mismatch: " + name)
        except OSError:
            errors.append("missing artifact: " + name)
    required = {"topology.json", "experiment_protocol.json", "raw_eigenmode_grid.csv",
                "vector_mode_overlap_diagnostics.json", "pcf_frozen_exploratory_grid_mesh.msh"}
    if not required.issubset(hashes):
        errors.append("unhashed required artifact")
    try:
        protocol = json.loads((root / "experiment_protocol.json").read_text())
        au_source = protocol.get("Au")
        if au_source == "Johnson_Christy_1972_measured_bulk_reference":
            from sprpcf.simulation.materials_reference import au_reference_manifest
            expected_reference_selected = True
            if protocol.get("Au_source_manifest") != au_reference_manifest():
                errors.append("measured Au manifest does not match pinned source")
            if report.get("Au_material_reference_only_not_sensor_film_measurement") is not True:
                errors.append("measured bulk Au must not be labeled sensor-film measurement")
        elif au_source == "repository_Rakic_style_fitted_model_not_measured_constants":
            expected_reference_selected = False
            if protocol.get("Au_source_manifest") is not None:
                errors.append("fitted gold must not carry measured Au reference")
        else:
            errors.append("unknown or missing Au material source")
            expected_reference_selected = False
        if report.get("Au_reference_dataset_selected") is not expected_reference_selected:
            errors.append("Au reference-selection flag mismatch")
        if report.get("Au_material_measured_data_provenance_verified") is not False:
            errors.append("unverified Au source must not claim verified provenance")
        if report.get("Au_reference_transcription_independently_verified") is not False:
            errors.append("independent Au transcription verification must remain false")
        ris, wavelengths, modes = protocol["analyte_ris"], protocol["wavelengths_um"], protocol["num_modes"]
        expected = {(i, j, k) for i in range(len(ris)) for j in range(len(wavelengths)) for k in range(modes)}
        with (root / "raw_eigenmode_grid.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        actual = {(int(row["ri_index"]), int(row["wavelength_index"]),
                   int(row["mode_index_native_untracked"])) for row in rows}
        if len(rows) != len(expected) or actual != expected:
            errors.append("incomplete/duplicated mode grid")
        if report.get("solver_calls_completed") != len(ris) * len(wavelengths):
            errors.append("solver call count mismatch")
        if report.get("raw_eigenmode_rows") != len(expected):
            errors.append("row count mismatch")
        for i in range(len(ris)):
            for j in range(len(wavelengths)):
                if f"raw_vector_eh_ri{i}_wl{j}.npz" not in hashes:
                    errors.append(f"unhashed vector field: {i}/{j}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append("unreadable grid structure: " + str(exc))
    return {"integrity_passed": not errors, "physics_qualified": False,
            "errors": errors, "artifact_count": len(hashes)}
