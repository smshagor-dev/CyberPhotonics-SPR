"""Fail-closed complex eigenvalue mesh convergence diagnostic (not physics qualification)."""
import math

def audit_complex_neff_convergence(records, relative_tolerance=1e-3):
    required = ("mesh_size_um", "neff_real", "neff_imag", "geometry_sha256",
                "material_sha256", "wavelength_um", "analyte_ri", "mode_id",
                "pml_config_sha256")
    failures = []
    if len(records) < 3:
        failures.append("at_least_three_mesh_levels_required")
    parsed = []
    for i, record in enumerate(records):
        if not isinstance(record, dict) or any(k not in record for k in required):
            failures.append(f"incomplete_record_{i}")
            continue
        numeric = [record[k] for k in ("mesh_size_um", "neff_real", "neff_imag",
                                     "wavelength_um", "analyte_ri")]
        if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                not math.isfinite(v) for v in numeric) or
                numeric[0] <= 0 or numeric[3] <= 0 or numeric[4] <= 0 or
                any(not isinstance(record[k], str) or not record[k] for k in
                    ("geometry_sha256", "material_sha256", "mode_id", "pml_config_sha256"))):
            failures.append(f"invalid_record_{i}")
            continue
        parsed.append(record)
    if (isinstance(relative_tolerance, bool) or
            not isinstance(relative_tolerance, (float, int)) or
            not math.isfinite(relative_tolerance) or relative_tolerance <= 0):
        failures.append("invalid_tolerance")
    if len(parsed) != len(records):
        failures.append("incomplete_grid")
    if parsed:
        for key in ("geometry_sha256", "material_sha256", "wavelength_um",
                    "analyte_ri", "mode_id", "pml_config_sha256"):
            if len({r[key] for r in parsed}) != 1:
                failures.append(f"changed_{key}")
        if any(a["mesh_size_um"] <= b["mesh_size_um"]
               for a, b in zip(parsed, parsed[1:])):
            failures.append("mesh_not_strictly_refined")
    drifts = []
    for coarse, fine in zip(parsed, parsed[1:]):
        a = complex(coarse["neff_real"], coarse["neff_imag"])
        b = complex(fine["neff_real"], fine["neff_imag"])
        drifts.append({"absolute": abs(a-b),
                       "relative": abs(a-b)/max(abs(b), 1e-15)})
    if len(drifts) >= 2 and drifts[-1]["relative"] > drifts[-2]["relative"]:
        failures.append("nondecreasing_refinement_drift")
    if drifts and drifts[-1]["relative"] > relative_tolerance:
        failures.append("tolerance_not_met")
    return {"numerical_diagnostic_pass": not failures,
            "failures": sorted(set(failures)),
            "adjacent_complex_neff_drifts": drifts,
            "physics_qualified": False, "publication_ready": False,
            "comsol_verified": False, "mode_identity_verified": False,
            "pml_convergence_verified": False}
