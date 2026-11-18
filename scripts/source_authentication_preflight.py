"""Fail-closed scientific provenance preflight. Self-attested metadata never qualifies physics."""
import hashlib
import json
import math

def audit(manifest):
    failures = []
    required = ("geometry_source", "geometry_sha256", "au_source", "au_data_sha256",
                "solver", "solver_version", "mesh_sha256", "pml_spec",
                "domain_convergence", "mesh_convergence", "pml_convergence",
                "mode_overlap_verified", "independent_solver_comparison", "spectra")
    for key in required:
        if key not in manifest:
            failures.append("missing:" + key)
    for key in ("geometry_sha256", "au_data_sha256", "mesh_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            failures.append("invalid_hash:" + key)
    for key in ("domain_convergence", "mesh_convergence", "pml_convergence",
                "mode_overlap_verified", "independent_solver_comparison"):
        if manifest.get(key) is not True:
            failures.append("unverified:" + key)
    if str(manifest.get("solver", "")).lower() == "comsol":
        failures.append("comsol_provenance_not_authenticated")
    spectra = manifest.get("spectra", [])
    if not isinstance(spectra, list) or len(spectra) < 3:
        failures.append("insufficient_ri_spectra")
        spectra = []
    seen_ri = set()
    for i, spectrum in enumerate(spectra):
        if not isinstance(spectrum, dict):
            failures.append(f"invalid_spectrum:{i}")
            continue
        if spectrum.get("geometry_sha256") != manifest.get("geometry_sha256"):
            failures.append(f"geometry_changed:{i}")
        if spectrum.get("mesh_sha256") != manifest.get("mesh_sha256"):
            failures.append(f"mesh_changed:{i}")
        ri = spectrum.get("analyte_ri")
        if not isinstance(ri, (int, float)) or not math.isfinite(ri) or ri in seen_ri:
            failures.append(f"invalid_ri:{i}")
        else:
            seen_ri.add(ri)
        samples = spectrum.get("samples", [])
        if not isinstance(samples, list) or len(samples) < 7:
            failures.append(f"underresolved_spectrum:{i}")
            continue
        last = -math.inf
        for j, point in enumerate(samples):
            if not isinstance(point, dict):
                failures.append(f"invalid_point:{i}:{j}")
                continue
            w, loss = point.get("wavelength_um"), point.get("loss_db_cm")
            if (not isinstance(w, (int, float)) or not math.isfinite(w) or w <= last
                or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0):
                failures.append(f"invalid_point:{i}:{j}")
            else:
                last = w
    failures.append("independent_external_authentication_required")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(physics_qualified=False, publication_ready=False, comsol_verified=False,
                reasons=sorted(set(failures)), manifest_sha256=digest)
