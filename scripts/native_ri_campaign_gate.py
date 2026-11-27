"""Fail-closed audit for fixed-geometry native PCF-SPR RI spectra.

This checks evidence *eligibility*, not Maxwell accuracy or publication readiness.
No fixture or surrogate spectrum can pass.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


class EvidenceError(ValueError):
    """A native spectrum cannot be used as qualified evidence."""


def audit_ri_campaign(manifest: dict, artifact_root: str | Path) -> dict:
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise EvidenceError("artifact root must be a directory")
    if manifest.get("source_kind") != "native_open_source_full_vector_fem":
        raise EvidenceError("only native full-vector FEM is eligible")
    if manifest.get("solver") in (None, "", "COMSOL"):
        raise EvidenceError("missing or prohibited solver identity")
    if manifest.get("comsol_verified") is not False:
        raise EvidenceError("COMSOL verification must remain false")
    if manifest.get("publication_ready") is not False:
        raise EvidenceError("publication flag must remain false")
    for key in ("geometry_sha256", "mesh_sha256", "material_source_sha256",
                "solver_config_sha256", "commit_sha"):
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EvidenceError(f"missing {key}")
    rows = manifest.get("spectra")
    if not isinstance(rows, list) or len(rows) < 3:
        raise EvidenceError("at least three RI spectra required")
    seen_ri, seen_paths = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise EvidenceError("invalid spectrum record")
        ri = row.get("analyte_ri")
        if isinstance(ri, bool) or not isinstance(ri, (float, int)) or not math.isfinite(ri):
            raise EvidenceError("invalid analyte RI")
        if ri in seen_ri:
            raise EvidenceError("duplicate analyte RI")
        seen_ri.add(ri)
        for key in ("geometry_sha256", "mesh_sha256", "material_source_sha256",
                    "solver_config_sha256"):
            if row.get(key) != manifest[key]:
                raise EvidenceError(f"RI-dependent {key} violates fixed campaign")
        if row.get("source_kind") != manifest["source_kind"]:
            raise EvidenceError("mixed provenance")
        path = row.get("path")
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise EvidenceError("unsafe path")
        parts = Path(path).parts
        if ".." in parts or "." in parts or path in seen_paths:
            raise EvidenceError("path traversal or duplicate artifact")
        seen_paths.add(path)
        candidate = root
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise EvidenceError("symlink artifact component")
        target = candidate.resolve(strict=True)
        if not target.is_file() or not target.is_relative_to(root):
            raise EvidenceError("artifact outside root")
        raw = target.read_bytes()
        if len(raw) != row.get("bytes") or hashlib.sha256(raw).hexdigest() != row.get("sha256"):
            raise EvidenceError("native artifact bytes differ from manifest")
        spectrum = json.loads(raw)
        wavelengths = spectrum.get("wavelength_nm")
        loss = spectrum.get("loss_db_per_cm")
        if not isinstance(wavelengths, list) or not isinstance(loss, list) or len(wavelengths) != len(loss) or len(loss) < 7:
            raise EvidenceError("insufficient wavelength samples")
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in wavelengths + loss):
            raise EvidenceError("nonfinite spectrum")
        if any(b <= a for a, b in zip(wavelengths, wavelengths[1:])) or any(x < 0 for x in loss):
            raise EvidenceError("invalid wavelength/loss values")
        if spectrum.get("analyte_ri") != ri:
            raise EvidenceError("artifact RI mismatch")
    return {
        "eligible_for_further_physics_review": True,
        "native_artifact_count": len(rows),
        "ri_values": sorted(seen_ri),
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "limitations": [
            "Does not establish PML/mesh/domain convergence",
            "Does not establish mode identity, resonance or FWHM",
            "Does not independently verify solver or material source",
        ],
    }
