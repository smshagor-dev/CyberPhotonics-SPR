"""Fail-closed material-step provenance for same-mesh E/H mode comparisons.

Diagnostic only: does not establish core/SPP identity or qualify PCF-SPR physics.
"""
import math


def authorize_material_step(*, mesh_before, mesh_after, geometry_before,
                            geometry_after, material_before, material_after,
                            kind=None, before=None, after=None):
    """Return explicit diagnostic authorization, never scientific qualification."""
    failures = []
    if not mesh_before or mesh_before != mesh_after:
        failures.append("mesh mismatch")
    if not geometry_before or geometry_before != geometry_after:
        failures.append("geometry mismatch")
    if not material_before or not material_after:
        failures.append("missing material provenance")
    changed = bool(material_before and material_after and material_before != material_after)
    if changed:
        if kind not in ("wavelength_um", "analyte_ri"):
            failures.append("unidentified material step")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
               not math.isfinite(x) or x <= 0 for x in (before, after)):
            failures.append("invalid physical step")
        elif before == after:
            failures.append("unchanged parameter with changed material hash")
        elif kind == "analyte_ri" and (before < 1 or after < 1):
            failures.append("invalid analyte refractive index")
    elif kind is not None or before is not None or after is not None:
        failures.append("declared step without changed material hash")
    return {
        "diagnostic_authorized": not failures,
        "material_changed": changed,
        "errors": failures,
        "mode_identity_physically_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
    }
