"""Conservative review gate for native PCF-SPR Maxwell evidence; never grants publication status."""
import re

REQUIRED_IDENTITIES = ("solver", "solver_version", "material_source", "boundary_condition")
REQUIRED_HASHES = ("geometry_sha256", "mesh_sha256", "material_sha256", "field_archive_sha256")
REQUIRED_VERIFICATIONS = ("pml_converged", "mesh_converged", "domain_converged",
                          "spectral_converged", "mode_tracking_verified",
                          "fixed_geometry_ri_verified", "independent_solver_verified")

def audit(record):
    """Only identify a review candidate, never certify physical or COMSOL evidence."""
    if not isinstance(record, dict):
        return dict(qualified_for_review=False, physics_qualified=False,
                    publication_ready=False, comsol_verified=False,
                    reasons=["record_not_mapping"])
    reasons = []
    if record.get("source_kind") != "native_open_source_maxwell":
        reasons.append("unsupported_source_kind")
    for key in REQUIRED_IDENTITIES:
        if not isinstance(record.get(key), str) or not record[key].strip():
            reasons.append("missing_identity:" + key)
    if record.get("boundary_condition") != "verified_pml":
        reasons.append("unverified_outgoing_boundary")
    if not isinstance(record.get("commit_sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", record["commit_sha"]):
        reasons.append("invalid_commit_sha")
    for key in REQUIRED_HASHES:
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
            reasons.append("invalid_hash:" + key)
    for key in REQUIRED_VERIFICATIONS:
        if record.get(key) is not True:
            reasons.append("unverified:" + key)
    for key in ("sample_count", "failure_count"):
        v = record.get(key)
        if type(v) is not int or v < 0 or (key == "sample_count" and v == 0):
            reasons.append("invalid_count:" + key)
    if record.get("failure_count") != 0:
        reasons.append("unresolved_failures")
    if record.get("comsol_verified") is True:
        reasons.append("unsupported_comsol_claim")
    return dict(qualified_for_review=not reasons, physics_qualified=False,
                publication_ready=False, comsol_verified=False,
                reasons=sorted(set(reasons)))
