"""Fail-closed review gate for native PCF-SPR evidence; never certifies physics."""
import re

REQUIRED_IDENTITIES = ("solver", "solver_version", "material_source", "boundary_condition")
REQUIRED_HASHES = ("geometry_sha256", "mesh_sha256", "material_sha256", "field_archive_sha256")
REQUIRED_VERIFICATIONS = ("pml_converged", "mesh_converged", "domain_converged",
                          "spectral_converged", "mode_tracking_verified",
                          "fixed_geometry_ri_verified", "independent_solver_verified")
PLACEHOLDERS = {"unknown", "none", "n/a", "placeholder", "recorded-version",
                "measured-au-source", "tbd", "todo"}


def audit(record):
    """Identify review candidates only; flags are self-attestations, not verification."""
    result = dict(qualified_for_review=False, physics_qualified=False,
                  publication_ready=False, comsol_verified=False)
    if not isinstance(record, dict):
        return {**result, "reasons": ["record_not_mapping"]}
    reasons = []
    if record.get("source_kind") != "native_open_source_maxwell":
        reasons.append("unsupported_source_kind")
    for key in REQUIRED_IDENTITIES:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip() or value.strip().lower() in PLACEHOLDERS:
            reasons.append("missing_or_placeholder_identity:" + key)
    if record.get("boundary_condition") != "verified_pml":
        reasons.append("unverified_outgoing_boundary")
    if not isinstance(record.get("commit_sha"), str) or not re.fullmatch(r"[0-9a-f]{40}", record["commit_sha"]):
        reasons.append("invalid_commit_sha")
    for key in REQUIRED_HASHES:
        value = record.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or len(set(value)) == 1:
            reasons.append("invalid_or_placeholder_hash:" + key)
    hashes = [record.get(k) for k in REQUIRED_HASHES]
    if all(isinstance(h, str) for h in hashes) and len(set(hashes)) != len(hashes):
        reasons.append("reused_artifact_hash")
    for key in REQUIRED_VERIFICATIONS:
        if record.get(key) is not True:
            reasons.append("unverified:" + key)
    for key in ("sample_count", "failure_count"):
        value = record.get(key)
        if type(value) is not int or value < 0 or (key == "sample_count" and value == 0):
            reasons.append("invalid_count:" + key)
    if record.get("failure_count") != 0:
        reasons.append("unresolved_failures")
    if record.get("comsol_verified") is True:
        reasons.append("unsupported_comsol_claim")
    for key in ("artifact_url", "artifact_sha256", "verification_manifest_sha256"):
        value = record.get(key)
        if key == "artifact_url":
            if not isinstance(value, str) or not value.startswith("https://github.com/"):
                reasons.append("missing_native_artifact_url")
        elif not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) or len(set(value)) == 1:
            reasons.append("invalid_verification_digest:" + key)
    return {**result, "qualified_for_review": not reasons,
            "reasons": sorted(set(reasons))}
