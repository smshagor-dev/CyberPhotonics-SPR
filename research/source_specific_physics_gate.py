"""Fail-closed, source-specific numerical-physics evidence manifest audit.

This checks metadata structure only; SHA-256 values are NOT recomputed from
artifacts. No manifest can independently qualify physical or COMSOL evidence.
"""
import hashlib
import json
import re
from pathlib import Path

REQUIRED = {
    "frozen_geometry": "literature_geometry",
    "measured_gold": "measured_au",
    "full_vector_modes": "native_maxwell",
    "mesh_convergence": "native_maxwell",
    "pml_convergence": "native_maxwell",
    "domain_convergence": "native_maxwell",
    "spectral_convergence": "native_maxwell",
    "core_spp_tracking": "native_maxwell",
    "fixed_geometry_ri_spectra": "native_maxwell",
    "independent_solver": "independent_maxwell",
}
SHA = re.compile(r"^[0-9a-f]{64}$")

def evaluate(registry):
    if not isinstance(registry, dict) or not isinstance(registry.get("sources"), list):
        raise ValueError("registry.sources must be an array")
    valid, seen, errors = {}, set(), []
    for i, src in enumerate(registry["sources"]):
        if not isinstance(src, dict):
            errors.append(f"source[{i}]: not an object")
            continue
        sid = src.get("id")
        if not isinstance(sid, str) or not sid or sid in seen:
            errors.append(f"source[{i}]: missing or duplicate id")
            continue
        seen.add(sid)
        digest = src.get("sha256")
        if not isinstance(digest, str) or not SHA.fullmatch(digest):
            errors.append(f"{sid}: invalid sha256")
            continue
        if src.get("kind") in ("fixture", "synthetic", "surrogate", "mock", "comsol_unverified"):
            errors.append(f"{sid}: ineligible source kind")
            continue
        if src.get("verified") is not True or src.get("failed_runs", 0) != 0:
            errors.append(f"{sid}: unverified or failed runs")
            continue
        if not isinstance(src.get("uri"), str) or not src["uri"].strip():
            errors.append(f"{sid}: missing artifact locator")
            continue
        valid[sid] = src
    gates = {}
    declared = registry.get("gates", {})
    if not isinstance(declared, dict):
        declared = {}
    for gate, kind in REQUIRED.items():
        refs = declared.get(gate, [])
        gates[gate] = bool(isinstance(refs, list) and refs) and all(
            isinstance(ref, str) and ref in valid and valid[ref].get("kind") == kind
            for ref in refs
        )
    return {
        "schema": "pcf_spr_source_gate_v1",
        "gates": gates,
        "errors": errors,
        "source_manifest_complete": all(gates.values()) and not errors,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }

def write_report(registry_path, output_path):
    raw = Path(registry_path).read_bytes()
    result = evaluate(json.loads(raw))
    result["registry_sha256"] = hashlib.sha256(raw).hexdigest()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("registry")
    p.add_argument("output")
    args = p.parse_args()
    print(json.dumps(write_report(args.registry, args.output), sort_keys=True))
