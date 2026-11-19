"""Read-only graph diagnostic for an existing immutable Femwell native grid.

This audit does NOT verify core/SPP identity or qualify sensor physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.simulation.vector_plaquette_audit import audit_mode_plaquettes


def audit_native_grid(root: Path) -> dict:
    root = Path(root)
    errors = []
    try:
        readiness_path = root / "GRID_READINESS.json"
        edges_path = root / "vector_mode_overlap_diagnostics.json"
        protocol_path = root / "experiment_protocol.json"
        for path in (readiness_path, edges_path, protocol_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing/nonregular native input: {path.name}")
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        hashes = readiness["artifacts_sha256"]
        for path in (edges_path, protocol_path):
            expected = hashes[path.name]
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError(f"invalid native digest: {path.name}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"native digest mismatch: {path.name}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        edges = json.loads(edges_path.read_text(encoding="utf-8"))
        for name in ("analyte_ris", "wavelengths_um"):
            values = protocol[name]
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"invalid spectral grid: {name}")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError(f"non-finite/non-numeric spectral grid: {name}")
            if any(a >= b for a, b in zip(values, values[1:])):
                raise ValueError(f"non-increasing spectral grid: {name}")
        if any(x <= 0 for x in protocol["wavelengths_um"]):
            raise ValueError("nonpositive wavelength in native protocol")
        if any(x <= 0 for x in protocol["analyte_ris"]):
            raise ValueError("nonpositive analyte refractive index")
        if not isinstance(edges, list):
            raise ValueError("native overlap diagnostic is not a list")
        result = audit_mode_plaquettes(
            edges, n_ri=len(protocol["analyte_ris"]),
            n_wavelength=len(protocol["wavelengths_um"]),
            n_modes=protocol["num_modes"],
        )
        errors.extend(result["errors"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"native input validation failed: {exc}")
        result = {}
    return {
        "diagnostic_type": "read_only_native_vector_mode_plaquettes",
        "source": "femwell_native_archive_unqualified",
        "graph_consistent": bool(result.get("graph_consistent", False)) and not errors,
        "physics_qualified": False,
        "core_spp_identity_verified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "errors": errors,
        "noncommuting_plaquettes": result.get("noncommuting_plaquettes", []),
        "ambiguous_plaquettes": result.get("ambiguous_plaquettes", []),
        "edges_seen": result.get("edges_seen", 0),
        "native_input_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("GRID_READINESS.json", "experiment_protocol.json", "vector_mode_overlap_diagnostics.json")
        } if not errors else {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-grid", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Write derived report outside native grid")
    args = parser.parse_args()
    root = args.native_grid.resolve()
    if args.report is not None and (args.report.resolve() == root or root in args.report.resolve().parents):
        parser.error("derived report must be outside immutable native grid")
    result = audit_native_grid(args.native_grid)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result["graph_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
