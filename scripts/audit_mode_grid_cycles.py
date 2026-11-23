"""Fail-closed plaquette cycle audit of unqualified same-mesh FEM mode overlaps.

Cycle closure is necessary, NOT sufficient, for core/SPP identification.
No sensor performance, PML, COMSOL, or publication qualification is inferred.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path


def audit_grid(rows: list[dict], edges: list[dict], min_margin: float = 0.0) -> dict:
    if isinstance(min_margin, bool) or not isinstance(min_margin, (float, int)) or not math.isfinite(min_margin) or min_margin < 0:
        raise ValueError("min_margin must be finite and nonnegative")
    errors = []
    nodes = {}
    for row in rows:
        try:
            r, w, m = (int(row[k]) for k in ("ri_index", "wavelength_index", "mode_index_native_untracked"))
            z = complex(float(row["n_eff_real"]), float(row["n_eff_imag"]))
            if min(r, w, m) < 0 or not all(map(math.isfinite, (z.real, z.imag))) or m in nodes.get((r, w), set()):
                raise ValueError("negative, duplicate or nonfinite eigenmode")
            nodes.setdefault((r, w), set()).add(m)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid eigenmode row: {exc}")
    nr = max((r for r, _ in nodes), default=-1) + 1
    nw = max((w for _, w in nodes), default=-1) + 1
    modes = set(range(len(next(iter(nodes.values()))))) if nodes else set()
    if nr < 2 or nw < 2 or not modes:
        errors.append("requires at least 2 RI, 2 wavelengths and one mode")
    for r in range(nr):
        for w in range(nw):
            if nodes.get((r, w)) != modes:
                errors.append(f"missing/noncontiguous modes at ({r},{w})")
    maps = {}
    for edge in edges:
        try:
            if edge["varying"] == "wavelength":
                r = int(edge["ri_index"])
                a, b = (r, int(edge["previous_wl_index"])), (r, int(edge["current_wl_index"]))
            elif edge["varying"] == "analyte_ri":
                w = int(edge["wavelength_index"])
                a, b = (int(edge["previous_ri_index"]), w), (int(edge["current_ri_index"]), w)
            else:
                raise ValueError("unknown edge direction")
            if (b[0]-a[0], b[1]-a[1]) not in ((1, 0), (0, 1)):
                raise ValueError("nonadjacent edge")
            i, j = int(edge["previous_mode_index"]), int(edge["current_mode_index"])
            margin = float(edge["bidirectional_min_overlap_margin"])
            overlap = float(edge["vector_eh_poynting_overlap_abs"])
            if not all(map(math.isfinite, (margin, overlap))) or overlap < 0:
                raise ValueError("invalid overlap")
            if edge.get("assignment_ambiguous_nonpositive_margin") is True or margin <= min_margin:
                errors.append(f"ambiguous overlap at {a}->{b} mode {i}")
            mapping = maps.setdefault((a, b), {})
            if i in mapping:
                raise ValueError("duplicate edge source")
            mapping[i] = j
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"invalid overlap edge: {exc}")
    for (a, b), mapping in maps.items():
        if nodes.get(a) != modes or nodes.get(b) != modes or set(mapping) != modes or set(mapping.values()) != modes:
            errors.append(f"nonbijective/incomplete assignment {a}->{b}")
    for r in range(nr):
        for w in range(nw):
            for b in ((r+1, w), (r, w+1)):
                if b[0] < nr and b[1] < nw and ((r, w), b) not in maps:
                    errors.append(f"missing edge {(r,w)}->{b}")
    cycles = 0
    for r in range(nr-1):
        for w in range(nw-1):
            a, b, c, d = (r,w), (r,w+1), (r+1,w), (r+1,w+1)
            if not all(k in maps for k in ((a,b), (b,d), (a,c), (c,d))):
                errors.append(f"incomplete cycle at ({r},{w})")
                continue
            cycles += 1
            for mode in modes:
                try:
                    via_w = maps[(b,d)][maps[(a,b)][mode]]
                    via_r = maps[(c,d)][maps[(a,c)][mode]]
                    if via_w != via_r:
                        errors.append(f"mode transport nonclosure at ({r},{w}) mode {mode}")
                except KeyError:
                    errors.append(f"incomplete transport at ({r},{w}) mode {mode}")
    return dict(schema_version=1, source_evidence_class="unqualified_solver_diagnostic",
                ri_count=nr, wavelength_count=nw, mode_count=len(modes),
                cycles_checked=cycles, failures=errors, cycle_consistent=not errors,
                physical_core_spp_identity_verified=False, pml_converged=False,
                publication_ready=False, comsol_verified=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("native_directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-margin", type=float, default=0.0)
    args = p.parse_args()
    with (args.native_directory / "raw_eigenmode_grid.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    edges = json.loads((args.native_directory / "vector_mode_overlap_diagnostics.json").read_text(encoding="utf-8"))
    report = audit_grid(rows, edges, args.min_margin)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    if report["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
