"""Conservative square-loop audit of native vector-mode assignments.

Graph consistency is necessary but NOT sufficient for physical core/SPP identity.
No synthetic test input is numerical-physics evidence.
"""
from __future__ import annotations

import math


def audit_mode_plaquettes(edges, *, n_ri: int, n_wavelength: int, n_modes: int) -> dict:
    if min(n_ri, n_wavelength, n_modes) < 2:
        raise ValueError("two RI, wavelength and mode samples required")
    directed = {}
    errors = []
    for edge in edges:
        try:
            axis = edge["varying"]
            if axis == "wavelength":
                src = (int(edge["ri_index"]), int(edge["previous_wl_index"]))
                dst = (int(edge["ri_index"]), int(edge["current_wl_index"]))
                valid = src[0] == dst[0] and dst[1] == src[1] + 1
            elif axis == "analyte_ri":
                src = (int(edge["previous_ri_index"]), int(edge["wavelength_index"]))
                dst = (int(edge["current_ri_index"]), int(edge["wavelength_index"]))
                valid = src[1] == dst[1] and dst[0] == src[0] + 1
            else:
                raise ValueError("unknown edge axis")
            a, b = int(edge["previous_mode_index"]), int(edge["current_mode_index"])
            margin = float(edge["bidirectional_min_overlap_margin"])
            overlap = float(edge["vector_eh_poynting_overlap_abs"])
            if not (valid and 0 <= src[0] < n_ri and 0 <= dst[0] < n_ri
                    and 0 <= src[1] < n_wavelength and 0 <= dst[1] < n_wavelength
                    and 0 <= a < n_modes and 0 <= b < n_modes
                    and math.isfinite(margin) and math.isfinite(overlap)
                    and 0 <= overlap <= 1 + 1e-8):
                raise ValueError("invalid edge geometry, mode or overlap")
            key = (src, dst, a)
            if key in directed:
                raise ValueError("duplicate directed mode edge")
            directed[key] = (b, margin)
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            errors.append(str(exc))
    bad_cycles, weak_cycles = [], []
    for ri in range(n_ri - 1):
        for wi in range(n_wavelength - 1):
            tl, tr, bl, br = (ri, wi), (ri, wi + 1), (ri + 1, wi), (ri + 1, wi + 1)
            for mode in range(n_modes):
                try:
                    top, m1 = directed[(tl, tr, mode)]
                    right, m2 = directed[(tr, br, top)]
                    left, m3 = directed[(tl, bl, mode)]
                    bottom, m4 = directed[(bl, br, left)]
                except KeyError:
                    errors.append(f"missing plaquette edge at {ri},{wi},mode={mode}")
                    continue
                if right != bottom:
                    bad_cycles.append([ri, wi, mode, right, bottom])
                if min(m1, m2, m3, m4) <= 0:
                    weak_cycles.append([ri, wi, mode])
    expected = n_modes * (n_ri * (n_wavelength - 1) + n_wavelength * (n_ri - 1))
    if len(directed) != expected:
        errors.append(f"incomplete edge inventory: {len(directed)} != {expected}")
    return {
        "graph_consistent": not errors and not bad_cycles and not weak_cycles,
        "physics_qualified": False,
        "core_spp_identity_verified": False,
        "edges_seen": len(directed),
        "noncommuting_plaquettes": bad_cycles,
        "ambiguous_plaquettes": weak_cycles,
        "errors": errors,
    }
