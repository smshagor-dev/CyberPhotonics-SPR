"""Conservative independent-axis eigenmode convergence diagnostic; never qualifies physics."""
import math


def audit(rows, tolerance=1e-3, residual_limit=1e-8):
    if not isinstance(rows, list) or not rows:
        return {"diagnostic_pass": False, "errors": ["missing numerical records"],
                "physics_qualified": False, "publication_ready": False, "comsol_verified": False}
    errors, grid, signatures = [], {}, set()
    for i, row in enumerate(rows):
        try:
            mesh, domain = row["mesh_um"], row["half_domain_um"]
            residual = row["relative_matrix_residual"]
            beta_parts = row["beta_per_um"]
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in (mesh, domain)):
                raise ValueError("invalid mesh/domain")
            if type(residual) not in (int, float) or not math.isfinite(residual) or not 0 <= residual <= residual_limit:
                raise ValueError("invalid eigenpair residual")
            if len(beta_parts) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in beta_parts):
                raise ValueError("invalid complex beta")
            beta = complex(*beta_parts)
            if abs(beta) == 0:
                raise ValueError("zero beta")
            for key in ("material_sha256", "topology_sha256"):
                digest = row[key]
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("invalid source digest")
            for key in ("wavelength_um", "analyte_ri"):
                v = row[key]
                if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                    raise ValueError("invalid optical coordinate")
            boundary = row["boundary_condition"]
            if not isinstance(boundary, str) or not boundary:
                raise ValueError("missing boundary condition")
            signatures.add((row["material_sha256"], row["topology_sha256"], row["wavelength_um"], row["analyte_ri"], boundary))
            key = (float(mesh), float(domain))
            if key in grid:
                errors.append("duplicate mesh/domain combinations")
            grid[key] = beta
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"record {i}: {exc}")
    if len(signatures) != 1:
        errors.append("source/geometry/optical condition changed across sweep")
    meshes = sorted({m for m, _ in grid})
    domains = sorted({d for _, d in grid})
    mesh_drift = domain_drift = None
    if len(meshes) < 3 or len(domains) < 3:
        errors.append("independent three-level mesh and domain sweeps required")
    else:
        common_domains = sorted(set(d for m, d in grid if m == meshes[0]) & set(d for m, d in grid if m == meshes[1]))
        common_meshes = sorted(set(m for m, d in grid if d == domains[-1]) & set(m for m, d in grid if d == domains[-2]))
        if not common_domains or not common_meshes:
            errors.append("missing common-coordinate convergence pairs")
        else:
            d, m = common_domains[-1], common_meshes[0]
            mesh_drift = abs(grid[meshes[0], d] - grid[meshes[1], d]) / abs(grid[meshes[0], d])
            domain_drift = abs(grid[m, domains[-1]] - grid[m, domains[-2]]) / abs(grid[m, domains[-1]])
            if mesh_drift > tolerance:
                errors.append("mesh drift exceeds tolerance")
            if domain_drift > tolerance:
                errors.append("domain drift exceeds tolerance")
    return {"diagnostic_pass": not errors, "mesh_relative_drift": mesh_drift,
            "domain_relative_drift": domain_drift, "errors": errors,
            "physics_qualified": False, "publication_ready": False,
            "comsol_verified": False, "pml_verified": False,
            "independent_solver_verified": False}
