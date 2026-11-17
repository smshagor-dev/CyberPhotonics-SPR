"""Independent 1D TM Maxwell finite-volume cross-check, NOT PCF FEM evidence.

Solves eps*d_x(eps^-1*d_x H_y)+k0^2*eps*H_y=beta^2*H_y.
Dirichlet walls are NOT PML. Constant complex metal epsilon is illustrative,
NOT independently sourced dispersive Au.
"""
import argparse
import json
import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


def solve(dx_um=0.02, half_domain_um=3.2, wavelength_um=0.8,
          analyte_ri=1.36, metal_epsilon=-24 + 1.5j):
    values = (dx_um, half_domain_um, wavelength_um, analyte_ri,
              metal_epsilon.real, metal_epsilon.imag)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite physical input")
    if min(dx_um, half_domain_um, wavelength_um, analyte_ri) <= 0:
        raise ValueError("positive geometry, wavelength and RI required")
    if metal_epsilon.imag <= 0 or metal_epsilon.real >= -analyte_ri**2:
        raise ValueError("passive bound SPP requires negative metal permittivity")
    cells = round(2 * half_domain_um / dx_um)
    if cells < 40 or cells > 3000 or abs(cells * dx_um - 2 * half_domain_um) > 1e-8:
        raise ValueError("domain must contain 40..3000 exactly fitting cells")
    if cells % 2:
        raise ValueError("even cell count required for aligned interface")
    x = (np.arange(cells) + 0.5 - cells / 2) * dx_um
    eps_d = analyte_ri**2
    eps = np.where(x < 0, metal_epsilon, eps_d).astype(complex)
    a = 1 / eps
    face = 2 * a[:-1] * a[1:] / (a[:-1] + a[1:])
    lower = eps[1:] * face / dx_um**2
    upper = eps[:-1] * face / dx_um**2
    k0 = 2 * np.pi / wavelength_um
    diagonal = k0**2 * eps.copy()
    diagonal[1:] -= lower
    diagonal[:-1] -= upper
    diagonal[0] -= 2 / dx_um**2
    diagonal[-1] -= 2 / dx_um**2
    matrix = diags([lower, diagonal, upper], [-1, 0, 1], format="csc")
    analytic_beta = k0 * np.sqrt(metal_epsilon * eps_d / (metal_epsilon + eps_d))
    eigenvalues, vectors = eigs(matrix, k=6, sigma=analytic_beta**2,
                               tol=1e-10, maxiter=10000)
    betas = np.sqrt(eigenvalues)
    idx = int(np.argmin(abs(betas - analytic_beta)))
    beta = betas[idx]
    vector = vectors[:, idx]
    residual = np.linalg.norm(matrix @ vector - eigenvalues[idx] * vector) / (
        np.linalg.norm(matrix @ vector) + abs(eigenvalues[idx]) * np.linalg.norm(vector))
    return {
        "dx_um": dx_um, "half_domain_um": half_domain_um,
        "wavelength_um": wavelength_um, "analyte_ri": analyte_ri,
        "beta_numeric_per_um": [float(beta.real), float(beta.imag)],
        "beta_analytic_per_um": [float(analytic_beta.real), float(analytic_beta.imag)],
        "relative_beta_error": float(abs(beta - analytic_beta) / abs(analytic_beta)),
        "relative_matrix_residual": float(residual),
        "mode_candidate_count": len(eigenvalues),
        "method": "1d_cell_centered_tm_finite_volume_dirichlet_not_pml",
        "material_provenance": "illustrative_constant_complex_permittivity_not_measured_Au",
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [solve(dx_um=dx) for dx in (0.08, 0.04, 0.02, 0.01)]
    payload = {"records": records, "physics_qualified": False,
               "publication_ready": False, "comsol_verified": False,
               "limitation": "Planar 1D illustrative TM solver; not vector PCF FEM or dispersive Au"}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    for record in records:
        print(record["dx_um"], record["relative_beta_error"], record["relative_matrix_residual"])
    if not all(record["relative_matrix_residual"] < 1e-8 for record in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
