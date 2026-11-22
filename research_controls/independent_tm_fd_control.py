"""Independent 1D TM finite-difference control; NOT full-vector PCF FEM evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs


def solve(dx, half_width=3.0, wavelength=0.8, eps_m=-12+1.2j, eps_d=2.25):
    if not (0 < dx <= 0.05 and half_width > 0 and wavelength > 0 and eps_m.imag > 0 and eps_d > 0):
        raise ValueError("invalid control parameters")
    n = int(round(2 * half_width / dx))
    if n % 2:
        n += 1
    dx = 2 * half_width / n
    x = -half_width + (np.arange(n) + 0.5) * dx
    eps = np.where(x < 0, eps_m, eps_d).astype(complex)
    inv = 1 / eps
    face = 2 * inv[:-1] * inv[1:] / (inv[:-1] + inv[1:])
    lower = eps[1:] * face / dx**2
    upper = eps[:-1] * face / dx**2
    diag = eps * (2 * np.pi / wavelength)**2
    diag -= eps * np.r_[inv[0], face] / dx**2
    diag -= eps * np.r_[face, inv[-1]] / dx**2
    matrix = diags([lower, diag, upper], [-1, 0, 1], format="csc")
    beta_ref = (2 * np.pi / wavelength) * np.sqrt(eps_m * eps_d / (eps_m + eps_d))
    values, fields = eigs(matrix, k=6, sigma=beta_ref**2, tol=1e-10, maxiter=30000)
    betas = np.sqrt(values)
    betas = np.where(betas.real < 0, -betas, betas)
    index = int(np.argmin(abs(betas - beta_ref)))
    beta = complex(betas[index])
    field = fields[:, index]
    near = np.sum(abs(field[abs(x) < 0.5])**2) / np.sum(abs(field)**2)
    k0 = 2 * np.pi / wavelength
    return dict(
        dx_um=dx,
        neff_real=beta.real / k0,
        neff_imag=beta.imag / k0,
        analytic_neff_real=beta_ref.real / k0,
        analytic_neff_imag=beta_ref.imag / k0,
        relative_beta_error=abs(beta - beta_ref) / abs(beta_ref),
        central_field_fraction=float(near),
        domain_half_width_um=half_width,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {
        "solver": "independent_1d_tm_finite_difference_control",
        "source_class": "analytical_discretization_control",
        "comsol_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
        "gold_permittivity_source": "illustrative_not_authenticated",
        "boundary": "finite_domain_dirichlet_no_pml",
        "runs": [solve(d) for d in (0.04, 0.02, 0.01)],
    }
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    path = Path(args.output)
    with path.open("xb") as output:
        output.write(raw)
    print(json.dumps({"output": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "runs": result["runs"]}))


if __name__ == "__main__":
    main()
