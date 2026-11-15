"""Planar TM SPP FDFD control; NOT a full-vector PCF-SPR Maxwell solver.

Independent comparison to the exact semi-infinite metal/dielectric SPP
eigenvalue. Scientific claims about a fabricated or simulated PCF are prohibited
by the report's explicitly false evidence flags.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

HC_EV_UM = 1.239841984
GOLD_MODEL = "Rakic-1998-parameterized-Drude-Lorentz-from-repository"
GOLD_REFERENCE = "https://doi.org/10.1364/AO.37.005271"


def fitted_gold_epsilon(wavelength_um: float) -> complex:
    """Approximate optical Au permittivity, exp(-i omega t) convention.

    Mirrors the fitted parameters in sprpcf.simulation.dispersion.
    No independent measured material dataset is represented.
    """
    wavelength = float(wavelength_um)
    if not math.isfinite(wavelength) or not 0.45 <= wavelength <= 1.7:
        raise ValueError("Gold approximation is only tested in the 0.45–1.7 μm demonstration window")
    energy = HC_EV_UM / wavelength
    plasma = 9.03
    eps = 1.53 - 0.760 * plasma**2 / (energy * (energy + 1j * 0.053))
    for strength, damping, resonance in zip(
        (0.024, 0.010, 0.071, 0.601, 4.384),
        (0.241, 0.345, 0.870, 2.494, 2.214),
        (0.415, 0.830, 2.969, 4.304, 13.32),
    ):
        eps += strength * plasma**2 / (resonance**2 - energy**2 - 1j * damping * energy)
    return complex(eps)


def attenuation_db_per_cm(n_effective: complex, wavelength_um: float) -> float:
    """Power attenuation for E~exp(i beta z-i omega t), wavelength in μm."""
    if not 0 < wavelength_um < 1e6 or not math.isfinite(wavelength_um):
        raise ValueError("wavelength must be positive and finite")
    if not (math.isfinite(n_effective.real) and math.isfinite(n_effective.imag)):
        raise ValueError("nonfinite effective index")
    if n_effective.imag < -1e-12:
        raise ValueError("negative Im(n_eff) indicates gain/wrong propagation convention")
    k0_per_m = 2 * math.pi / (wavelength_um * 1e-6)
    return (20 / math.log(10)) * k0_per_m * max(0.0, n_effective.imag) / 100


def planar_spp_exact(wavelength_um: float, dielectric_ri: float, metal_epsilon: complex) -> complex:
    """Analytical TM surface plasmon at a *single semi-infinite* interface."""
    if not math.isfinite(dielectric_ri) or dielectric_ri <= 1:
        raise ValueError("dielectric refractive index must exceed unity")
    epsilon_d = dielectric_ri**2
    if not (math.isfinite(metal_epsilon.real) and math.isfinite(metal_epsilon.imag)):
        raise ValueError("metal permittivity must be finite")
    if metal_epsilon.real >= -epsilon_d or metal_epsilon.imag <= 0:
        raise ValueError("material is not a passive, bound-SPP metallic reference in this regime")
    effective = complex(np.sqrt(metal_epsilon * epsilon_d / (metal_epsilon + epsilon_d)))
    if effective.real <= 0 or effective.imag < 0:
        raise ValueError("analytical SPP returned an unphysical propagation branch")
    return effective


@dataclass(frozen=True)
class PlanarControl:
    mesh_step_um: float
    nodes: int
    n_effective: complex
    n_effective_exact: complex
    relative_error: float
    relative_residual: float
    near_interface_field_l2_fraction: float
    loss_db_per_cm: float
    x_um: np.ndarray
    h_y: np.ndarray


def solve_planar_tm(
    *, wavelength_um: float = 0.8, dielectric_ri: float = 1.36,
    half_domain_um: float = 6.0, mesh_step_um: float = 0.02,
    metal_epsilon: complex | None = None,
) -> PlanarControl:
    """Numerical 1D TM H_y finite-volume eigenmode control.

    d_x[(1/epsilon) d_x H_y] + (k0²-beta²/epsilon) H_y = 0.
    H_y=0 at remote boundaries; dielectric/metal interface lies between cells.
    This bound-mode control uses Dirichlet boundaries, *not PML*.
    """
    for name, value in (("half_domain_um", half_domain_um), ("mesh_step_um", mesh_step_um)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 1 <= half_domain_um <= 100:
        raise ValueError("half_domain_um must be between 1 and 100")
    if not 0.001 <= mesh_step_um <= 0.25:
        raise ValueError("mesh_step_um must be between 0.001 and 0.25")
    n = int(round(2 * half_domain_um / mesh_step_um))
    if n % 2:
        n += 1
    if not 32 <= n <= 10000:
        raise ValueError("mesh resolution must contain between 32 and 10000 cells")
    dx = 2 * half_domain_um / n
    if abs(dx - mesh_step_um) > mesh_step_um * 0.05:
        raise ValueError("half-domain size and mesh step require >5% adjustment")
    metal = fitted_gold_epsilon(wavelength_um) if metal_epsilon is None else complex(metal_epsilon)
    expected = planar_spp_exact(wavelength_um, dielectric_ri, metal)
    x = (np.arange(n) + 0.5) * dx - half_domain_um
    eps = np.where(x < 0, metal, dielectric_ri**2).astype(complex)
    inv_face = 2 / (eps[:-1] + eps[1:])
    inv_left = np.r_[1 / eps[0], inv_face]
    inv_right = np.r_[inv_face, 1 / eps[-1]]
    k0 = 2 * math.pi / wavelength_um
    diag = k0**2 * eps - eps * (inv_left + inv_right) / dx**2
    upper = eps[:-1] * inv_right[:-1] / dx**2
    lower = eps[1:] * inv_left[1:] / dx**2
    operator = diags((lower, diag, upper), offsets=(-1, 0, 1), format="csc")
    eigenvalues, vectors = eigs(
        operator, k=4, sigma=(k0 * expected)**2, which="LM", tol=1e-10,
        maxiter=20000, v0=np.ones(n, dtype=complex),
    )
    n_candidates = np.sqrt(eigenvalues.astype(complex)) / k0
    idx = int(np.argmin(np.abs(n_candidates - expected)))
    result = complex(n_candidates[idx])
    vec = vectors[:, idx]
    vec /= np.sqrt(float(np.vdot(vec, vec).real) * dx)
    phase = vec[int(np.argmin(abs(x)))]
    vec *= np.exp(-1j * np.angle(phase))
    residual = operator @ vec - eigenvalues[idx] * vec
    operator_inf = float(np.asarray(abs(operator).sum(axis=1)).max())
    rel_residual = float(np.linalg.norm(residual) / (operator_inf * np.linalg.norm(vec)))
    fraction = float(np.sum(np.abs(vec[abs(x) <= 1.0]) ** 2) * dx)
    return PlanarControl(
        mesh_step_um=dx, nodes=n, n_effective=result, n_effective_exact=expected,
        relative_error=float(abs(result - expected) / abs(expected)),
        relative_residual=rel_residual, near_interface_field_l2_fraction=fraction,
        loss_db_per_cm=attenuation_db_per_cm(result, wavelength_um), x_um=x, h_y=vec,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_planar_control(
    out: Path, *, wavelength_um: float = 0.8, dielectric_ri: float = 1.36,
    half_domain_um: float = 6.0,
    mesh_steps_um: tuple[float, ...] = (0.08, 0.04, 0.02, 0.01),
) -> dict[str, Any]:
    """Run and archive a genuine 1D physics control. Never qualify it as PCF."""
    if len(mesh_steps_um) < 3 or any(not math.isfinite(x) for x in mesh_steps_um):
        raise ValueError("At least three finite mesh-step levels are required")
    if any(a <= b for a, b in zip(mesh_steps_um, mesh_steps_um[1:])):
        raise ValueError("mesh steps must be strictly decreasing")
    controls = [
        solve_planar_tm(
            wavelength_um=wavelength_um, dielectric_ri=dielectric_ri,
            half_domain_um=half_domain_um, mesh_step_um=dx,
        )
        for dx in mesh_steps_um
    ]
    final = controls[-1]
    errors = [c.relative_error for c in controls]
    passed = bool(
        all(a > b for a, b in zip(errors, errors[1:]))
        and final.relative_error <= 0.005
        and final.relative_residual <= 1e-8
        and final.near_interface_field_l2_fraction >= 0.5
        and final.n_effective.imag > 0
        and final.loss_db_per_cm > 0
    )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fields_path = out / "planar_mode_field.csv"
    with fields_path.open("w", encoding="utf-8") as handle:
        handle.write("x_um,region,h_y_real,h_y_imag,h_y_abs\n")
        for x, h in zip(final.x_um, final.h_y):
            region = "metal" if x < 0 else "dielectric"
            handle.write(f"{x:.14g},{region},{h.real:.14g},{h.imag:.14g},{abs(h):.14g}\n")
    curves_path = out / "mesh_convergence.csv"
    with curves_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "mesh_step_um,nodes,n_eff_real,n_eff_imag,relative_error,"
            "relative_residual,loss_db_per_cm,near_interface_field_l2_fraction\n"
        )
        for c in controls:
            handle.write(
                f"{c.mesh_step_um:.14g},{c.nodes},{c.n_effective.real:.14g},"
                f"{c.n_effective.imag:.14g},{c.relative_error:.14g},"
                f"{c.relative_residual:.14g},{c.loss_db_per_cm:.14g},"
                f"{c.near_interface_field_l2_fraction:.14g}\n"
            )
    metal = fitted_gold_epsilon(wavelength_um)
    report = {
        "schema_version": 1,
        "experiment": "single_interface_planar_tm_sp_plasmon_control",
        "method": "1d-fdfd-TM-eigenmode-vs-analytic-SPP",
        "material": {
            "model": GOLD_MODEL, "doi": GOLD_REFERENCE,
            "epsilon_gold_real": metal.real, "epsilon_gold_imag": metal.imag,
            "measured_constants_used": False,
        },
        "wavelength_um": wavelength_um, "dielectric_ri": dielectric_ri,
        "half_domain_um": half_domain_um, "mesh_steps_um": list(mesh_steps_um),
        "analytical_n_eff": {"real": final.n_effective_exact.real, "imag": final.n_effective_exact.imag},
        "numerical_n_eff": {"real": final.n_effective.real, "imag": final.n_effective.imag},
        "relative_error": final.relative_error,
        "relative_residual": final.relative_residual,
        "mesh_error_monotonically_decreasing": all(a > b for a, b in zip(errors, errors[1:])),
        "pass": passed,
        "limitations": [
            "planar semi-infinite single interface only",
            "one-dimensional TM discretization, not full-vector PCF-SPR",
            "fitted gold optical constants, no measured input table",
            "Dirichlet bound-mode boundary, not PML",
            "analytical solution used only to identify/check the control mode",
        ],
        "scientific_submission_ready": False,
        "pcf_spr_electromagnetic_model_verified": False,
        "physical_sensor_sensitivity_verified": False,
        "open_fem_physics_qualified": False,
        "artifact_sha256": {
            fields_path.name: _sha256(fields_path),
            curves_path.name: _sha256(curves_path),
        },
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy_version, "platform": platform.platform(),
        },
    }
    (out / "CONTROL_READINESS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return report
