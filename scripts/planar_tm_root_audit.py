"""Independent numerical planar TM root audit. NOT PCF/FEM/COMSOL evidence."""
import cmath
import math


def _decay(beta, k0, eps):
    kappa = cmath.sqrt(beta * beta - k0 * k0 * eps)
    return -kappa if kappa.real < 0 else kappa


def _boundary(beta, k0, eps_m, eps_d):
    return _decay(beta, k0, eps_d) / eps_d + _decay(beta, k0, eps_m) / eps_m


def numerical_planar_tm_root(wavelength_um, eps_m, eps_d, *, max_iter=120, tolerance=1e-10):
    """Numerically solve the TM boundary equation, not the closed-form dispersion."""
    if not (math.isfinite(wavelength_um) and wavelength_um > 0 and
            math.isfinite(eps_d) and eps_d > 0 and
            math.isfinite(eps_m.real) and math.isfinite(eps_m.imag) and
            eps_m.imag > 0 and eps_m.real < -eps_d):
        raise ValueError("invalid passive bound-SPP inputs")
    k0 = 2 * math.pi / wavelength_um
    beta = k0 * math.sqrt(eps_d) * (1.04 + 0.002j)
    converged = False
    for iteration in range(1, max_iter + 1):
        f = _boundary(beta, k0, eps_m, eps_d)
        scale = max(abs(_decay(beta, k0, eps_d)/eps_d),
                    abs(_decay(beta, k0, eps_m)/eps_m))
        if abs(f) / scale < tolerance:
            converged = True
            break
        h = 1e-6 * max(1.0, abs(beta))
        derivative = (_boundary(beta + h, k0, eps_m, eps_d) -
                      _boundary(beta - h, k0, eps_m, eps_d)) / (2*h)
        if abs(derivative) < 1e-14:
            break
        step = f / derivative
        accepted = False
        for j in range(20):
            candidate = beta - step * (0.5 ** j)
            if (candidate.real > k0*math.sqrt(eps_d) and candidate.imag > 0 and
                    abs(_boundary(candidate, k0, eps_m, eps_d)) < abs(f)):
                beta = candidate
                accepted = True
                break
        if not accepted:
            break
    if not converged:
        raise RuntimeError("planar TM numerical root did not converge")
    kd = _decay(beta, k0, eps_d)
    km = _decay(beta, k0, eps_m)
    if min(kd.real, km.real) <= 0 or beta.imag <= 0:
        raise RuntimeError("nonbound or amplifying root")
    return {
        "neff_real": (beta/k0).real,
        "neff_imag": (beta/k0).imag,
        "boundary_relative_residual": abs(_boundary(beta, k0, eps_m, eps_d))/scale,
        "iterations": iteration,
        "method": "numerical_complex_Newton_TM_boundary",
        "geometry": "single_flat_interface",
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
