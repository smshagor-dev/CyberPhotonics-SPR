"""Independent planar TM boundary residual audit; never PCF/COMSOL evidence."""
import cmath
import math


def audit_tm_interface(wavelength_um, eps_m, eps_d, n_eff, tolerance=1e-8):
    """Check kappa_d/eps_d + kappa_m/eps_m = 0 for decaying half-spaces."""
    try:
        wavelength_um = float(wavelength_um)
        eps_m, eps_d, n_eff = complex(eps_m), complex(eps_d), complex(n_eff)
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid numeric input") from exc
    values = (wavelength_um, tolerance, eps_m.real, eps_m.imag, eps_d.real,
              eps_d.imag, n_eff.real, n_eff.imag)
    if not all(math.isfinite(x) for x in values) or wavelength_um <= 0 or not 0 < tolerance < 1:
        raise ValueError("nonfinite or invalid wavelength/tolerance")
    if eps_d.imag != 0 or eps_d.real <= 0 or eps_m.imag <= 0 or eps_m.real >= -eps_d.real:
        raise ValueError("outside passive planar SPP reference domain")
    if n_eff.real <= 0 or n_eff.imag < 0:
        raise ValueError("invalid propagation branch")
    kd = cmath.sqrt(n_eff * n_eff - eps_d)
    km = cmath.sqrt(n_eff * n_eff - eps_m)
    if kd.real < 0:
        kd = -kd
    if km.real < 0:
        km = -km
    left, right = kd / eps_d, km / eps_m
    scale = abs(left) + abs(right)
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("degenerate boundary terms")
    residual = abs(left + right) / scale
    return {"normalized_tm_boundary_residual": residual,
            "boundary_consistent": residual <= tolerance,
            "decaying_half_spaces": kd.real > 0 and km.real > 0,
            "publication_qualified": False, "pcf_mode_verified": False,
            "comsol_verified": False}
