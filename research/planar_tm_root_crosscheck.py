"""Independent planar TM-SPP root verification; NOT a PCF or measured sensor result.
Fields exp(i*beta*z-i*omega*t); passive propagation has Im(beta)>0.
"""
import cmath
import math
from scipy.optimize import root

def _kappa(beta, eps, k0):
    q = cmath.sqrt(beta*beta-eps*k0*k0)
    return q if q.real >= 0 else -q

def solve_planar_tm(wavelength_nm, eps_m, eps_d, tolerance=1e-9):
    if not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength must be positive and finite")
    if not all(math.isfinite(v) for z in (eps_m, eps_d) for v in (z.real, z.imag)):
        raise ValueError("permittivity must be finite")
    if eps_m.imag <= 0 or eps_d.imag < 0 or eps_d.real <= 0:
        raise ValueError("passive metal and dielectric required")
    k0 = 2*math.pi/(wavelength_nm*1e-9)
    analytic = k0*cmath.sqrt(eps_m*eps_d/(eps_m+eps_d))
    if analytic.real < 0:
        analytic = -analytic
    def residual(x):
        beta = complex(*x)*k0
        r = _kappa(beta,eps_m,k0)/eps_m + _kappa(beta,eps_d,k0)/eps_d
        return [r.real/k0, r.imag/k0]
    fit = root(residual, [analytic.real/k0*1.0001, analytic.imag/k0*0.9999], tol=1e-12)
    beta = complex(*fit.x)*k0
    error = abs(beta-analytic)/abs(analytic)
    boundary = math.hypot(*residual(fit.x))
    bound = _kappa(beta,eps_m,k0).real > 0 and _kappa(beta,eps_d,k0).real > 0
    passed = bool(fit.success and error <= tolerance and boundary <= tolerance and beta.imag > 0 and bound)
    return dict(wavelength_nm=wavelength_nm, numerical_neff=[(beta/k0).real,(beta/k0).imag],
                analytic_neff=[(analytic/k0).real,(analytic/k0).imag],
                relative_error=error, normalized_boundary_residual=boundary,
                root_success=bool(fit.success), bound_mode=bool(bound), pass_check=passed,
                source_class="constructed_planar_crosscheck_not_pcf",
                physics_qualified=False, publication_ready=False)
