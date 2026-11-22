"""Independent complex Newton root of planar TM-SPP dispersion; NOT PCF evidence."""
import cmath
import math

def solve_planar_tm_root(wavelength_um, dielectric_epsilon, metal_epsilon, *, tolerance=1e-11, max_steps=80):
    wl = float(wavelength_um)
    ed, em = complex(dielectric_epsilon), complex(metal_epsilon)
    if not (math.isfinite(wl) and wl > 0 and all(math.isfinite(v) for v in (ed.real, ed.imag, em.real, em.imag))):
        raise ValueError("invalid finite wavelength or permittivity")
    if ed.imag != 0 or ed.real <= 0 or em.imag <= 0 or em.real >= -ed.real:
        raise ValueError("outside passive bound TM-SPP reference domain")
    if not (math.isfinite(tolerance) and 0 < tolerance < 1) or max_steps < 1:
        raise ValueError("invalid iteration settings")
    # Deliberately initialize away from the closed-form dispersion solution.
    n = cmath.sqrt(ed) * (1.02 + 0.01j)
    def residual(z):
        kd = cmath.sqrt(z*z-ed)
        km = cmath.sqrt(z*z-em)
        if kd.real < 0: kd = -kd
        if km.real < 0: km = -km
        return kd/ed + km/em, kd, km
    for step in range(max_steps):
        f,kd,km = residual(n)
        scale = abs(kd/ed)+abs(km/em)
        if scale and abs(f)/scale < tolerance and kd.real > 0 and km.real > 0 and n.imag > 0:
            return {"n_eff_real":n.real,"n_eff_imag":n.imag,"normalized_tm_boundary_residual":abs(f)/scale,
                    "iterations":step,"physics_qualified":False,"publication_ready":False,"comsol_verified":False}
        if abs(kd) < 1e-14 or abs(km) < 1e-14:
            raise ArithmeticError("singular transverse decay constant")
        derivative = n/(ed*kd) + n/(em*km)
        if abs(derivative) < 1e-15:
            raise ArithmeticError("singular TM dispersion derivative")
        delta=f/derivative
        accepted=False
        for j in range(20):
            trial=n-delta/(2**j)
            ft,kt,mt=residual(trial)
            if abs(ft) < abs(f) and kt.real > 0 and mt.real > 0 and trial.imag > 0:
                n=trial; accepted=True; break
        if not accepted: raise ArithmeticError("TM Newton iteration stalled")
    raise ArithmeticError("TM Newton iteration did not converge")
