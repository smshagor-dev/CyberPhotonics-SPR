"""Analytic planar TM SPP reference. Not PCF or COMSOL evidence."""
import cmath
import math

def planar_tm_spp(wavelength_um, eps_m, eps_d):
    """Passive exp(-i omega t) convention; beta in 1/um, loss in dB/cm."""
    if not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("invalid wavelength")
    for e in (eps_m, eps_d):
        if not all(math.isfinite(v) for v in (e.real, e.imag)) or e.imag < 0:
            raise ValueError("nonfinite or active permittivity")
    if eps_d.real <= 0 or eps_m.real >= -eps_d.real or abs(eps_m+eps_d) < 1e-14:
        raise ValueError("no supported bound SPP regime")
    k0 = 2*math.pi/wavelength_um
    neff = cmath.sqrt(eps_m*eps_d/(eps_m+eps_d))
    if neff.real < 0: neff = -neff
    beta = k0*neff
    if beta.imag < -1e-12: raise ValueError("gain branch")
    km = cmath.sqrt(beta**2-k0**2*eps_m)
    kd = cmath.sqrt(beta**2-k0**2*eps_d)
    if km.real < 0: km = -km
    if kd.real < 0: kd = -kd
    if min(km.real,kd.real) <= 0: raise ValueError("unbound transverse field")
    residual = abs(km/eps_m+kd/eps_d)/(abs(km/eps_m)+abs(kd/eps_d))
    return dict(beta_per_um=beta,neff=neff,
                loss_db_cm=20/math.log(10)*max(0,beta.imag)*1e4,
                kappa_m_per_um=km,kappa_d_per_um=kd,
                tm_boundary_residual=residual,
                physics_qualified=False,publication_ready=False,
                comsol_verified=False)
