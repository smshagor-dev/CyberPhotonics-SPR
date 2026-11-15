"""Fail-closed transverse E/H mode overlap diagnostics; not physics qualification."""
import numpy as np

def power_overlap(ea, ha, eb, hb, weights):
    """Symmetrized Poynting overlap on identical mesh/quadrature, diagnostic only.

    Inputs are complex transverse E/H samples (N,2); weights are positive.
    Does not establish core/SPP identity, biorthogonality or mesh convergence.
    """
    a,b,c,d = [np.asarray(x, dtype=np.complex128) for x in (ea,ha,eb,hb)]
    w = np.asarray(weights, dtype=float)
    if any(x.ndim != 2 or x.shape[1] != 2 for x in (a,b,c,d)) or not (a.shape == b.shape == c.shape == d.shape):
        raise ValueError("E/H samples must share (N,2) shape")
    if w.shape != (a.shape[0],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("positive finite quadrature weights required")
    if not all(np.all(np.isfinite(x)) for x in (a,b,c,d)):
        raise ValueError("nonfinite fields")
    def flux(e,h):
        return np.sum(w * (e[:,0]*np.conj(h[:,1])-e[:,1]*np.conj(h[:,0])))
    pa,pb=flux(a,b),flux(c,d)
    if not (np.isfinite(pa) and np.isfinite(pb)) or abs(pa)<=1e-20 or abs(pb)<=1e-20:
        raise ValueError("zero/invalid modal flux")
    cross=0.5*(flux(a,d)+np.conj(flux(c,b)))
    score=abs(cross)/np.sqrt(abs(pa)*abs(pb))
    if not np.isfinite(score):
        raise ValueError("nonfinite overlap")
    return float(score)

def attenuation_db_per_cm(neff, wavelength_um):
    """Intensity loss, exp(+i beta z - i omega t), lambda in micrometres."""
    n=complex(neff)
    lam=float(wavelength_um)
    if not (np.isfinite(n.real) and np.isfinite(n.imag) and np.isfinite(lam) and lam>0 and n.imag>=0):
        raise ValueError("invalid passive neff or wavelength")
    return float(8.686*(2*np.pi/lam)*n.imag*1e4)
