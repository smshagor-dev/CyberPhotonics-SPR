"""Transverse Poynting flux and passive absorption diagnostic, exp(-i omega t).

Input: counterclockwise triangle mesh in metres; nodal complex E,H phasors;
piecewise-constant element Im(epsilon_r). No physics qualification implied.
"""
import numpy as np

EPS0 = 8.8541878128e-12
C0 = 299792458.0

def audit_transverse_fields(xy_m, triangles, e_xyz, h_xyz, eps_imag, wavelength_m):
    xy = np.asarray(xy_m, dtype=float)
    tri = np.asarray(triangles)
    E = np.asarray(e_xyz, dtype=complex)
    H = np.asarray(h_xyz, dtype=complex)
    loss = np.asarray(eps_imag, dtype=float)
    wl = float(wavelength_m)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("invalid xy")
    if tri.ndim != 2 or tri.shape[1] != 3 or not np.issubdtype(tri.dtype, np.integer):
        raise ValueError("invalid triangles")
    if E.shape != (len(xy), 3) or H.shape != E.shape or loss.shape != (len(tri),):
        raise ValueError("inconsistent field/material dimensions")
    if not all(np.isfinite(a).all() for a in (xy, E, H, loss)):
        raise ValueError("nonfinite inputs")
    if not np.isfinite(wl) or wl <= 0 or np.any(loss < 0):
        raise ValueError("invalid wavelength or active material")
    if np.any(tri < 0) or np.any(tri >= len(xy)) or any(len(set(t)) != 3 for t in tri):
        raise ValueError("invalid connectivity")
    p = xy[tri]
    twice = ((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])
             -(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))
    if np.any(twice <= 0):
        raise ValueError("degenerate or clockwise triangle")
    area = twice/2
    bary = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    Eq = np.einsum("qv,mvc->mqc", bary, E[tri])
    Hq = np.einsum("qv,mvc->mqc", bary, H[tri])
    sz = 0.5*np.real(Eq[:,:,0]*np.conj(Hq[:,:,1])-Eq[:,:,1]*np.conj(Hq[:,:,0]))
    q = 0.5*(2*np.pi*C0/wl)*EPS0*loss[:,None]*np.sum(np.abs(Eq)**2,axis=2)
    pz = float(np.sum(area*np.mean(sz,axis=1)))
    absorbed = float(np.sum(area*np.mean(q,axis=1)))
    if pz <= 0:
        raise ValueError("nonpositive forward power")
    alpha = absorbed/pz
    return {"forward_power_w":pz, "absorption_w_per_m":absorbed,
            "absorption_alpha_np_per_m":alpha,
            "absorption_loss_db_per_cm":alpha*10/np.log(10)/100,
            "physics_qualified":False, "publication_ready":False}
