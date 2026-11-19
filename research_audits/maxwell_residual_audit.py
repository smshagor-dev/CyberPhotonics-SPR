"""Independent curl-equation residual diagnostic; not a PCF solver or physics qualification.

Phasor convention: exp(i beta z - i omega t). Input arrays: E,H (3,ny,nx),
complex epsilon_r (ny,nx). SI units throughout. Uniform transverse grid only.
"""
import numpy as np

C0 = 299792458.0
MU0 = 4e-7 * np.pi
EPS0 = 1 / (MU0 * C0**2)


def audit_uniform_fields(E, H, epsilon_r, *, dx_m, dy_m, wavelength_m,
                         beta_per_m, trim=2):
    E = np.asarray(E, dtype=np.complex128)
    H = np.asarray(H, dtype=np.complex128)
    er = np.asarray(epsilon_r, dtype=np.complex128)
    if E.ndim != 3 or E.shape[0] != 3 or H.shape != E.shape or er.shape != E.shape[1:]:
        raise ValueError("inconsistent E/H/epsilon shapes")
    ny, nx = er.shape
    if not isinstance(trim, int) or trim < 1 or min(ny, nx) <= 2 * trim + 2:
        raise ValueError("insufficient interior cells")
    if any(not np.isfinite(v) or v <= 0 for v in (dx_m, dy_m, wavelength_m)):
        raise ValueError("invalid physical spacing or wavelength")
    if not np.isfinite(beta_per_m.real) or not np.isfinite(beta_per_m.imag):
        raise ValueError("invalid propagation constant")
    if not (np.isfinite(E).all() and np.isfinite(H).all() and np.isfinite(er).all()):
        raise ValueError("nonfinite field/material data")
    if np.any(er.imag < -1e-12):
        raise ValueError("nonpassive epsilon for exp(-i omega t) convention")
    omega = 2 * np.pi * C0 / wavelength_m

    def curl(F):
        fx, fy, fz = F
        dzy, dzx = np.gradient(fz, dy_m, dx_m, edge_order=2)
        _, dxx = np.gradient(fy, dy_m, dx_m, edge_order=2)
        dy_y, _ = np.gradient(fx, dy_m, dx_m, edge_order=2)
        return np.stack((dzy - 1j * beta_per_m * fy,
                         1j * beta_per_m * fx - dzx,
                         dxx - dy_y))

    ce, ch = curl(E), curl(H)
    rhs_e = 1j * omega * MU0 * H
    rhs_h = -1j * omega * EPS0 * er[None] * E
    interior = (slice(None), slice(trim, -trim), slice(trim, -trim))

    def normalized(lhs, rhs):
        num = np.linalg.norm((lhs - rhs)[interior].ravel())
        den = np.linalg.norm(lhs[interior].ravel()) + np.linalg.norm(rhs[interior].ravel())
        if den <= 0:
            raise ValueError("zero electromagnetic field norm")
        return float(num / den)

    return {"curl_E_residual": normalized(ce, rhs_e),
            "curl_H_residual": normalized(ch, rhs_h),
            "interior_cells": (ny - 2 * trim) * (nx - 2 * trim),
            "diagnostic_only": True, "physics_qualified": False,
            "publication_ready": False}
