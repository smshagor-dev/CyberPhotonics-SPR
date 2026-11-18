"""Independent hyperbolic TM transfer-matrix check of illustrative planar film roots.

Not an independent FEM solver, PCF model, measured Au evidence, or sensor result.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
try:
    from .film_thickness_continuation import campaign
except ImportError:
    from film_thickness_continuation import campaign


def transfer_residual(neff, thickness_um, wavelength_um, epsilon_d, epsilon_m):
    k0 = 2 * np.pi / wavelength_um
    beta = k0 * neff
    qd = np.sqrt(beta**2 - k0**2 * epsilon_d + 0j)
    qm = np.sqrt(beta**2 - k0**2 * epsilon_m + 0j)
    if qd.real < 0:
        qd = -qd
    if qm.real < 0:
        qm = -qm
    yd, ym = epsilon_d / qd, epsilon_m / qm
    return (2 * yd * ym + (yd**2 + ym**2) * np.tanh(qm * thickness_um)) / ym**2


def crosscheck(thicknesses=(.05, .075, .1, .15, .2, .3, .5),
               wavelength_um=.8, epsilon_d=1.77, epsilon_m=-24+1.5j,
               tolerance=1e-7):
    baseline = campaign(thicknesses, wavelength_um, epsilon_d, epsilon_m)
    reference = np.sqrt(epsilon_d * epsilon_m / (epsilon_d + epsilon_m))
    rows = []
    for row in baseline['rows']:
        t = row['thickness_um']
        if row['status'] != 'converged':
            rows.append({'thickness_um': t, 'status': 'baseline_failed'})
            continue
        seed = complex(*row['neff'])
        def fun(x):
            z = transfer_residual(complex(*x), t, wavelength_um, epsilon_d, epsilon_m)
            return [z.real, z.imag]
        candidates = []
        for guess in (reference, seed):
            result = root(fun, [guess.real, guess.imag], tol=1e-12)
            z = complex(*result.x)
            residual = float(abs(transfer_residual(z, t, wavelength_um, epsilon_d, epsilon_m)))
            if np.isfinite(residual):
                candidates.append((abs(z-seed), z, residual, result))
        if not candidates:
            rows.append({'thickness_um': t, 'status': 'solver_failed'})
            continue
        drift, z, residual, result = min(candidates, key=lambda x: x[0])
        passed = bool(residual < 1e-9 and drift < tolerance and z.imag >= 0)
        rows.append({'thickness_um': t, 'status': 'matched' if passed else 'failed',
                     'neff_transfer': [float(z.real), float(z.imag)],
                     'neff_reference': row['neff'], 'absolute_neff_difference': float(drift),
                     'transfer_residual': residual, 'solver_success': bool(result.success)})
    return {'source_kind': 'illustrative_planar_tm_algebraic_crosscheck',
            'physics_qualified': False, 'publication_ready': False,
            'comsol_verified': False, 'independent_fem_crosscheck': False,
            'tolerance_neff': tolerance,
            'matched': sum(r['status'] == 'matched' for r in rows),
            'failed': sum(r['status'] != 'matched' for r in rows), 'rows': rows}


def write_immutable(path, report):
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as f:
        f.write(payload)
    return hashlib.sha256(payload).hexdigest()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = crosscheck()
    digest = write_immutable(args.output, result)
    print(json.dumps({'matched': result['matched'], 'failed': result['failed'],
                      'sha256': digest}))
