"""Spectral-grid refinement diagnostic. Does not qualify Maxwell physics."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Spectrum:
    wavelength_nm: tuple[float, ...]
    loss_db_per_cm: tuple[float, ...]
    geometry_sha256: str
    material_sha256: str
    mode_id: str
    source_archive_sha256: str

def _digest(s):
    return isinstance(s, str) and len(s) == 64 and all(c in '0123456789abcdef' for c in s)

def _peak(s):
    x, y = s.wavelength_nm, s.loss_db_per_cm
    if len(x) < 9 or len(x) != len(y):
        raise ValueError('at least nine samples required')
    if not all(math.isfinite(v) for v in (*x, *y)) or any(v < 0 for v in y):
        raise ValueError('nonfinite or negative loss')
    if any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError('wavelengths must increase')
    maxima = [i for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if len(maxima) != 1:
        raise ValueError('ambiguous or absent peak')
    i = maxima[0]
    if i < 2 or i > len(y)-3:
        raise ValueError('peak near boundary')
    left = (y[i]-y[i-1])/(x[i]-x[i-1])
    right = (y[i+1]-y[i])/(x[i+1]-x[i])
    curvature = 2*(right-left)/(x[i+1]-x[i-1])
    if curvature >= 0:
        raise ValueError('nonconcave peak')
    derivative = (left*(x[i+1]-x[i]) + right*(x[i]-x[i-1]))/(x[i+1]-x[i-1])
    vertex = x[i]-derivative/curvature
    if not x[i-1] <= vertex <= x[i+1]:
        raise ValueError('unresolved peak')
    baseline = max(y[0], y[-1])
    half = baseline+(y[i]-baseline)/2
    if half <= baseline:
        raise ValueError('no prominence')
    def crossing(indices):
        for j in indices:
            if (y[j]-half)*(y[j+1]-half) <= 0 and y[j] != y[j+1]:
                return x[j]+(half-y[j])*(x[j+1]-x[j])/(y[j+1]-y[j])
        raise ValueError('unbracketed half-prominence crossing')
    a = crossing(range(i-1, -1, -1))
    b = crossing(range(i, len(y)-1))
    return vertex, b-a

def assess_spectral_refinement(spectra, *, max_resonance_drift_nm=0.5,
                               max_fwhm_relative_drift=0.05):
    """Three nested, independently archived grids; software gate only."""
    if len(spectra) != 3:
        raise ValueError('three grids required')
    if not (math.isfinite(max_resonance_drift_nm) and max_resonance_drift_nm > 0 and
            math.isfinite(max_fwhm_relative_drift) and max_fwhm_relative_drift > 0):
        raise ValueError('invalid tolerance')
    for field in ('geometry_sha256', 'material_sha256', 'mode_id'):
        if len({getattr(s, field) for s in spectra}) != 1:
            raise ValueError('inconsistent '+field)
    if not spectra[0].mode_id.strip():
        raise ValueError('missing mode ID')
    hashes = [s.source_archive_sha256 for s in spectra]
    if len(set(hashes)) != 3 or not all(_digest(h) for h in hashes):
        raise ValueError('invalid/reused archive digest')
    if not all(_digest(s.geometry_sha256) and _digest(s.material_sha256) for s in spectra):
        raise ValueError('invalid provenance digest')
    sizes = [len(s.wavelength_nm) for s in spectra]
    if not sizes[0] < sizes[1] < sizes[2]:
        raise ValueError('grids not refined')
    for coarse, fine in zip(spectra, spectra[1:]):
        if coarse.wavelength_nm[0] != fine.wavelength_nm[0] or coarse.wavelength_nm[-1] != fine.wavelength_nm[-1]:
            raise ValueError('domain changed')
        if not set(coarse.wavelength_nm).issubset(fine.wavelength_nm):
            raise ValueError('grids not nested')
    metrics = [_peak(s) for s in spectra]
    dr = abs(metrics[2][0]-metrics[1][0])
    df = abs(metrics[2][1]-metrics[1][1])/metrics[2][1]
    return {'resonance_nm': [m[0] for m in metrics],
            'fwhm_nm': [m[1] for m in metrics],
            'finest_resonance_drift_nm': dr,
            'finest_fwhm_relative_drift': df,
            'spectral_grid_pass': dr <= max_resonance_drift_nm and df <= max_fwhm_relative_drift,
            'qualified_physics': False, 'publication_ready': False,
            'qualification_note': 'Interpolation stability only; no solver/PML/material qualification.'}
