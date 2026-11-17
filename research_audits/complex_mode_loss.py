"""Fail-closed native complex-mode loss calculation; NOT a Maxwell solver.

Convention exp(i beta z - i omega t). Native inputs require independent verification.
"""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ModeSample:
    wavelength_um: float
    analyte_ri: float
    neff_real: float
    neff_imag: float
    geometry_sha256: str
    solver_source: str
    mode_track_id: str
    native_artifact_sha256: str
    mode_tracking_verified: bool = False
    pml_converged: bool = False

def _hash(value):
    return len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def attenuation_db_per_cm(sample):
    """Power loss dB/cm = (20/ln 10) * 2*pi*Im(n_eff)/lambda_cm."""
    vals = (sample.wavelength_um, sample.analyte_ri, sample.neff_real, sample.neff_imag)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('nonfinite sample')
    if sample.wavelength_um <= 0 or sample.analyte_ri <= 1 or sample.neff_imag < 0:
        raise ValueError('invalid wavelength, RI, or passive-loss sign')
    if not (_hash(sample.geometry_sha256) and _hash(sample.native_artifact_sha256)):
        raise ValueError('missing SHA256 provenance')
    if not sample.solver_source or sample.solver_source.lower() in ('comsol','synthetic','fixture','mock'):
        raise ValueError('unsupported solver source')
    if not sample.mode_track_id:
        raise ValueError('missing mode track identity')
    return (20/math.log(10))*(2*math.pi/(sample.wavelength_um*1e-4))*sample.neff_imag

def fixed_geometry_spectrum(samples, *, require_qualified=True):
    """Do not infer resonance or publication readiness from these samples."""
    if len(samples) < 5:
        raise ValueError('insufficient spectral sampling')
    first = samples[0]
    if any(s.geometry_sha256 != first.geometry_sha256 or
           s.analyte_ri != first.analyte_ri or
           s.solver_source != first.solver_source or
           s.mode_track_id != first.mode_track_id for s in samples):
        raise ValueError('geometry/RI/source/mode changed')
    if require_qualified and any(not (s.mode_tracking_verified and s.pml_converged) for s in samples):
        raise ValueError('unverified mode tracking or PML')
    waves = [s.wavelength_um for s in samples]
    if any(b <= a for a,b in zip(waves,waves[1:])):
        raise ValueError('wavelengths must be strictly increasing')
    return [(s.wavelength_um,attenuation_db_per_cm(s)) for s in samples]
