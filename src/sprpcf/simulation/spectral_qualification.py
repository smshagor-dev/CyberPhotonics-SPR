"""Fail-closed spectral qualification for open-source FEM pilot outputs.

Software gate only: cannot establish PML, modal identity or COMSOL equivalence.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicsEvidence:
    geometry: bool = False
    measured_au: bool = False
    mesh_convergence: bool = False
    pml_domain_convergence: bool = False
    core_spp_identity: bool = False
    independent_solver: bool = False


def qualify_spectrum(wavelength_nm, loss_db_cm, evidence=PhysicsEvidence()):
    """Do not release resonance/FWHM for underresolved or unverified spectra."""
    x, y = list(wavelength_nm), list(loss_db_cm)
    failures = []
    if len(x) != len(y) or len(x) < 11:
        failures.append("minimum_11_matched_samples")
    if not x or not all(math.isfinite(v) for v in x + y):
        failures.append("nonfinite_or_empty")
    if any(v <= 0 for v in x) or any(a >= b for a, b in zip(x, x[1:])):
        failures.append("wavelength_axis")
    if any(v < 0 for v in y):
        failures.append("negative_loss")
    for name in evidence.__dataclass_fields__:
        if getattr(evidence, name) is not True:
            failures.append("unverified_" + name)
    if y and all(math.isfinite(v) for v in y):
        peak = max(range(len(y)), key=y.__getitem__)
        if peak in (0, len(y) - 1):
            failures.append("unbracketed_peak")
        else:
            half = min(y) + (y[peak] - min(y)) / 2
            left = [i for i in range(peak) if y[i] <= half < y[i+1]]
            right = [i for i in range(peak, len(y)-1) if y[i] > half >= y[i+1]]
            if not left or not right:
                failures.append("unbracketed_fwhm")
            elif len(x) > 2 and all(a < b for a, b in zip(x, x[1:])):
                lo, hi = left[-1], right[0]
                span = x[hi+1] - x[lo]
                step = max(x[i+1] - x[i] for i in range(lo, hi+1))
                if span <= 0 or step / span > 0.1:
                    failures.append("underresolved_fwhm")
    return {"qualified": not failures, "failures": sorted(set(failures)),
            "resonance_nm": None, "fwhm_nm": None,
            "sensitivity_nm_per_riu": None,
            "scientific_submission_ready": False}
