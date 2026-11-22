"""Necessary, never sufficient, spectral sampling screen (not physics evidence)."""
from __future__ import annotations
import math


def screen_resonance_sampling(wavelengths_um, losses_db_cm, *, min_samples=21,
                              min_halfmax_intervals=3, max_gap_ratio=3.0):
    """Screen a single *physically tracked* loss branch for FWHM feasibility.

    Baseline-relative half-height must be bracketed on both sides. No FWHM,
    sensitivity or publication claim is computed by this screening function.
    """
    if (type(min_samples) is not int or min_samples < 7 or
            type(min_halfmax_intervals) is not int or min_halfmax_intervals < 2):
        raise ValueError("invalid spectral sampling thresholds")
    if isinstance(max_gap_ratio, bool) or not isinstance(max_gap_ratio, (int, float)) or not math.isfinite(max_gap_ratio) or max_gap_ratio < 1:
        raise ValueError("invalid wavelength gap ratio")
    failures = []
    try:
        raw_xs = list(wavelengths_um)
        raw_ys = list(losses_db_cm)
        if any(isinstance(x, bool) for x in raw_xs + raw_ys):
            raise ValueError("boolean is not a physical sample")
        xs = [float(x) for x in raw_xs]
        ys = [float(y) for y in raw_ys]
    except (TypeError, ValueError, OverflowError) as exc:
        return {"eligible_for_fwhm_estimation": False,
                "failures": [f"invalid numeric spectrum: {exc}"],
                "physics_qualified": False, "publication_ready": False,
                "fwhm_verified": False}
    if len(xs) != len(ys) or len(xs) < min_samples:
        failures.append("insufficient or unequal spectral samples")
    if (not xs or any(not math.isfinite(x) or x <= 0 for x in xs)
            or any(not math.isfinite(y) or y < 0 for y in ys)):
        failures.append("nonfinite/nonphysical spectrum")
    if any(a >= b for a, b in zip(xs, xs[1:])):
        failures.append("wavelength grid not strictly increasing")
    if len(xs) >= 3 and all(math.isfinite(x) for x in xs):
        gaps = [b-a for a, b in zip(xs, xs[1:])]
        if all(g > 0 for g in gaps):
            ordered = sorted(gaps)
            median_gap = (ordered[(len(ordered)-1)//2] + ordered[len(ordered)//2]) / 2
            if max(gaps) > max_gap_ratio * median_gap:
                failures.append("unresolved wavelength sampling gap")
    if failures:
        return {"eligible_for_fwhm_estimation": False, "failures": failures,
                "physics_qualified": False, "publication_ready": False,
                "fwhm_verified": False}
    peak = max(range(len(ys)), key=ys.__getitem__)
    baseline = min(ys)
    height = ys[peak] - baseline
    if not (0 < peak < len(ys) - 1) or height <= 0:
        failures.append("no resolved interior peak above baseline")
    else:
        half = baseline + height * 0.5
        left = [i for i in range(peak) if ys[i] <= half < ys[i + 1]]
        right = [i for i in range(peak, len(ys)-1) if ys[i] > half >= ys[i+1]]
        if not left or not right:
            failures.append("unbracketed baseline-relative half-maximum")
        elif right[0] - left[-1] < min_halfmax_intervals:
            failures.append("insufficient sampling across half-maximum width")
        # Multiple crossings indicate an unresolved or multimodal spectral branch.
        if len(left) != 1 or len(right) != 1:
            failures.append("multiple half-maximum crossings; ambiguous resonance")
        if sum(y == ys[peak] for y in ys) > 1:
            failures.append("ambiguous peak maximum")
    return {"eligible_for_fwhm_estimation": not failures, "failures": failures,
            "samples": len(xs), "peak_wavelength_um_diagnostic": xs[peak],
            "physics_qualified": False, "publication_ready": False,
            "fwhm_verified": False, "resonance_verified": False}
