"""Constructed spectra test numerical postprocessing, never Maxwell physics."""
import math
import unittest
from research_validation.spectral_peak_width_audit import audit


def spectrum(step):
    n = round(100 / step)
    w = [700 + i * 100 / n for i in range(n + 1)]
    y = [1 + 8 / (1 + ((x - 746.3) / 9) ** 2) for x in w]
    return w, y


class SpectralPeakWidthAuditTests(unittest.TestCase):
    def test_refinement_never_qualifies_physics(self):
        c, cy = spectrum(2)
        f, fy = spectrum(0.5)
        result = audit(c, cy, f, fy,
                       max_resonance_drift_nm=2, max_fwhm_drift_nm=2)
        self.assertTrue(result.passed)
        self.assertFalse(result.physics_qualified)
        self.assertFalse(result.publication_ready)

    def test_identical_grid_rejected(self):
        w, y = spectrum(1)
        with self.assertRaises(ValueError):
            audit(w, y, w, y, max_resonance_drift_nm=1, max_fwhm_drift_nm=1)

    def test_different_domain_rejected(self):
        c, cy = spectrum(2)
        f, fy = spectrum(1)
        f[0] += 1
        with self.assertRaises(ValueError):
            audit(c, cy, f, fy, max_resonance_drift_nm=1, max_fwhm_drift_nm=1)

    def test_nonfinite_rejected(self):
        c, cy = spectrum(2)
        f, fy = spectrum(1)
        fy[12] = math.nan
        with self.assertRaises(ValueError):
            audit(c, cy, f, fy, max_resonance_drift_nm=1, max_fwhm_drift_nm=1)

    def test_flat_spectrum_rejected(self):
        c, cy = spectrum(2)
        f, fy = spectrum(1)
        fy = [1] * len(fy)
        with self.assertRaises(ValueError):
            audit(c, cy, f, fy, max_resonance_drift_nm=1, max_fwhm_drift_nm=1)


if __name__ == "__main__":
    unittest.main()
