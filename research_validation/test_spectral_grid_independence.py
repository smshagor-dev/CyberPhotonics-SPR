import unittest
from research_validation.spectral_grid_independence import validate_spectral_grids

def grid(n, lo=.7, hi=.9):
    return [lo+(hi-lo)*i/n for i in range(n+1)]

class SpectralGridTests(unittest.TestCase):
    def test_refined_grid_software_only(self):
        r=validate_spectral_grids(grid(100),grid(200))
        self.assertTrue(r["grid_refinement_verified"])
        self.assertFalse(r["physics_qualified"])
        self.assertFalse(r["publication_ready"])
    def test_identical_grid_rejected(self):
        with self.assertRaisesRegex(ValueError,"not finer"):
            validate_spectral_grids(grid(100),grid(100))
    def test_reversed_grid_rejected(self):
        with self.assertRaises(ValueError):
            validate_spectral_grids(grid(200),grid(100))
    def test_domain_shift_rejected(self):
        with self.assertRaisesRegex(ValueError,"domain mismatch"):
            validate_spectral_grids(grid(100),grid(200,.701,.9))
    def test_insufficient_refinement_rejected(self):
        with self.assertRaisesRegex(ValueError,"insufficient"):
            validate_spectral_grids(grid(100),grid(120))
    def test_duplicate_wavelength_rejected(self):
        x=grid(100); x[20]=x[19]
        with self.assertRaisesRegex(ValueError,"nonmonotonic"):
            validate_spectral_grids(x,grid(200))
    def test_nan_rejected(self):
        x=grid(100); x[5]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            validate_spectral_grids(x,grid(200))
    def test_threshold_validation(self):
        with self.assertRaisesRegex(ValueError,"threshold"):
            validate_spectral_grids(grid(100),grid(200),min_refinement_ratio=1)
if __name__=="__main__": unittest.main()
