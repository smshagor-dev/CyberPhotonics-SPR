import unittest
from planar_tm_root_crosscheck import solve_planar_tm

class TestPlanarRoot(unittest.TestCase):
    def test_passive_root(self):
        r=solve_planar_tm(800,-24+1.5j,1.33**2)
        self.assertTrue(r["pass_check"],r)
        self.assertFalse(r["physics_qualified"])
        self.assertFalse(r["publication_ready"])
    def test_scale_constant_permittivity(self):
        a=solve_planar_tm(800,-24+1.5j,1.33**2)
        b=solve_planar_tm(1000,-24+1.5j,1.33**2)
        for x,y in zip(a["numerical_neff"],b["numerical_neff"]):
            self.assertAlmostEqual(x,y,places=9)
    def test_gain_rejected(self):
        with self.assertRaises(ValueError):
            solve_planar_tm(800,-24-1j,1.33**2)
    def test_zero_wavelength_rejected(self):
        with self.assertRaises(ValueError):
            solve_planar_tm(0,-24+1j,1.33**2)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            solve_planar_tm(800,complex(float("nan"),1),1.33**2)

if __name__=="__main__":
    unittest.main()
