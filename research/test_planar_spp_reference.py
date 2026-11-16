import math
import unittest
from planar_spp_reference import planar_tm_spp

class PlanarSPPTests(unittest.TestCase):
    def test_dispersion_and_passivity(self):
        # Constructed permittivity: software regression only, NOT measured Au.
        p = planar_tm_spp(800, -24+1.5j, 1.77)
        self.assertAlmostEqual(p.neff**2 * (p.epsilon_m+p.epsilon_d), p.epsilon_m*p.epsilon_d)
        self.assertGreater(p.beta_per_m.imag, 0)
        self.assertGreater(p.propagation_length_um, 0)
        self.assertGreater(p.decay_metal_nm, 0)
        self.assertGreater(p.decay_dielectric_nm, 0)
    def test_wavelength_scaling_fixed_permittivity(self):
        a=planar_tm_spp(800,-24+1.5j,1.77)
        b=planar_tm_spp(1600,-24+1.5j,1.77)
        self.assertAlmostEqual(b.propagation_length_um/a.propagation_length_um,2)
        self.assertAlmostEqual(b.decay_dielectric_nm/a.decay_dielectric_nm,2)
    def test_metal_dissipation_monotonic_small_change(self):
        a=planar_tm_spp(800,-24+0.5j,1.77)
        b=planar_tm_spp(800,-24+1j,1.77)
        self.assertLess(b.propagation_length_um,a.propagation_length_um)
    def test_reject_invalid(self):
        for args in [(0,-24+1j,1.7),(800,-24-1j,1.7),(800,-1+1j,1.7),(800,-24+1j,0),(800,complex(float('nan'),1),1.7),(True,-24+1j,1.7)]:
            with self.subTest(args=args), self.assertRaises(ValueError): planar_tm_spp(*args)
    def test_tm_boundary_relation(self):
        p=planar_tm_spp(800,-24+1.5j,1.77)
        k0=2*math.pi/800e-9
        import cmath
        km=cmath.sqrt(p.beta_per_m**2-k0**2*p.epsilon_m)
        kd=cmath.sqrt(p.beta_per_m**2-k0**2*p.epsilon_d)
        self.assertLess(abs(km/p.epsilon_m+kd/p.epsilon_d),1e-7)
if __name__=='__main__': unittest.main()
