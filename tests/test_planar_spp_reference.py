"""Physics identity and fail-closed checks for independent planar SPP reference."""
import math
import unittest
from pathlib import Path
import importlib.util

MODULE = Path(__file__).resolve().parents[1] / "research" / "planar_spp_reference.py"
spec = importlib.util.spec_from_file_location("planar_spp_reference", MODULE)
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class PlanarSPPTests(unittest.TestCase):
    def test_complex_dispersion_identity(self):
        r = module.planar_tm_spp(0.8, 1.36, -24 + 1.5j)
        expected = (-24 + 1.5j) * 1.36**2 / (-24 + 1.5j + 1.36**2)
        self.assertAlmostEqual((r.n_eff**2).real, expected.real, places=12)
        self.assertAlmostEqual((r.n_eff**2).imag, expected.imag, places=12)
        self.assertGreater(r.beta_per_um.imag, 0)
        self.assertFalse(r.publication_qualified)

    def test_attenuation_consistency(self):
        r = module.planar_tm_spp(0.8, 1.36, -24 + 1.5j)
        self.assertAlmostEqual(r.intensity_length_um * 2 * r.beta_per_um.imag, 1)
        self.assertAlmostEqual(r.attenuation_db_per_cm, 1e4 * 20 / math.log(10) * r.beta_per_um.imag)

    def test_reject_invalid_inputs(self):
        for args in [(0,1.36,-24+1j), (float("nan"),1.36,-24+1j),
                     (0.8,0,-24+1j), (0.8,1.36,-24-1j),
                     (0.8,1.36,-1+1j), (0.8,1.36,complex(float("nan"),1))]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                module.planar_tm_spp(*args)


if __name__ == "__main__":
    unittest.main()
