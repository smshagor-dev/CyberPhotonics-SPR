"""Constructed-input software regressions; no physical qualification."""
import unittest
import numpy as np
from scripts.planar_three_layer_tm import dispersion, solve

class FilmTMTests(unittest.TestCase):
    def test_symmetric_film(self):
        eps=1.77
        metal=-24+1.5j
        guess=np.sqrt(eps*metal/(eps+metal))
        result=solve(.8,.05,eps,metal,eps,guess)
        self.assertLess(result["residual"],1e-8)
        self.assertGreater(result["attenuation_db_cm"],0)
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_invalid_geometry(self):
        with self.assertRaises(ValueError):
            dispersion(1.4+.01j,.8,0,1.77,-24+1.5j,1.77)

    def test_thick_film_limit(self):
        eps=1.77
        metal=-24+1.5j
        neff=np.sqrt(eps*metal/(eps+metal))
        self.assertLess(abs(dispersion(neff,.8,2.,eps,metal,eps)),1e-8)

if __name__=="__main__":
    unittest.main()
