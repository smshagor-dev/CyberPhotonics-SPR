"""Software-only analytical controls for planar TM transfer-matrix reference."""
import math
import unittest
from research.planar_tm_film_reference import tm_reflectance


class TestPlanarTMFilmReference(unittest.TestCase):
    def test_identical_media(self):
        self.assertAlmostEqual(tm_reflectance(.8,.4,2.25,2.25,[]),0,places=12)

    def test_fresnel_normal(self):
        self.assertAlmostEqual(tm_reflectance(.8,0,1,2.25,[]),.04,places=12)

    def test_quarter_wave_antireflection(self):
        n=math.sqrt(1.5)
        self.assertLess(tm_reflectance(.8,0,1,2.25,[(n*n,.8/(4*n))]),1e-12)

    def test_passive_illustrative_metal(self):
        # Constructed epsilon, NOT measured gold or sensor evidence.
        r=tm_reflectance(.8,.7,2.25,1.7689,[(-20+1.5j,.05)])
        self.assertGreaterEqual(r,0)
        self.assertLessEqual(r,1)

    def test_fail_closed(self):
        cases=[(0,0,1,1,[]),(.8,0,1,1,[(-20-1j,.05)]),
               (.8,0,1,1,[(2,0)]),(.8,0,1,1,[(2,True)])]
        for args in cases:
            with self.subTest(args=args),self.assertRaises(ValueError):
                tm_reflectance(*args)


if __name__=="__main__":
    unittest.main()
