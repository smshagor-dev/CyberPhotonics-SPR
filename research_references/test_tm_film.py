import unittest
from research_references.tm_film import tm_reflection, reflectance


class FilmTests(unittest.TestCase):
    def test_zero_thickness_reduces_to_direct_interface(self):
        a = tm_reflection(800, 0, 2.25, -20+1.5j, 1.77, 40)
        b = tm_reflection(800, 0, 2.25, 2.25, 1.77, 40)
        self.assertAlmostEqual(abs(a-b), 0, places=12)

    def test_identical_media_no_reflection(self):
        self.assertAlmostEqual(reflectance(800, 47, 2.25, 2.25, 2.25, 20), 0, places=12)

    def test_passive_film_bounded_reflectance(self):
        for angle in (0, 25, 45, 65, 80):
            self.assertLessEqual(reflectance(800, 50, 2.25, -20+1.5j, 1.77, angle), 1+1e-12)

    def test_reject_gain(self):
        with self.assertRaises(ValueError):
            reflectance(800, 50, 2.25, -20-1j, 1.77, 45)

    def test_reject_invalid(self):
        for w, t, a in ((0, 50, 30), (800, -1, 30), (800, 50, 90), (float('nan'), 50, 30)):
            with self.assertRaises(ValueError):
                reflectance(w, t, 2.25, -20+1j, 1.77, a)


if __name__ == '__main__':
    unittest.main()
