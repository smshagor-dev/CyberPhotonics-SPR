"""Constructed illustrative control tests, not photonic sensor validation."""
import unittest
from scripts.film_thickness_continuation import campaign

class PlanarFilmContinuationTests(unittest.TestCase):
    def test_large_film_limit(self):
        result = campaign()
        self.assertEqual((result['converged'], result['failed']), (7, 0))
        self.assertLess(result['rows'][-1]['distance_to_single_interface'], 1e-7)
        self.assertGreater(result['rows'][0]['distance_to_single_interface'], 0.01)
        self.assertFalse(result['physics_qualified'])
        self.assertFalse(result['publication_ready'])
        self.assertFalse(result['comsol_verified'])

    def test_invalid_thickness(self):
        for grid in ([.1,.1],[.2,.1],[-.1],[]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                campaign(grid)

    def test_active_metal_rejected(self):
        with self.assertRaises(ValueError):
            campaign(metal=-24-1j)

    def test_thick_limit_monotonic(self):
        distances = [row['distance_to_single_interface'] for row in campaign()['rows']]
        self.assertTrue(all(b < a for a,b in zip(distances,distances[1:])))

if __name__ == '__main__':
    unittest.main()
