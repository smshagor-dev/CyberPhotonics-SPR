import unittest

from research.overlap_mode_assignment import assign_modes, AmbiguousModeAssignment


class AssignmentTests(unittest.TestCase):
    def test_permutation(self):
        self.assertEqual(assign_modes([[0.12, 0.97], [0.96, 0.11]]), (1, 0))

    def test_global_not_row_greedy(self):
        self.assertEqual(assign_modes([[0.92, 0.95], [0.90, 0.10]], minimum_overlap=0.85), (1, 0))

    def test_near_tie_fails(self):
        with self.assertRaises(AmbiguousModeAssignment):
            assign_modes([[0.90, 0.89], [0.89, 0.90]], minimum_margin=0.05)

    def test_low_overlap_fails(self):
        with self.assertRaises(AmbiguousModeAssignment):
            assign_modes([[0.70, 0.1], [0.1, 0.99]])

    def test_nan_fails(self):
        with self.assertRaises(ValueError):
            assign_modes([[float("nan")]])

    def test_rectangular_fails(self):
        with self.assertRaises(ValueError):
            assign_modes([[0.9, 0.8]])

    def test_single_mode(self):
        self.assertEqual(assign_modes([[0.99]]), (0,))


if __name__ == "__main__":
    unittest.main()
