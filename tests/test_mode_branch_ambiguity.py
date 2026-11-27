import unittest
from sprpcf.simulation.mode_branch_ambiguity import audit_overlap_matrix


class TestOverlapAmbiguity(unittest.TestCase):
    def test_swapped_modes(self):
        result = audit_overlap_matrix([[0.05, 0.96], [0.94, 0.04]])
        self.assertEqual([r["current_mode_index"] for r in result["assignments"]], [1, 0])
        self.assertTrue(result["all_diagnostically_unambiguous"])
        self.assertFalse(result["physics_qualified"])

    def test_degenerate_modes(self):
        self.assertFalse(audit_overlap_matrix([[0.89, 0.87], [0.88, 0.90]])["all_diagnostically_unambiguous"])

    def test_low_overlap(self):
        self.assertFalse(audit_overlap_matrix([[0.60, 0.10], [0.10, 0.70]])["all_diagnostically_unambiguous"])

    def test_invalid_matrices(self):
        for matrix in ([[1.2]], [[float("nan")]], [[0.2, 0.3]], [[True]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                audit_overlap_matrix(matrix)

    def test_single_mode_is_not_physical_identity(self):
        result = audit_overlap_matrix([[0.99]])
        self.assertTrue(result["all_diagnostically_unambiguous"])
        self.assertFalse(result["assignments"][0]["core_spp_identity_verified"])


if __name__ == "__main__":
    unittest.main()
