import unittest
import numpy as np
from research.native_field_correspondence import (
    normalized_field_overlap, correspondence_matrix, assign_correspondence,
    AmbiguousCorrespondence,
)


class FieldCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([[1,0,0],[0,1,0]],complex)
        self.h = np.array([[0,1,0],[1,0,0]],complex)
        self.w = np.array([0.5,1.5])

    def test_phase_invariance(self):
        phase = np.exp(1j*1.37)
        self.assertAlmostEqual(normalized_field_overlap(self.e,self.h,self.e*phase,self.h*phase,self.w),1)

    def test_orthogonal_modes(self):
        e = np.array([[0,0,1],[0,0,1]],complex)
        self.assertAlmostEqual(normalized_field_overlap(self.e,self.h,e,np.zeros_like(e),self.w),0)

    def test_permuted_fields(self):
        a = (self.e,self.h)
        b = (np.array([[0,0,1],[0,0,1]],complex),np.zeros_like(self.h))
        self.assertEqual(assign_correspondence(correspondence_matrix([a,b],[b,a],self.w)),(1,0))

    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError):
            normalized_field_overlap(self.e,self.h,np.zeros_like(self.e),np.zeros_like(self.h),self.w)

    def test_mesh_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            normalized_field_overlap(self.e,self.h,self.e[:1],self.h[:1],self.w)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            normalized_field_overlap(self.e,self.h,self.e,self.h,[float('nan'),1])

    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):
            normalized_field_overlap(self.e,self.h,self.e,self.h,self.w,magnetic_scale=0)

    def test_ambiguous_assignment_rejected(self):
        with self.assertRaises(AmbiguousCorrespondence):
            assign_correspondence([[0.9,0.89],[0.89,0.9]])

    def test_weak_assignment_rejected(self):
        with self.assertRaises(AmbiguousCorrespondence):
            assign_correspondence([[0.5,0.1],[0.1,0.9]])

    def test_no_physics_qualification(self):
        self.assertFalse(hasattr(normalized_field_overlap,'publication_qualified'))


if __name__ == '__main__':
    unittest.main()
