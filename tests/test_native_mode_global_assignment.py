"""Synthetic adversarial assignment tests, not physical mode evidence."""
import unittest
import numpy as np
from scripts.native_vector_mode_overlap import match_modes

class GlobalAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.w=np.ones(3)
        self.a=np.eye(3,dtype=complex)
        self.b=np.roll(self.a,1,axis=0)

    def test_global_tie_rejected(self):
        # Both rows have a confident top candidate, but prefer the SAME candidate.
        # The two global permutations tie; a row-wise gap test alone misses it.
        ref=[(self.a,self.a),(self.a,self.a)]
        cand=[(self.a,self.a),(self.b,self.b)]
        with self.assertRaises(ValueError):
            match_modes(ref,cand,self.w,minimum=0,margin=.05)

    def test_unique_swapped_assignment(self):
        ref=[(self.a,self.a),(self.b,self.b)]
        out=match_modes(ref,ref[::-1],self.w)
        self.assertEqual(out["assignment"],[1,0])
        self.assertFalse(out["physics_qualified"])
        self.assertFalse(out["publication_ready"])
        self.assertFalse(out["comsol_verified"])

if __name__=="__main__": unittest.main()
