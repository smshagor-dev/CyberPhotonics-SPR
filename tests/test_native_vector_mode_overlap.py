"""Synthetic unit tests only; not native Maxwell validation."""
import unittest
import numpy as np
from scripts.native_vector_mode_overlap import overlap,match_modes

class ModeOverlapTests(unittest.TestCase):
    def setUp(self):
        self.a=np.eye(4,3,dtype=complex)
        self.b=np.roll(self.a,1,axis=0)
        self.w=np.ones(4)
    def test_identity(self):
        self.assertAlmostEqual(overlap(self.a,self.a,self.a,self.a,self.w),1)
    def test_phase(self):
        self.assertAlmostEqual(overlap(self.a,self.a,1j*self.a,-1j*self.a,self.w),1)
    def test_orthogonal(self):
        a=np.array([[1,0,0],[0,0,0]],complex)
        b=np.array([[0,0,0],[1,0,0]],complex)
        self.assertAlmostEqual(overlap(a,a,b,b,np.ones(2)),0)
    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError): overlap(self.a,self.a,0*self.a,self.a,self.w)
    def test_bad_weight_rejected(self):
        with self.assertRaises(ValueError): overlap(self.a,self.a,self.a,self.a,[1,1,float('nan'),1])
    def test_mesh_mismatch_rejected(self):
        with self.assertRaises(ValueError): overlap(self.a,self.a,self.a[:2],self.a,self.w)
    def test_assignment(self):
        r=[(self.a,self.a),(self.b,self.b)]
        out=match_modes(r,r[::-1],self.w)
        self.assertEqual(out["assignment"],[1,0])
        self.assertFalse(out["physics_qualified"])
    def test_ambiguous_rejected(self):
        r=[(self.a,self.a),(self.a,self.a)]
        with self.assertRaises(ValueError): match_modes(r,r,self.w)
if __name__=="__main__": unittest.main()
