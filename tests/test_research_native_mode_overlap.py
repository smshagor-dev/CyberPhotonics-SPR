import unittest
import numpy as np
from research_native_mode_overlap import power_overlap, attenuation_db_per_cm

class NativeModeOverlapTests(unittest.TestCase):
    def setUp(self):
        self.e=np.array([[1+0j,0j],[2+0j,0j]])
        self.h=np.array([[0j,1+0j],[0j,2+0j]])
        self.w=np.ones(2)
    def test_self(self):
        self.assertAlmostEqual(power_overlap(self.e,self.h,self.e,self.h,self.w),1)
    def test_phase(self):
        self.assertAlmostEqual(power_overlap(self.e,self.h,1j*self.e,1j*self.h,self.w),1)
    def test_zero_flux(self):
        with self.assertRaises(ValueError): power_overlap(self.e,self.h,self.e,np.zeros_like(self.h),self.w)
    def test_invalid_mesh(self):
        with self.assertRaises(ValueError): power_overlap(self.e,self.h,self.e,self.h,[1])
    def test_nonfinite(self):
        e=self.e.copy();e[0,0]=np.nan
        with self.assertRaises(ValueError): power_overlap(e,self.h,self.e,self.h,self.w)
    def test_attenuation_units(self):
        self.assertAlmostEqual(attenuation_db_per_cm(1.4+1e-5j,1),8.686*2*np.pi*0.1)
    def test_gain_rejected(self):
        with self.assertRaises(ValueError): attenuation_db_per_cm(1.4-1e-5j,1)
