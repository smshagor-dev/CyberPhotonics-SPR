import unittest
from scripts.native_spectral_refinement_audit import peak_metrics, compare_refinement

def spectrum(n):
    x=[i/n for i in range(-4*n,4*n+1)]
    return x,[1/(1+v*v) for v in x]

class SpectralRefinementTests(unittest.TestCase):
    def test_nested_lorentzian(self):
        a,b=spectrum(4); c,d=spectrum(16)
        result=compare_refinement(a,b,c,d)
        self.assertTrue(result["spectral_grid_converged"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_same_grid_rejected(self):
        x,y=spectrum(8)
        with self.assertRaises(ValueError): compare_refinement(x,y,x,y)

    def test_non_nested_grid_rejected(self):
        x,y=spectrum(4); xx,yy=spectrum(16); xx[4]+=0.0001
        with self.assertRaises(ValueError): compare_refinement(x,y,xx,yy)

    def test_nonfinite_rejected(self):
        x,y=spectrum(4); y[2]=float("nan")
        with self.assertRaises(ValueError): peak_metrics(x,y)

    def test_monotonic_rejected(self):
        with self.assertRaises(ValueError): peak_metrics(list(range(7)),[7,6,5,4,3,2,1])

    def test_multiple_peaks_rejected(self):
        with self.assertRaises(ValueError): peak_metrics(list(range(9)),[0,1,2,1,0,1,2,1,0])

if __name__=="__main__":
    unittest.main()
