import math
import unittest
from sprpcf.simulation.attenuation_identity import loss_db_per_cm, audit_attenuation_rows


class AttenuationIdentityTests(unittest.TestCase):
    def test_direct_power_ratio_independent_formula(self):
        wavelength, imaginary, distance_cm = 0.8, 0.001, 0.002
        beta_imag_cm = 2 * math.pi * imaginary / (wavelength * 1e-4)
        power_ratio = math.exp(-2 * beta_imag_cm * distance_cm)
        independently_derived = -10 * math.log10(power_ratio) / distance_cm
        self.assertAlmostEqual(loss_db_per_cm(wavelength, imaginary), independently_derived, places=9)

    def test_zero_loss_and_passivity(self):
        self.assertEqual(loss_db_per_cm(0.8, 0), 0)
        for wl, im in [(0, 0.01), (-1, 0.01), (0.8, -0.1),
                       (float('nan'), 0), (0.8, float('inf'))]:
            with self.assertRaises(ValueError):
                loss_db_per_cm(wl, im)

    def test_false_loss_and_missing_rows_fail_closed(self):
        row = {'wavelength_um': '0.8', 'n_eff_imag': '0.001',
               'raw_attenuation_db_per_cm_unqualified': str(loss_db_per_cm(0.8, 0.001))}
        self.assertTrue(audit_attenuation_rows([row])['dimensional_identity_pass'])
        wrong = dict(row, raw_attenuation_db_per_cm_unqualified='100')
        self.assertFalse(audit_attenuation_rows([row, wrong])['dimensional_identity_pass'])
        self.assertEqual(audit_attenuation_rows([row, wrong])['failures'][0]['row'], 2)
        self.assertFalse(audit_attenuation_rows([])['dimensional_identity_pass'])
        self.assertFalse(audit_attenuation_rows([row])['physics_qualified'])


if __name__ == '__main__':
    unittest.main()
