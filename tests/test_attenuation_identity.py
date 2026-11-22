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


    def test_overflow_and_malformed_rows_are_accounted_for(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            loss_db_per_cm(1e-300, 1e300)
        good = {'wavelength_um': 0.8, 'n_eff_imag': 0.001,
                'raw_attenuation_db_per_cm_unqualified': loss_db_per_cm(0.8, 0.001)}
        bad = [None, [], {'wavelength_um': True, 'n_eff_imag': 0.001,
                            'raw_attenuation_db_per_cm_unqualified': 1},
               {'wavelength_um': 1e-300, 'n_eff_imag': 1e300,
                'raw_attenuation_db_per_cm_unqualified': 0}]
        result = audit_attenuation_rows([good, *bad])
        self.assertEqual(result['rows_checked'], 5)
        self.assertEqual([failure['row'] for failure in result['failures']], [2, 3, 4, 5])
        self.assertFalse(result['dimensional_identity_pass'])
        self.assertFalse(result['publication_ready'])

    def test_boolean_and_nonfinite_tolerances_rejected(self):
        for kw in ({'relative_tolerance': True}, {'absolute_tolerance': False},
                   {'relative_tolerance': float('nan')},
                   {'absolute_tolerance': float('inf')}):
            with self.assertRaises(ValueError):
                audit_attenuation_rows([], **kw)


if __name__ == '__main__':
    unittest.main()
