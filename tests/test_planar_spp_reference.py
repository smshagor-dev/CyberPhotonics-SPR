"""Software-only analytic reference tests; illustrative epsilon is NOT measured Au."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from sprpcf.simulation.planar_spp_reference import planar_tm_spp, write_reference_json

class TestPlanarSPP(unittest.TestCase):
    def setUp(self):
        self.kw = dict(wavelength_um=0.8, analyte_ri=1.35,
                       epsilon_m_real=-24., epsilon_m_imag=1.5,
                       source_id="illustrative-fixture-not-Au-data")

    def test_attenuation_decay_and_units(self):
        r = planar_tm_spp(**self.kw)
        self.assertGreater(r.beta_imag_per_um, 0)
        self.assertGreater(r.metal_decay_real_per_um, 0)
        self.assertGreater(r.dielectric_decay_real_per_um, 0)
        self.assertAlmostEqual(r.power_loss_db_per_cm,
                               20/math.log(10)*1e4*r.beta_imag_per_um)
        self.assertFalse(r.publication_ready)
        self.assertFalse(r.pcf_full_vector_verified)
        self.assertFalse(r.comsol_verified)

    def test_passive_loss_limit(self):
        a = planar_tm_spp(**{**self.kw, "epsilon_m_imag": 0.1})
        b = planar_tm_spp(**{**self.kw, "epsilon_m_imag": 0.01})
        self.assertLess(b.beta_imag_per_um, a.beta_imag_per_um)

    def test_reject_invalid_inputs(self):
        for change in ({"epsilon_m_imag": -1}, {"epsilon_m_imag": 0},
                       {"epsilon_m_real": -1}, {"wavelength_um": 0},
                       {"analyte_ri": float("nan")}, {"epsilon_m_real": float("inf")},
                       {"source_id": ""}, {"wavelength_um": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                planar_tm_spp(**{**self.kw, **change})

    def test_verified_material_not_verified_physics(self):
        r = planar_tm_spp(**{**self.kw, "source_independently_verified": True})
        self.assertTrue(r.source_independently_verified)
        self.assertFalse(r.publication_ready)
        self.assertFalse(r.pcf_full_vector_verified)

    def test_immutable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"reference.json"
            write_reference_json(path, planar_tm_spp(**self.kw))
            self.assertEqual(json.loads(path.read_text())["source_id"], self.kw["source_id"])
            with self.assertRaises(FileExistsError):
                write_reference_json(path, planar_tm_spp(**self.kw))
