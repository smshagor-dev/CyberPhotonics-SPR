"""Metadata gate adversarial regression tests; not physical validation."""
import unittest
from research.source_specific_physics_gate import REQUIRED, evaluate

def src(sid, kind):
    return {"id": sid, "kind": kind, "sha256": "a" * 64,
            "uri": "artifact://" + sid, "verified": True, "failed_runs": 0}

class SourceGateTests(unittest.TestCase):
    def test_empty(self):
        r = evaluate({"sources": [], "gates": {}})
        self.assertFalse(r["physics_qualified"])
        self.assertFalse(r["source_manifest_complete"])
    def test_complete_manifest_still_unqualified(self):
        r = evaluate({"sources": [src(k, v) for k, v in REQUIRED.items()],
                      "gates": {k: [k] for k in REQUIRED}})
        self.assertTrue(r["source_manifest_complete"])
        self.assertFalse(r["physics_qualified"])
        self.assertFalse(r["publication_ready"])
        self.assertFalse(r["comsol_verified"])
    def test_wrong_source_type(self):
        r = evaluate({"sources": [src("a", "native_maxwell")],
                      "gates": {"measured_gold": ["a"]}})
        self.assertFalse(r["gates"]["measured_gold"])
    def test_fixture(self):
        r = evaluate({"sources": [src("a", "fixture")],
                      "gates": {"full_vector_modes": ["a"]}})
        self.assertTrue(r["errors"])
    def test_duplicate(self):
        r = evaluate({"sources": [src("a", "native_maxwell"),
                                   src("a", "native_maxwell")]})
        self.assertTrue(r["errors"])
    def test_invalid_hash(self):
        s = src("a", "native_maxwell")
        s["sha256"] = "bad"
        self.assertTrue(evaluate({"sources": [s]})["errors"])
    def test_failed_run(self):
        s = src("a", "native_maxwell")
        s["failed_runs"] = 1
        self.assertTrue(evaluate({"sources": [s]})["errors"])
    def test_unverified(self):
        s = src("a", "native_maxwell")
        s["verified"] = False
        self.assertTrue(evaluate({"sources": [s]})["errors"])
    def test_missing_sources(self):
        with self.assertRaises(ValueError):
            evaluate({})

if __name__ == "__main__":
    unittest.main()
