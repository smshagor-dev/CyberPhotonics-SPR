"""Constructed manifests test software safeguards, never Maxwell physics."""
import copy

from scripts.validate_solver_training_split import validate


def sample():
    return {"records": [
        dict(split=split, geometry_sha256=char * 64,
             solver_artifact_sha256=str(i + 1) * 64,
             solver_source="native_femwell", physics_qualified=True,
             source_verified=True,
             evidence_registry={key: "d" * 64 for key in
                 ("topology_sha256", "material_dataset_sha256", "mesh_sha256",
                  "solver_config_sha256", "raw_field_archive_sha256")},
             wavelength_um=0.8, analyte_ri=1.36, loss_db_per_cm=0.25)
        for i, (split, char) in enumerate(zip(
            ("train", "validation", "test"), "abc"))
    ]}


def test_constructed_qualified_manifest_does_not_certify_physics():
    result = validate(sample())
    assert result["valid"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


def test_geometry_leakage_rejected():
    data = sample()
    data["records"][1]["geometry_sha256"] = "a" * 64
    assert not validate(data)["valid"]


def test_unqualified_solver_rejected():
    data = sample()
    data["records"][0]["physics_qualified"] = False
    assert not validate(data)["valid"]


def test_synthetic_source_rejected():
    data = sample()
    data["records"][0]["solver_source"] = "synthetic"
    assert not validate(data)["valid"]


def test_duplicate_observation_rejected():
    data = sample()
    data["records"].append(copy.deepcopy(data["records"][0]))
    assert not validate(data)["valid"]


def test_nonfinite_loss_rejected():
    data = sample()
    data["records"][0]["loss_db_per_cm"] = float("nan")
    assert not validate(data)["valid"]


def test_missing_test_split_rejected():
    data = sample()
    data["records"] = data["records"][:2]
    assert not validate(data)["valid"]


def test_unsafe_claim_rejected():
    data = sample()
    data["publication_ready"] = True
    assert not validate(data)["valid"]


def test_missing_native_evidence_registry_rejected():
    data = sample()
    del data["records"][0]["evidence_registry"]
    assert not validate(data)["valid"]
