from __future__ import annotations

import hashlib
from pathlib import Path

from sprpcf.utils.reproducibility import artifact_metadata, create_reproducibility_bundle, sha256_file
from sprpcf.utils.release import _read_package_version, _read_project_version


def test_sha256_and_artifact_metadata_are_portable(tmp_path: Path) -> None:
    artifact = tmp_path / "data.bin"
    artifact.write_bytes(b"abc")
    expected = hashlib.sha256(b"abc").hexdigest()

    assert sha256_file(artifact) == expected
    row = artifact_metadata(artifact, "dataset", repo_root=tmp_path)
    assert row == {
        "role": "dataset",
        "path": "data.bin",
        "size_bytes": 3,
        "sha256": expected,
    }


def test_reproducibility_bundle_contains_hashes_and_environment_lock(tmp_path: Path) -> None:
    artifact = tmp_path / "checkpoint.pt"
    artifact.write_bytes(b"model-bytes")
    output = tmp_path / "bundle"

    manifest = create_reproducibility_bundle(
        output,
        experiment_name="unit-test",
        seed=17,
        artifacts=[("checkpoint", artifact)],
        config={"epochs": 1},
        repo_root=tmp_path,
    )

    assert manifest["schema_version"] == "1.0"
    assert manifest["experiment"]["seed"] == 17
    assert manifest["artifacts"][0]["sha256"] == sha256_file(artifact)
    assert (output / "manifest.json").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "environment.lock.txt").is_file()
    assert (output / "checksums.sha256").is_file()
    assert (output / "REPRODUCE.md").is_file()


def test_release_version_parsers_are_section_scoped() -> None:
    pyproject = '[build-system]\nrequires=["x"]\n\n[project]\nname="sprpcf"\nversion = "1.2.3"\n\n[tool.demo]\nversion="9"\n'
    init_text = '__version__ = "1.2.3"\n'
    assert _read_project_version(pyproject) == "1.2.3"
    assert _read_package_version(init_text) == "1.2.3"
