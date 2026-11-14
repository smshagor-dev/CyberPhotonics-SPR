from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint, save_tandem_checkpoint
from sprpcf.ml.dataset import DesignDataModule, GEOMETRY_COLUMNS
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator
from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.synthetic import build_synthetic_dataset
from sprpcf.validation.inverse_benchmark import run_inverse_benchmark


@pytest.fixture(scope="module")
def synthetic_inputs(tmp_path_factory):
    """Controlled, intentionally RANDOM-WEIGHT test fixture; not research evidence."""
    root = tmp_path_factory.mktemp("inverse-fixture")
    data_path = root / "synthetic.csv"
    write_dataset(
        build_synthetic_dataset(samples=20, wavelengths=48, seed=11),
        data_path, metadata={"source": "synthetic", "fixture_only": True},
    )
    module = DesignDataModule(data_path, batch_size=32, seed=7)
    module.setup()
    checkpoint_path = root / "fixture.pt"
    torch_checkpoint = {
        "forward_state_dict": ForwardNetwork().state_dict(),
        "inverse_state_dict": InverseGenerator().state_dict(),
        "geometry_mean": module.geometry_scaler.mean_,
        "geometry_scale": module.geometry_scaler.scale_,
        "condition_mean": module.condition_scaler.mean_,
        "condition_scale": module.condition_scaler.scale_,
        "metric_mean": module.metric_scaler.mean_,
        "metric_scale": module.metric_scaler.scale_,
        "seed": 7,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_row_indices": module.train_indices.tolist(),
        "validation_row_indices": module.val_indices.tolist(),
        "train_geometry_ids": sorted({format(int(module.groups[i]), "016x") for i in module.train_indices}),
        "validation_geometry_ids": sorted({
            format(int(module.groups[i]), "016x") for i in module.val_indices
        }),
        "training_parameters": {"forward_epochs": 1, "inverse_epochs": 1},
    }
    save_tandem_checkpoint(torch_checkpoint, checkpoint_path)
    return data_path, checkpoint_path, root


def _run(synthetic_inputs, out: Path):
    data, checkpoint, _ = synthetic_inputs
    return run_inverse_benchmark(
        data, checkpoint, out, seeds=(7, 17), candidates_per_target=4,
        bootstrap_resamples=200, allow_software_only=True,
    )


def test_heldout_targets_and_equal_surrogate_query_budget(synthetic_inputs, tmp_path: Path):
    report = _run(synthetic_inputs, tmp_path / "run")
    out = tmp_path / "run"
    splits = pd.read_csv(out / "split_membership.csv")
    targets = pd.read_csv(out / "validation_targets.csv")
    candidate = pd.read_csv(out / "all_candidates.csv")
    selected = pd.read_csv(out / "selected_designs.csv")
    requests = pd.read_csv(out / "PHYSICS_VERIFICATION_REQUESTS.csv")
    assert report["scientific_submission_ready"] is False
    assert report["evidence_class"] == "software_only"
    assert report["physics_solver_evaluations"] == 0
    assert report["seeds_are_independent_training_replicates"] is False
    train = set(splits.loc[splits["split"] == "train", "geometry_id"])
    val = set(splits.loc[splits["split"] == "heldout_validation", "geometry_id"])
    assert train.isdisjoint(val)
    assert set(targets["target_group_id"]) == val
    assert len(targets) == 4
    assert len(candidate) == 4 * 4 * 4 * 2  # target groups * methods * budget * generation seeds
    assert len(selected) == len(requests) == 4 * 4 * 2
    assert selected.groupby(["design_seed", "target_id", "method"]).size().eq(1).all()
    assert candidate.groupby(["design_seed", "target_id", "method"]).size().eq(4).all()
    assert requests["verification_status"].eq("not_run").all()
    assert requests["request_id"].nunique() == len(requests)
    for filename, digest in report["outputs_sha256"].items():
        assert hashlib.sha256((out / filename).read_bytes()).hexdigest() == digest
    assert (out / "INVERSE_BENCHMARK_REPORT.md").is_file()


def test_geometry_constraints_zero_latent_and_paired_statistics(synthetic_inputs, tmp_path: Path):
    _run(synthetic_inputs, tmp_path / "run")
    out = tmp_path / "run"
    selected = pd.read_csv(out / "selected_designs.csv")
    paired = pd.read_csv(out / "paired_geometry_bootstrap.csv")
    lower = np.asarray([0.8, 0.20, 15.0, 0.20])
    upper = np.asarray([4.0, 0.90, 80.0, 1.50])
    geom = selected[GEOMETRY_COLUMNS].to_numpy(dtype=float)
    assert np.all(geom >= lower - 1e-8)
    assert np.all(geom <= upper + 1e-8)
    assert selected.loc[selected["method"] == "tandem_zero_latent", "unique_candidate_geometries"].eq(1).all()
    assert paired["paired_independent_target_geometries"].eq(4).all()
    assert paired["interval_status"].eq("descriptive_group_bootstrap").all()
    assert paired["ci_low"].notna().all()
    assert (paired["ci_low"] <= paired["ci_high"]).all()


def test_rejects_implicit_synthetic_claim_and_hash_mismatch(synthetic_inputs, tmp_path: Path):
    data, checkpoint, _ = synthetic_inputs
    with pytest.raises(ValueError, match="allow-software-only"):
        run_inverse_benchmark(data, checkpoint, tmp_path / "refused", seeds=(7, 17))
    bad_data = tmp_path / "tampered.csv"
    bad_data.write_bytes(data.read_bytes() + b"\n")
    bad_sidecar = bad_data.with_name(bad_data.name + ".meta.json")
    bad_sidecar.write_bytes(data.with_name(data.name + ".meta.json").read_bytes())
    with pytest.raises(ValueError, match="SHA-256"):
        run_inverse_benchmark(bad_data, checkpoint, tmp_path / "refused2",
                              seeds=(7, 17), allow_software_only=True)


def test_rejects_bad_checkpoint_split_or_training_metadata(synthetic_inputs, tmp_path: Path):
    data, checkpoint, _ = synthetic_inputs
    payload = load_tandem_checkpoint(checkpoint)
    payload["validation_row_indices"] = payload["train_row_indices"]
    copy = tmp_path / "wrong.pt"
    save_tandem_checkpoint(payload, copy)
    with pytest.raises(ValueError, match="row partitions"):
        run_inverse_benchmark(data, copy, tmp_path / "wrong", seeds=(7, 17),
                              allow_software_only=True)
    payload = load_tandem_checkpoint(checkpoint)
    payload["training_parameters"]["forward_epochs"] = 0
    save_tandem_checkpoint(payload, copy)
    with pytest.raises(ValueError, match="positive training epoch"):
        run_inverse_benchmark(data, copy, tmp_path / "wrong2", seeds=(7, 17),
                              allow_software_only=True)


def test_candidate_seeds_are_reproducible_and_output_is_immutable(synthetic_inputs, tmp_path: Path):
    _run(synthetic_inputs, tmp_path / "first")
    _run(synthetic_inputs, tmp_path / "second")
    first = pd.read_csv(tmp_path / "first" / "selected_designs.csv")
    second = pd.read_csv(tmp_path / "second" / "selected_designs.csv")
    independent = [col for col in first.columns if col != "surrogate_scoring_seconds"]
    pd.testing.assert_frame_equal(first[independent], second[independent])
    with pytest.raises(FileExistsError, match="empty"):
        _run(synthetic_inputs, tmp_path / "first")


def test_cli_generates_software_only_report(synthetic_inputs, tmp_path: Path):
    data, checkpoint, _ = synthetic_inputs
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_inverse_benchmark.py"
    output = tmp_path / "cli"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--data", str(data), "--checkpoint", str(checkpoint), "--out", str(output),
            "--seeds", "7,17", "--candidates-per-target", "4",
            "--bootstrap-resamples", "200", "--allow-software-only",
        ],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads((output / "INVERSE_BENCHMARK_MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["independent_validation_status"] == "not_run"
    assert payload["scientific_submission_ready"] is False
