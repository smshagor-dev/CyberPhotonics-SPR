from __future__ import annotations

import numpy as np
import torch

from sprpcf.ml.dataset import CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS, DesignDataModule
from sprpcf.ml.ensemble import build_forward_ensemble_checkpoint
from sprpcf.ml.losses import GEOMETRY_MAX, GEOMETRY_MIN
from sprpcf.ml.multiobjective import fit_calibration_profile, optimize_target_table, pareto_ranks
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator
from sprpcf.simulation.synthetic import build_synthetic_dataset


def _make_checkpoint(tmp_path):
    data_path = tmp_path / "reference.csv"
    frame = build_synthetic_dataset(samples=8, wavelengths=48, seed=13)
    frame.to_csv(data_path, index=False)
    data = DesignDataModule(data_path, batch_size=16, seed=7)
    data.setup()

    torch.manual_seed(11)
    forward_a = ForwardNetwork()
    torch.manual_seed(17)
    forward_b = ForwardNetwork()
    torch.manual_seed(23)
    inverse = InverseGenerator()
    checkpoint = {
        "forward_state_dict": forward_a.state_dict(),
        "forward_ensemble_state_dicts": [forward_a.state_dict(), forward_b.state_dict()],
        "inverse_state_dict": inverse.state_dict(),
        "geometry_mean": data.geometry_scaler.mean_,
        "geometry_scale": data.geometry_scaler.scale_,
        "condition_mean": data.condition_scaler.mean_,
        "condition_scale": data.condition_scaler.scale_,
        "metric_mean": data.metric_scaler.mean_,
        "metric_scale": data.metric_scaler.scale_,
        "seed": 7,
    }
    checkpoint_path = tmp_path / "tandem.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, data_path, frame


def test_pareto_ranks_separate_dominated_candidates():
    objectives = np.array([[1.0, 1.0], [2.0, 2.0], [0.5, 3.0], [3.0, 0.5]])
    ranks = pareto_ranks(objectives)
    assert ranks.tolist() == [0, 1, 0, 0]


def test_calibration_and_multiobjective_design_are_finite_and_bounded(tmp_path):
    checkpoint_path, data_path, frame = _make_checkpoint(tmp_path)
    profile = fit_calibration_profile(checkpoint_path, data_path, confidence=0.9)
    assert profile.calibration_rows > 0
    assert profile.training_reference_rows > 0
    assert np.all(np.isfinite(profile.metric_half_width))
    assert profile.ood_distance_threshold > 0

    targets = frame.loc[:1, METRIC_COLUMNS + CONDITION_COLUMNS].copy()
    targets["source_target_id"] = [101, 102]
    result = optimize_target_table(
        checkpoint_path,
        targets,
        data_path,
        candidates_per_target=8,
        confidence=0.9,
        latent_scale=0.05,
        seed=5,
    )
    assert len(result.candidates) == 16
    assert len(result.selected) == 2
    assert result.candidates["ensemble_members"].eq(2).all()
    assert result.selected["selected"].all()
    assert np.isfinite(result.candidates["ood_score"]).all()
    assert np.isfinite(result.candidates["confidence_score"]).all()
    assert np.isfinite(result.candidates["uncertainty"]).all()
    lower = np.asarray(GEOMETRY_MIN)
    upper = np.asarray(GEOMETRY_MAX)
    geometry = result.candidates[GEOMETRY_COLUMNS].to_numpy(dtype=float)
    assert np.all(geometry >= lower - 1e-6)
    assert np.all(geometry <= upper + 1e-6)
    assert set(METRIC_COLUMNS + CONDITION_COLUMNS + GEOMETRY_COLUMNS).issubset(result.selected.columns)


def test_forward_ensemble_builder_preserves_primary_and_adds_members(tmp_path):
    checkpoint_path, data_path, _ = _make_checkpoint(tmp_path)
    base = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base.pop("forward_ensemble_state_dicts")
    torch.save(base, checkpoint_path)
    output_path = tmp_path / "ensemble.pt"
    summary = build_forward_ensemble_checkpoint(
        checkpoint_path,
        data_path,
        output_path,
        members=2,
        epochs=1,
        batch_size=16,
        device_name="cpu",
    )
    upgraded = torch.load(output_path, map_location="cpu", weights_only=False)
    assert summary["members"] == 2
    assert upgraded["forward_ensemble_members"] == 2
    assert len(upgraded["forward_ensemble_state_dicts"]) == 2
    for key, value in base["forward_state_dict"].items():
        assert torch.equal(value, upgraded["forward_ensemble_state_dicts"][0][key])
