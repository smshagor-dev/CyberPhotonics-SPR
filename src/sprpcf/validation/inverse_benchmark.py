"""Matched *surrogate-query* inverse-design comparison on geometry-heldout targets.

Equal candidate budgets are NOT equal total compute. A forward network trained
jointly with the tandem method is NOT an independent physics evaluator.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint
from sprpcf.ml.dataset import (
    CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS, DesignDataModule,
    geometry_group_labels, read_table,
)
from sprpcf.ml.losses import GEOMETRY_MAX, GEOMETRY_MIN
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator
from sprpcf.utils.reproducibility import git_state, sha256_file
from sprpcf.validation.publication_ml import (
    _bootstrap_group_mean, _dataset_source, _validate_frame,
)

METHODS = ("tandem_latent", "training_retrieval", "uniform_random", "tandem_zero_latent")
LOWER = np.asarray(GEOMETRY_MIN, dtype=np.float64)
UPPER = np.asarray(GEOMETRY_MAX, dtype=np.float64)


def _hash_object(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _scaler(checkpoint: dict[str, Any], stem: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint[stem + "_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint[stem + "_scale"], dtype=np.float64)
    if (mean.shape != (width,) or scale.shape != (width,)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale))
            or not np.all(scale > 0)):
        raise ValueError(f"Invalid checkpoint {stem} scaler")
    return mean, scale


def _validate_checkpoint(
    checkpoint: dict[str, Any], frame: pd.DataFrame, groups: np.ndarray, data_path: Path
) -> tuple[np.ndarray, np.ndarray, str]:
    if checkpoint.get("dataset_sha256") != sha256_file(data_path):
        raise ValueError("Checkpoint dataset SHA-256 mismatch or missing training provenance")
    for key in ("train_row_indices", "validation_row_indices",
                "train_geometry_ids", "validation_geometry_ids", "training_parameters"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing frozen split/training provenance: {key}")
    training = checkpoint["training_parameters"]
    if not isinstance(training, dict) or any(
        not isinstance(training.get(key), int) or training[key] < 1
        for key in ("forward_epochs", "inverse_epochs")
    ):
        raise ValueError("Checkpoint must record positive training epoch counts")

    seed = int(checkpoint["seed"])
    reference = DesignDataModule(data_path, batch_size=256, seed=seed)
    reference.setup()
    train = np.asarray(checkpoint["train_row_indices"], dtype=np.int64)
    heldout = np.asarray(checkpoint["validation_row_indices"], dtype=np.int64)
    if not np.array_equal(train, reference.train_indices) or not np.array_equal(heldout, reference.val_indices):
        raise ValueError("Checkpoint row partitions do not match reproducible training split")
    if set(train).intersection(set(heldout)) or len(train) + len(heldout) != len(frame):
        raise ValueError("Train/validation rows overlap or are incomplete")
    identities = np.array([format(int(value), "016x") for value in groups], dtype=object)
    expected_train = sorted(set(identities[train]))
    expected_heldout = sorted(set(identities[heldout]))
    if (expected_train != list(checkpoint["train_geometry_ids"])
            or expected_heldout != list(checkpoint["validation_geometry_ids"])
            or set(expected_train).intersection(expected_heldout)):
        raise ValueError("Checkpoint geometry partitions differ from dataset")
    if len(expected_heldout) < 3 or len(expected_train) < 5:
        raise ValueError("At least 3 independent validation and 5 training geometries are required")
    split_sha = _hash_object({
        "dataset_sha256": checkpoint["dataset_sha256"],
        "train_groups": expected_train, "validation_groups": expected_heldout,
        "training_seed": seed,
    })
    return train, heldout, split_sha


def _models(checkpoint: dict[str, Any]) -> tuple[ForwardNetwork, InverseGenerator]:
    forward = ForwardNetwork()
    inverse = InverseGenerator()
    forward.load_state_dict(checkpoint["forward_state_dict"])
    inverse.load_state_dict(checkpoint["inverse_state_dict"])
    forward.eval()
    inverse.eval()
    return forward, inverse


def _targets(frame: pd.DataFrame, heldout: np.ndarray, identities: np.ndarray) -> pd.DataFrame:
    """One central-RI target per independent validation geometry, no row leakage."""
    by_group: dict[str, list[int]] = {}
    for index in heldout:
        by_group.setdefault(str(identities[index]), []).append(int(index))
    selected: list[dict[str, Any]] = []
    for group_id in sorted(by_group):
        indices = sorted(by_group[group_id], key=lambda i: (float(frame.iloc[i]["analyte_ri"]), i))
        index = indices[(len(indices) - 1) // 2]
        row = frame.iloc[index]
        selected.append({
            "target_id": len(selected), "target_group_id": group_id,
            "source_row_index": index,
            **{column: float(row[column]) for column in [*METRIC_COLUMNS, *CONDITION_COLUMNS]},
        })
    return pd.DataFrame(selected)


def _reference_pool(
    frame: pd.DataFrame, train: np.ndarray, identities: np.ndarray,
    target_metrics: np.ndarray, ri: float, scale: np.ndarray, budget: int,
) -> np.ndarray:
    """Find closest training examples; only one candidate per distinct train geometry initially."""
    subset = frame.iloc[train]
    observed = subset[METRIC_COLUMNS].to_numpy(dtype=np.float64)
    conditions = subset[CONDITION_COLUMNS].to_numpy(dtype=np.float64).ravel()
    distance = np.linalg.norm((observed - target_metrics[None, :]) / scale[None, :], axis=1)
    distance += np.abs(conditions - ri) / 0.04
    ordering = np.argsort(distance, kind="stable")
    distinct: list[int] = []
    seen: set[str] = set()
    for candidate in ordering:
        idx = int(train[candidate])
        group = str(identities[idx])
        if group not in seen:
            distinct.append(idx)
            seen.add(group)
    if not distinct:
        raise ValueError("No training reference geometries")
    tiled = [distinct[i % len(distinct)] for i in range(budget)]
    return frame.iloc[tiled][GEOMETRY_COLUMNS].to_numpy(dtype=np.float64)


def _tandem_pool(
    inverse: InverseGenerator, target_metrics: np.ndarray, ri: float,
    metric_mean: np.ndarray, metric_scale: np.ndarray,
    condition_mean: np.ndarray, condition_scale: np.ndarray,
    geometry_mean: np.ndarray, geometry_scale: np.ndarray,
    budget: int, generator: torch.Generator, *, randomized: bool,
    latent_scale: float,
) -> np.ndarray:
    metric = ((target_metrics - metric_mean) / metric_scale).astype(np.float32)
    condition = np.array([(ri - condition_mean[0]) / condition_scale[0]], dtype=np.float32)
    latent = (
        torch.randn((budget, inverse.latent_dim), generator=generator) * latent_scale
        if randomized else torch.zeros((budget, inverse.latent_dim))
    )
    with torch.no_grad():
        standardized = inverse(
            torch.tensor(np.repeat(metric[None, :], budget, axis=0)),
            torch.tensor(np.repeat(condition[None, :], budget, axis=0)),
            latent,
        ).numpy()
    return standardized.astype(np.float64) * geometry_scale[None, :] + geometry_mean[None, :]


def _forward_predictions(
    forward: ForwardNetwork, geometry: np.ndarray, ri: float,
    geometry_mean: np.ndarray, geometry_scale: np.ndarray,
    condition_mean: np.ndarray, condition_scale: np.ndarray,
    metric_mean: np.ndarray, metric_scale: np.ndarray,
) -> np.ndarray:
    standardized_geometry = (geometry - geometry_mean[None, :]) / geometry_scale[None, :]
    standardized_ri = np.full((len(geometry), 1), (ri - condition_mean[0]) / condition_scale[0])
    x = np.concatenate([standardized_geometry, standardized_ri], axis=1).astype(np.float32)
    with torch.no_grad():
        output = forward(torch.from_numpy(x)).numpy().astype(np.float64)
    predictions = output * metric_scale[None, :] + metric_mean[None, :]
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite forward surrogate predictions")
    return predictions


def run_inverse_benchmark(
    dataset: str | Path, checkpoint_file: str | Path, output_dir: str | Path, *,
    seeds: Sequence[int] = (7, 17, 29), candidates_per_target: int = 16,
    latent_scale: float = 0.10, bootstrap_resamples: int = 500,
    allow_software_only: bool = False,
) -> dict[str, Any]:
    """Benchmark four methods with identical forward-surrogate candidate counts.

    Results are *surrogate-ranked* and must not be used as verified physics
    or independent training-seed replications.
    """
    data_path = Path(dataset)
    checkpoint_path = Path(checkpoint_file)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty: " + str(output))
    if (len(seeds) < 2 or len(set(int(s) for s in seeds)) != len(seeds)
            or any(int(s) != s for s in seeds)):
        raise ValueError("At least two unique integer candidate-generation seeds required")
    if candidates_per_target < 4 or bootstrap_resamples < 200 or not 0 < latent_scale <= 10:
        raise ValueError("Budget >= 4, bootstrap >= 200 and positive latent scale required")
    provenance = _dataset_source(data_path, allow_synthetic=allow_software_only)
    frame = read_table(data_path)
    groups = _validate_frame(frame)
    identities = np.array([format(int(v), "016x") for v in groups], dtype=object)
    checkpoint = load_tandem_checkpoint(checkpoint_path)
    train, heldout, split_sha = _validate_checkpoint(checkpoint, frame, groups, data_path)
    geometry_mean, geometry_scale = _scaler(checkpoint, "geometry", 4)
    condition_mean, condition_scale = _scaler(checkpoint, "condition", 1)
    metric_mean, metric_scale = _scaler(checkpoint, "metric", 3)
    forward, inverse = _models(checkpoint)
    target_table = _targets(frame, heldout, identities)

    candidates_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    target_groups = target_table["target_group_id"].tolist()
    for design_seed in seeds:
        seed = int(design_seed)
        for _, target in target_table.iterrows():
            target_id = int(target["target_id"])
            group_id = str(target["target_group_id"])
            desired = target[METRIC_COLUMNS].to_numpy(dtype=np.float64)
            ri = float(target["analyte_ri"])
            generator = torch.Generator(device="cpu")
            generator.manual_seed((seed * 104729 + target_id) % (2**63))
            rng = np.random.default_rng((seed * 11117 + target_id) % (2**63))
            pools: dict[str, np.ndarray] = {
                "tandem_latent": _tandem_pool(
                    inverse, desired, ri, metric_mean, metric_scale,
                    condition_mean, condition_scale, geometry_mean, geometry_scale,
                    candidates_per_target, generator, randomized=True, latent_scale=latent_scale,
                ),
                "training_retrieval": _reference_pool(
                    frame, train, identities, desired, ri, metric_scale, candidates_per_target,
                ),
                "uniform_random": rng.uniform(LOWER, UPPER, size=(candidates_per_target, 4)),
                "tandem_zero_latent": _tandem_pool(
                    inverse, desired, ri, metric_mean, metric_scale,
                    condition_mean, condition_scale, geometry_mean, geometry_scale,
                    candidates_per_target, generator, randomized=False, latent_scale=latent_scale,
                ),
            }
            for method in METHODS:
                started = time.perf_counter()
                raw = pools[method]
                if not np.isfinite(raw).all():
                    raise ValueError("Non-finite proposed candidate geometry")
                projected = np.clip(raw, LOWER[None, :], UPPER[None, :])
                valid_raw = np.all((raw >= LOWER[None, :]) & (raw <= UPPER[None, :]), axis=1)
                predictions = _forward_predictions(
                    forward, projected, ri, geometry_mean, geometry_scale,
                    condition_mean, condition_scale, metric_mean, metric_scale,
                )
                errors = np.abs(predictions - desired[None, :])
                normalized = np.mean(errors / metric_scale[None, :], axis=1)
                best_index = int(np.argmin(normalized))
                unique_count = len(np.unique(np.round(projected, decimals=8), axis=0))
                for candidate_id in range(candidates_per_target):
                    candidates_rows.append({
                        "design_seed": seed, "method": method, "target_id": target_id,
                        "target_group_id": group_id, "candidate_id": candidate_id,
                        "surrogate_query_number": candidate_id + 1,
                        "chosen": candidate_id == best_index,
                        "raw_geometry_valid": bool(valid_raw[candidate_id]),
                        "surrogate_normalized_mae": float(normalized[candidate_id]),
                        **{name: float(projected[candidate_id, i]) for i, name in enumerate(GEOMETRY_COLUMNS)},
                        **{f"surrogate_pred_{name}": float(predictions[candidate_id, i])
                           for i, name in enumerate(METRIC_COLUMNS)},
                    })
                selected = projected[best_index]
                row: dict[str, Any] = {
                    "design_seed": seed, "method": method,
                    "target_id": target_id, "target_group_id": group_id,
                    "source_row_index": int(target["source_row_index"]),
                    "split_sha256": split_sha, "candidate_budget": candidates_per_target,
                    "unique_candidate_geometries": unique_count,
                    "surrogate_normalized_mae": float(normalized[best_index]),
                    "raw_valid_rate": float(valid_raw.mean()),
                    "selected_projection_distance": float(
                        np.mean(np.abs(raw[best_index] - selected) / (UPPER - LOWER))
                    ),
                    "surrogate_scoring_seconds": float(time.perf_counter() - started),
                    "independent_solver_queries": 0,
                    **{name: float(selected[i]) for i, name in enumerate(GEOMETRY_COLUMNS)},
                    "analyte_ri": ri,
                }
                for i, name in enumerate(METRIC_COLUMNS):
                    row[f"target_{name}"] = float(desired[i])
                    row[f"surrogate_pred_{name}"] = float(predictions[best_index, i])
                    row[f"surrogate_abs_error_{name}"] = float(errors[best_index, i])
                selected_rows.append(row)
                request_id = _hash_object({
                    "checkpoint_sha256": sha256_file(checkpoint_path), "split_sha256": split_sha,
                    "design_seed": seed, "method": method,
                    "target_id": target_id, "geometry": selected.tolist(),
                    "analyte_ri": ri,
                })[:24]
                requests.append({
                    "request_id": request_id, "method": method, "design_seed": seed,
                    "target_id": target_id, "target_group_id": group_id,
                    "analyte_ri": ri,
                    **{name: float(selected[i]) for i, name in enumerate(GEOMETRY_COLUMNS)},
                    **{f"target_{name}": float(desired[i]) for i, name in enumerate(METRIC_COLUMNS)},
                    "fixed_geometry_ri_sweep_required": True,
                    "verification_status": "not_run",
                    "physical_solver_result": "not_supplied",
                })
        subset = pd.DataFrame([row for row in selected_rows if row["design_seed"] == seed])
        for rival in METHODS:
            if rival == "tandem_latent":
                continue
            paired_differences: list[float] = []
            for group in target_groups:
                tandem = subset.loc[(subset["target_group_id"] == group)
                                    & (subset["method"] == "tandem_latent")]
                other = subset.loc[(subset["target_group_id"] == group)
                                   & (subset["method"] == rival)]
                if len(tandem) != 1 or len(other) != 1:
                    raise RuntimeError("Paired comparison is missing a target group")
                paired_differences.append(
                    float(other.iloc[0]["surrogate_normalized_mae"]
                          - tandem.iloc[0]["surrogate_normalized_mae"])
                )
            summary = _bootstrap_group_mean(
                np.array(paired_differences, dtype=np.float64),
                resamples=bootstrap_resamples, seed=seed,
            )
            paired_rows.append({
                "design_seed": seed, "method": rival, "reference": "tandem_latent",
                "paired_independent_target_geometries": summary["geometry_groups"],
                "mean_rival_minus_tandem_surrogate_error": summary["mean"],
                "ci_low": summary["ci_low"] if summary["ci_low"] is not None else "",
                "ci_high": summary["ci_high"] if summary["ci_high"] is not None else "",
                "interval_status": summary["interval_status"],
            })

    split_rows = [{
        "row_index": i, "geometry_id": identities[i],
        "split": "train" if i in set(map(int, train)) else "heldout_validation",
        "split_sha256": split_sha,
    } for i in range(len(frame))]
    output.mkdir(parents=True, exist_ok=True) if not output.exists() else None
    tables = {
        "split_membership.csv": pd.DataFrame(split_rows),
        "validation_targets.csv": target_table,
        "all_candidates.csv": pd.DataFrame(candidates_rows),
        "selected_designs.csv": pd.DataFrame(selected_rows),
        "PHYSICS_VERIFICATION_REQUESTS.csv": pd.DataFrame(requests),
        "paired_geometry_bootstrap.csv": pd.DataFrame(paired_rows),
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)
    manifest = {
        "schema_version": 1,
        "experiment": "matched_surrogate_query_inverse_design",
        "evidence_class": provenance["evidence_class"],
        "dataset_provenance": provenance,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trained_checkpoint_dataset_sha256": checkpoint["dataset_sha256"],
        "training_seed": int(checkpoint["seed"]),
        "training_parameters": checkpoint["training_parameters"],
        "candidate_generation_seeds": [int(value) for value in seeds],
        "seeds_are_independent_training_replicates": False,
        "heldout_geometry_groups": len(target_table),
        "training_geometry_groups": len(set(identities[train])),
        "split_sha256": split_sha,
        "methods": list(METHODS),
        "candidates_per_target_per_method": candidates_per_target,
        "forward_surrogate_evaluations_per_target_per_method": candidates_per_target,
        "matched_total_compute_or_wallclock": False,
        "physics_solver_evaluations": 0,
        "model_selection_uses_training_coupled_forward_surrogate": True,
        "independent_validation_status": "not_run",
        "scientific_submission_ready": False,
        "methodology_warnings": [
            "All methods are ranked by the tandem-trained forward surrogate, not independent COMSOL.",
            "The checkpoint training procedure used the held-out groups for validation; this is not a fresh untouched external test.",
            "Candidate generation seeds share one fixed trained checkpoint and fixed validation geometries.",
            "Equal surrogate-query counts do not equal wall-clock, acquisition or physical solver costs.",
            "The fixed-zero-latent variant samples repeated geometry and is an exploration ablation only.",
            "Group bootstrap intervals are descriptive within seed; never treat different generation seeds as independent experiments.",
            "Dataset declared COMSOL/measured source still requires a separately qualified physics evidence registry.",
        ],
        "git": git_state(Path(__file__).resolve().parents[3]),
        "outputs_sha256": {name: sha256_file(output / name) for name in tables},
    }
    (output / "INVERSE_BENCHMARK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    (output / "INVERSE_BENCHMARK_REPORT.md").write_text(
        "# Matched-query inverse-design benchmark\n\n"
        f"Source evidence: **{provenance['evidence_class']}**; frozen split: `{split_sha}`.\n"
        f"One target per held-out geometry: **{len(target_table)}**. "
        f"Candidate budgets: **{candidates_per_target} per method per target**.\n\n"
        "All listed errors are **forward-surrogate predicted**, not independently verified optical metrics.\n"
        "Independent fixed-geometry COMSOL verification requests are in "
        "`PHYSICS_VERIFICATION_REQUESTS.csv`. No physics result is claimed.\n", encoding="utf-8",
    )
    return manifest
