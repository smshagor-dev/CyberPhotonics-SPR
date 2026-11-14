"""Leakage-resistant, evidence-aware forward-baseline and seed-ablation statistics.

Rows at different analyte refractive indices from the same sensor geometry are
correlated. All splits and bootstrap units therefore operate on *geometries*.
This cannot independently verify COMSOL physics or fabricated sensor performance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sprpcf.ml.dataset import (
    FORWARD_INPUT_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS,
    geometry_group_labels, read_table,
)
from sprpcf.utils.reproducibility import git_state, sha256_file

METHODS = ("ridge_full", "random_forest", "ridge_no_ri", "dummy_mean")
ABLATION_METRICS = ("forward_r2", "inverse_target_r2", "constraint_valid_rate",
                    "constraint_violation_rate")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_source(data_path: Path, *, allow_synthetic: bool) -> dict[str, Any]:
    if not data_path.is_file() or data_path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Dataset must be an existing CSV or Parquet file")
    sidecar = data_path.with_name(data_path.name + ".meta.json")
    if not sidecar.is_file():
        raise ValueError("Dataset source sidecar is required: " + str(sidecar))
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Cannot parse dataset source sidecar") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Dataset sidecar must be a JSON object")
    digest = sha256_file(data_path)
    if metadata.get("sha256") != digest:
        raise ValueError("Dataset sidecar SHA-256 mismatch; refusing untraceable data")
    source = str(metadata.get("source") or "").strip().lower()
    if source in {"synthetic", "software_only", "mock", "replay"}:
        if not allow_synthetic:
            raise ValueError("Synthetic/mock data require explicit --allow-software-only")
        evidence_class = "software_only"
    elif source in {"comsol", "experimental", "measured_sensor"}:
        # This is a *declaration*, not evidence of real solver/sensor execution.
        evidence_class = "declared_physical_unqualified"
    else:
        raise ValueError("Dataset sidecar source must declare synthetic, comsol or measured sensor")
    return {
        "dataset_sha256": digest,
        "sidecar_sha256": sha256_file(sidecar),
        "declared_source": source,
        "evidence_class": evidence_class,
        "independently_verified_physics": False,
    }


def _validate_frame(frame: pd.DataFrame) -> np.ndarray:
    needed = FORWARD_INPUT_COLUMNS + METRIC_COLUMNS
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise ValueError(f"Required forward model columns missing: {missing}")
    if len(frame) < 16:
        raise ValueError("At least sixteen labeled rows are required")
    raw = frame[needed]
    try:
        numeric = raw.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All geometry, RI and sensing metric columns must be numerical") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("NaN/Inf or missing numeric data cannot be silently dropped")
    groups = geometry_group_labels(frame)
    if len(np.unique(groups)) < 8:
        raise ValueError("At least eight independent base geometries are required")
    if frame.duplicated(subset=GEOMETRY_COLUMNS + ["analyte_ri"]).any():
        raise ValueError("Duplicate geometry/RI observations create pseudoreplication")
    counts = pd.Series(groups).value_counts()
    if int(counts.min()) < 2:
        raise ValueError("Each base geometry needs at least two different RI observations")
    return groups


def _models(seed: int, trees: int) -> dict[str, Any]:
    return {
        "ridge_full": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=1,
        ),
        "ridge_no_ri": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "dummy_mean": DummyRegressor(strategy="mean"),
    }


def _r2(truth: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.square(truth - np.mean(truth)).sum())
    if denominator < 1e-14:
        return None
    return float(1.0 - np.square(truth - predicted).sum() / denominator)


def _bootstrap_group_mean(
    values: np.ndarray, *, resamples: int, seed: int, confidence: float = 0.95
) -> dict[str, Any]:
    """Percentile CI over independent geometry units within one fixed split."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Grouped bootstrap requires finite group-level values")
    result: dict[str, Any] = {
        "mean": float(values.mean()), "geometry_groups": int(len(values)),
        "confidence": confidence, "ci_low": None, "ci_high": None,
        "interval_status": "insufficient_independent_geometry_groups",
    }
    if len(values) < 3:
        return result
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        means[index] = float(rng.choice(values, size=len(values), replace=True).mean())
    low, high = np.quantile(means, [(1.0 - confidence) / 2.0, (1.0 + confidence) / 2.0])
    result.update(ci_low=float(low), ci_high=float(high), interval_status="descriptive_group_bootstrap")
    return result


def summarize_paired_seed_ablation(
    ablation_csv: str | Path, *, bootstrap_resamples: int = 1000, seed: int = 7,
) -> dict[str, Any]:
    """Compare recorded tandem ablation variants by *paired training seed*.

    Identical seeds are necessary, but do not prove identical held-out datasets.
    Without verified split hashes the contrast remains descriptive only.
    """
    path = Path(ablation_csv)
    if not path.is_file():
        raise ValueError("Missing ablation runs CSV: " + str(path))
    frame = pd.read_csv(path)
    columns = {"seed", "variant", *ABLATION_METRICS}
    if not columns.issubset(frame.columns):
        raise ValueError(f"Ablation CSV missing columns: {sorted(columns - set(frame.columns))}")
    if frame.empty or frame.duplicated(subset=["seed", "variant"]).any():
        raise ValueError("Ablation contains no rows or repeated seed/variant pair")
    values = frame[["seed", *ABLATION_METRICS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Ablation seed/metric values must be finite")
    seed_column = values[:, 0]
    if any(int(value) != value for value in seed_column):
        raise ValueError("Ablation seeds must be integers")
    expected = {"physics_informed", "no_physics_penalty"}
    if set(frame["variant"]) != expected:
        raise ValueError("Ablation must contain exactly the physics and no-physics variants")
    pairs = {}
    for seed_value, group in frame.groupby("seed"):
        if set(group["variant"]) != expected or len(group) != 2:
            raise ValueError(f"Missing matched physics/no-physics pair for seed {seed_value}")
        pairs[int(seed_value)] = group.set_index("variant")
    if len(pairs) < 2:
        raise ValueError("Two or more independently seeded ablation pairs required")
    contrast: dict[str, Any] = {}
    for index, metric in enumerate(ABLATION_METRICS):
        deltas = np.array([
            float(pair.loc["physics_informed", metric] - pair.loc["no_physics_penalty", metric])
            for pair in pairs.values()
        ], dtype=float)
        # Seed-level bootstrap describes training stochasticity, not independent physics.
        stats = _bootstrap_group_mean(deltas, resamples=bootstrap_resamples, seed=seed + index)
        if len(pairs) < 5:
            stats["ci_low"] = None
            stats["ci_high"] = None
            stats["interval_status"] = "insufficient_seed_pairs_for_interval"
        contrast[metric] = stats
    return {
        "source_csv_sha256": sha256_file(path),
        "paired_seeds": sorted(pairs),
        "contrasts_physics_minus_no_penalty": contrast,
        "evidence_class": "surrogate_model_only",
        "split_identity_verified": False,
        "limitation": "Original ablation CSV does not record per-run held-out split hashes. "
                      "Seed-paired contrasts cannot establish identical test sets, physical validity "
                      "or inferential significance; use the same frozen split manifest before publication.",
    }


def run_grouped_ml_study(
    dataset: str | Path,
    output_dir: str | Path,
    *,
    seeds: Sequence[int] = (7, 17, 29),
    test_fraction: float = 0.25,
    trees: int = 60,
    bootstrap_resamples: int = 500,
    allow_software_only: bool = False,
    ablation_runs: str | Path | None = None,
) -> dict[str, Any]:
    """Run paired, split-locked forward baselines and RI-feature ablation."""
    data_path = Path(dataset)
    out = Path(output_dir)
    unique_seeds = [int(value) for value in seeds]
    if len(unique_seeds) < 2 or len(set(unique_seeds)) != len(unique_seeds):
        raise ValueError("Supply at least two unique integer seeds")
    if not 0.15 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.15 and 0.50")
    if trees < 2 or bootstrap_resamples < 200:
        raise ValueError("trees must be >=2 and bootstrap_resamples >=200")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory must be empty: " + str(out))

    provenance = _dataset_source(data_path, allow_synthetic=allow_software_only)
    frame = read_table(data_path)
    groups = _validate_frame(frame)
    features = frame[FORWARD_INPUT_COLUMNS].to_numpy(dtype=float)
    targets = frame[METRIC_COLUMNS].to_numpy(dtype=float)
    group_names = np.array([format(int(value), "016x") for value in groups], dtype=object)
    splits: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []

    for seed in unique_seeds:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        train_idx, test_idx = next(splitter.split(features, targets, groups=groups))
        train_groups = set(group_names[train_idx])
        test_groups = set(group_names[test_idx])
        if train_groups.intersection(test_groups):
            raise RuntimeError("Group leakage detected")
        if len(test_groups) < 2 or len(train_groups) < 3:
            raise ValueError("Split has insufficient train/test independent geometries")
        split_key = _sha256_json({
            "seed": seed, "train": sorted(train_groups), "test": sorted(test_groups),
            "dataset_sha256": provenance["dataset_sha256"],
        })
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[train_idx] = True
        for i in range(len(frame)):
            splits.append({
                "seed": seed, "row_index": i, "geometry_id": group_names[i],
                "partition": "train" if train_mask[i] else "test", "split_sha256": split_key,
            })
        predictions: dict[str, np.ndarray] = {}
        for method, estimator in _models(seed, trees).items():
            selected = slice(None, -1) if method == "ridge_no_ri" else slice(None)
            estimator.fit(features[train_idx, selected], targets[train_idx])
            predict = np.asarray(estimator.predict(features[test_idx, selected]), dtype=float)
            if predict.shape != targets[test_idx].shape or not np.all(np.isfinite(predict)):
                raise ValueError(f"Invalid predictions from {method}")
            predictions[method] = predict
            for metric_index, metric in enumerate(METRIC_COLUMNS):
                y = targets[test_idx, metric_index]
                p = predict[:, metric_index]
                errors = np.abs(y - p)
                metrics.append({
                    "seed": seed, "split_sha256": split_key, "method": method, "target": metric,
                    "mae": float(np.mean(errors)), "rmse": float(np.sqrt(np.mean((y-p)**2))),
                    "r2": _r2(y, p), "train_groups": len(train_groups), "test_groups": len(test_groups),
                    "test_rows": len(test_idx),
                })
                for offset, row_idx in enumerate(test_idx):
                    predicted_rows.append({
                        "seed": seed, "split_sha256": split_key, "method": method,
                        "target": metric, "row_index": int(row_idx),
                        "geometry_id": group_names[row_idx], "actual": float(y[offset]),
                        "predicted": float(p[offset]), "absolute_error": float(errors[offset]),
                    })

        # All contrasts use identical test indices and independent geometry-level means.
        for index, metric in enumerate(METRIC_COLUMNS):
            reference = np.abs(targets[test_idx, index] - predictions["ridge_full"][:, index])
            for method in METHODS:
                if method == "ridge_full":
                    continue
                treatment = np.abs(targets[test_idx, index] - predictions[method][:, index])
                contrast = []
                for group in sorted(test_groups):
                    mask = group_names[test_idx] == group
                    contrast.append(float((treatment[mask] - reference[mask]).mean()))
                statistics = _bootstrap_group_mean(
                    np.array(contrast), resamples=bootstrap_resamples, seed=seed + index * 10007,
                )
                paired.append({
                    "seed": seed, "split_sha256": split_key, "method": method,
                    "reference_method": "ridge_full", "target": metric,
                    "delta_group_mean_mae": statistics["mean"],
                    "geometry_groups": statistics["geometry_groups"],
                    "ci_low": statistics["ci_low"], "ci_high": statistics["ci_high"],
                    "interval_status": statistics["interval_status"],
                })

    metrics_frame = pd.DataFrame(metrics)
    summary_rows: list[dict[str, Any]] = []
    for (method, target), group in metrics_frame.groupby(["method", "target"], sort=True):
        summary_rows.append({
            "method": method, "target": target, "seeds": len(group),
            "mean_split_mae": float(group["mae"].mean()),
            "sd_split_mae": float(group["mae"].std(ddof=1)),
            "mean_split_rmse": float(group["rmse"].mean()),
            "sd_split_rmse": float(group["rmse"].std(ddof=1)),
            "mean_split_r2": None if group["r2"].isna().any() else float(group["r2"].mean()),
        })

    ablation_summary = None
    if ablation_runs is not None:
        ablation_summary = summarize_paired_seed_ablation(
            ablation_runs, bootstrap_resamples=bootstrap_resamples, seed=unique_seeds[0],
        )
    out.mkdir(parents=True, exist_ok=False) if not out.exists() else None
    tables = {
        "splits.csv": pd.DataFrame(splits),
        "per_seed_metrics.csv": metrics_frame,
        "predictions.csv": pd.DataFrame(predicted_rows),
        "paired_group_bootstrap.csv": pd.DataFrame(paired),
        "aggregate_metrics.csv": pd.DataFrame(summary_rows),
    }
    for filename, table in tables.items():
        table.to_csv(out / filename, index=False)
    report = {
        "schema_version": 1, "study": "paired_group_disjoint_forward_baselines",
        "source": provenance, "git": git_state(Path(__file__).resolve().parents[3]),
        "parameters": {
            "seeds": unique_seeds, "test_fraction": test_fraction,
            "random_forest_trees": trees, "bootstrap_resamples": bootstrap_resamples,
            "features_full": FORWARD_INPUT_COLUMNS, "features_ablated": GEOMETRY_COLUMNS,
            "targets": METRIC_COLUMNS, "methods": list(METHODS),
        },
        "observations": {
            "rows": len(frame), "unique_geometry_groups": len(np.unique(groups)),
            "split_count": len(unique_seeds),
        },
        "outputs_sha256": {filename: sha256_file(out / filename) for filename in tables},
        "sklearn_version": sklearn.__version__,
        "ablation_study": ablation_summary if ablation_summary is not None else {
            "status": "not_supplied", "needed": "real run_ablation_study output with paired split evidence",
        },
        "statistical_interpretation": [
            "Per-seed comparisons use identical train/test geometry partitions across all methods.",
            "The confidence intervals are paired bootstrap over *geometry* means within one split, "
            "not an independent sample of RI rows.",
            "Repeated random seeds reuse geometry groups across splits; pooled seed-level significance "
            "or p-values must not be inferred from these descriptive summaries.",
            "The RI-feature ablation is a forward-prediction ablation, not a tandem physics-loss ablation.",
            "Ablation results from existing runner have no frozen split IDs and are only descriptive.",
        ],
        "physics_performance_claim_ready": False,
        "publication_ready": False,
        "scientific_blockers": [
            "Independent physics/model and raw COMSOL validation with mesh convergence",
            "Fair proposed tandem inverse-design comparison on frozen held-out geometries",
            "Qualified dataset provenance and trained checkpoint with identical split identities",
            "Independent physical target verification, calibration/OOD and constraints ablations",
        ],
    }
    (out / "STUDY_MANIFEST.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    (out / "STUDY_REPORT.md").write_text(
        "# Grouped ML Baselines and RI Ablation\n\n"
        f"Evidence class: **{provenance['evidence_class']}**.\n\n"
        f"Rows: {len(frame)}; unique sensor geometries: {len(np.unique(groups))}; "
        f"seeds: {', '.join(map(str, unique_seeds))}.\n\n"
        "Results: `aggregate_metrics.csv`, `per_seed_metrics.csv`, "
        "`paired_group_bootstrap.csv` and `predictions.csv`. "
        "See `splits.csv` for frozen geometry-level partitions.\n\n"
        "**This is not verified optical physics or proof that the tandem inverse method is superior.** "
        "The manifest records remaining requirements for journal claims.\n", encoding="utf-8",
    )
    return report
