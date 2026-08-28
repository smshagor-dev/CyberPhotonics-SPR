"""Research-grade scientific validation and ablation utilities."""

from sprpcf.validation.benchmark import evaluate_checkpoint, evaluate_ridge_baseline, run_validation_pack
from sprpcf.validation.scientific import (
    bootstrap_mean_ci,
    fabrication_constraint_report,
    fixed_geometry_sweep_report,
    summarize_sweep_report,
)

__all__ = [
    "bootstrap_mean_ci",
    "evaluate_checkpoint",
    "evaluate_ridge_baseline",
    "fabrication_constraint_report",
    "fixed_geometry_sweep_report",
    "run_validation_pack",
    "summarize_sweep_report",
]
