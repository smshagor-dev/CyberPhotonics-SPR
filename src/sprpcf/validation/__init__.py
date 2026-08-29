"""Research-grade scientific validation, ablation, and closed-loop utilities."""

from sprpcf.validation.benchmark import evaluate_checkpoint, evaluate_ridge_baseline, run_validation_pack
from sprpcf.validation.closed_loop import (
    AcceptanceThresholds,
    ClosedLoopArtifacts,
    append_accepted_simulation_rows,
    build_validation_ri_values,
    evaluate_closed_loop_results,
    run_closed_loop_iteration,
)
from sprpcf.validation.scientific import (
    bootstrap_mean_ci,
    fabrication_constraint_report,
    fixed_geometry_sweep_report,
    summarize_sweep_report,
)

__all__ = [
    "AcceptanceThresholds",
    "ClosedLoopArtifacts",
    "append_accepted_simulation_rows",
    "bootstrap_mean_ci",
    "build_validation_ri_values",
    "evaluate_checkpoint",
    "evaluate_closed_loop_results",
    "evaluate_ridge_baseline",
    "fabrication_constraint_report",
    "fixed_geometry_sweep_report",
    "run_closed_loop_iteration",
    "run_validation_pack",
    "summarize_sweep_report",
]
