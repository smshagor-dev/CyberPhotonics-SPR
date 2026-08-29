from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sprpcf.ml.active_learning import select_uncertain_candidates, trigger_comsol_for_uncertain_candidates
from sprpcf.ml.dataset import CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.ml.tandem import InverseGenerator


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            METRIC_COLUMNS[0]: [800.0, 900.0, 1000.0],
            METRIC_COLUMNS[1]: [20.0, 25.0, 30.0],
            METRIC_COLUMNS[2]: [650.0, 700.0, 750.0],
            CONDITION_COLUMNS[0]: [1.33, 1.35, 1.37],
        }
    )


def test_select_uncertain_candidates_returns_generated_physical_geometry() -> None:
    result = select_uncertain_candidates(
        inverse=InverseGenerator(dropout=0.2),
        candidate_metrics=_candidate_frame(),
        metric_mean=np.array([900.0, 25.0, 700.0], dtype=np.float32),
        metric_scale=np.array([100.0, 5.0, 50.0], dtype=np.float32),
        condition_mean=np.array([1.35], dtype=np.float32),
        condition_scale=np.array([0.02], dtype=np.float32),
        geometry_mean=np.array([2.0, 0.5, 45.0, 0.6], dtype=np.float32),
        geometry_scale=np.array([0.5, 0.1, 10.0, 0.2], dtype=np.float32),
        uncertainty_threshold=-1.0,
        passes=4,
    )
    assert result.uncertainty.shape == (3,)
    assert len(result.selected) == 3
    assert set(GEOMETRY_COLUMNS).issubset(result.selected.columns)
    assert "uncertainty" in result.selected


def test_comsol_runner_receives_only_selected_rows(tmp_path: Path) -> None:
    selected = _candidate_frame().iloc[:1].copy()
    selected["pitch_um"] = 2.0
    selected["d_over_lambda"] = 0.5
    selected["metal_thickness_nm"] = 45.0
    selected["channel_radius_um"] = 0.6
    selected["uncertainty"] = 0.2
    from sprpcf.ml.active_learning import ActiveLearningResult

    result = ActiveLearningResult(_candidate_frame(), np.array([0.2]), selected)
    seen: list[int] = []

    def runner(rows: pd.DataFrame) -> pd.DataFrame:
        seen.append(len(rows))
        return rows.assign(status="ok")

    out = tmp_path / "comsol.csv"
    updated = trigger_comsol_for_uncertain_candidates(
        result,
        model_path=tmp_path / "unused.mph",
        config_path=tmp_path / "unused.yaml",
        output_path=out,
        runner=runner,
    )
    assert seen == [1]
    assert updated.comsol_results is not None
    assert out.exists()
