from __future__ import annotations

import numpy as np
import pandas as pd

from sprpcf.ml.active_learning import select_uncertain_candidates
from sprpcf.ml.dataset import METRIC_COLUMNS
from sprpcf.ml.tandem import InverseGenerator


def test_select_uncertain_candidates_returns_thresholded_rows() -> None:
    candidates = pd.DataFrame(
        {
            METRIC_COLUMNS[0]: [800.0, 900.0, 1000.0],
            METRIC_COLUMNS[1]: [20.0, 25.0, 30.0],
            METRIC_COLUMNS[2]: [650.0, 700.0, 750.0],
        }
    )
    result = select_uncertain_candidates(
        inverse=InverseGenerator(),
        candidate_metrics=candidates,
        metric_mean=np.array([900.0, 25.0, 700.0], dtype=np.float32),
        metric_scale=np.array([100.0, 5.0, 50.0], dtype=np.float32),
        uncertainty_threshold=0.0,
        passes=4,
        latent_std=0.1,
    )
    assert result.uncertainty.shape == (3,)
    assert len(result.selected) == 3
    assert "uncertainty" in result.selected
