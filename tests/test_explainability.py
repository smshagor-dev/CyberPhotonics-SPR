from __future__ import annotations

import numpy as np

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.ml.explainability import attribution_matrix
from sprpcf.ml.tandem import ForwardNetwork


def test_integrated_gradients_outputs_feature_matrix() -> None:
    geometry = np.array([[2.0, 0.5, 45.0], [2.2, 0.55, 50.0]], dtype=np.float32)
    matrix = attribution_matrix(ForwardNetwork(), geometry, method="integrated-gradients", steps=4)
    assert list(matrix.columns) == GEOMETRY_COLUMNS
    assert matrix.shape == geometry.shape
