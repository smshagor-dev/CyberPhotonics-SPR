from __future__ import annotations

import numpy as np

from sprpcf.ml.dataset import FORWARD_INPUT_COLUMNS
from sprpcf.ml.explainability import attribution_matrix
from sprpcf.ml.tandem import ForwardNetwork


def test_integrated_gradients_outputs_full_conditioned_feature_matrix() -> None:
    standardized_inputs = np.array(
        [[0.0, 0.0, 0.0, 0.0, -1.0], [0.2, 0.5, -0.1, 0.3, 1.0]],
        dtype=np.float32,
    )
    matrix = attribution_matrix(ForwardNetwork(), standardized_inputs, method="integrated-gradients", steps=4)
    assert list(matrix.columns) == FORWARD_INPUT_COLUMNS
    assert matrix.shape == standardized_inputs.shape
