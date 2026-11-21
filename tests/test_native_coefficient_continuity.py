"""Software-only adversarial controls; not numerical-physics validation."""
import numpy as np
import pytest
from native_coefficient_continuity import match


def test_swapped_native_labels():
    vectors = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    assert match(vectors, vectors[::-1])["permutation"] == [1, 0]


def test_identity():
    vectors = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    assert match(vectors, vectors)["permutation"] == [0, 1]


def test_nan_rejected():
    with pytest.raises(ValueError):
        match([np.array([float("nan")]), np.array([1.0])], [np.array([1.0]), np.array([1.0])])
