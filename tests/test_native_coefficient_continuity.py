"""Software-only adversarial controls; not numerical-physics validation."""
import numpy as np
import pytest
from research.native_coefficient_continuity import match


def test_swapped_native_labels():
    vectors = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    assert match(vectors, vectors[::-1])["permutation"] == [1, 0]


def test_identity():
    vectors = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    assert match(vectors, vectors)["permutation"] == [0, 1]


def test_nan_rejected():
    with pytest.raises(ValueError):
        match([np.array([float("nan")]), np.array([1.0])], [np.array([1.0]), np.array([1.0])])


from research.native_coefficient_continuity import plaquette_holonomy


def _grid_edges():
    edges = []
    for i in range(3):
        for j in range(2):
            edges.append({"from": [i, j], "to": [i, j + 1], "permutation": [0, 1]})
    for j in range(3):
        for i in range(2):
            edges.append({"from": [i, j], "to": [i + 1, j], "permutation": [0, 1]})
    return edges


def test_consistent_grid_has_four_identity_plaquettes():
    cells = plaquette_holonomy(_grid_edges())
    assert len(cells) == 4
    assert all(cell["path_consistent"] for cell in cells)


def test_one_swapped_edge_exposes_path_inconsistency():
    edges = _grid_edges()
    edges[0]["permutation"] = [1, 0]
    cells = plaquette_holonomy(edges)
    assert sum(not cell["path_consistent"] for cell in cells) == 1


def test_missing_edge_fails_closed():
    with pytest.raises(ValueError, match="Missing grid edge"):
        plaquette_holonomy(_grid_edges()[:-1])


def test_invalid_permutation_fails_closed():
    edges = _grid_edges()
    edges[0]["permutation"] = [0, 0]
    with pytest.raises(ValueError, match="Invalid two-mode permutation"):
        plaquette_holonomy(edges)
