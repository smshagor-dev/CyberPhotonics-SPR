"""Conservative unit tests; fixtures MUST NOT constitute electromagnetic proof."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sprpcf.simulation.open_vector_grid import matched_vector_overlaps, preflight_grid, run_vector_grid


def test_grid_preflight_rejects_duplicated_or_invalid_optical_inputs() -> None:
    preflight_grid((0.75, 0.8), (1.35, 1.36), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        preflight_grid((0.8, 0.8), (1.35, 1.36), 2)
    with pytest.raises(ValueError, match="strictly increasing"):
        preflight_grid((0.75, 0.8), (1.35, 1.35), 2)
    with pytest.raises(ValueError, match="wavelengths exceed"):
        preflight_grid((0.2, 0.8), (1.35, 1.36), 2)
    with pytest.raises(ValueError, match="num_modes"):
        preflight_grid((0.75, 0.8), (1.35, 1.36), 0)
    with pytest.raises(ValueError, match="RI values"):
        preflight_grid((0.75, 0.8), (0.9, 1.36), 2)


def test_vector_eh_overlap_matcher_handles_mode_swap() -> None:
    a, b = [MagicMock(), MagicMock()], [MagicMock(), MagicMock()]
    # Reference A0 has its strongest overlap with candidate B1.
    a[0].calculate_overlap.side_effect = [0.10 + 0j, 0.95 + 0j]
    a[1].calculate_overlap.side_effect = [0.92 + 0j, 0.05 + 0j]
    matches = matched_vector_overlaps(a, b)
    assert [(r["previous_mode_index"], r["current_mode_index"]) for r in matches] == [(0, 1), (1, 0)]
    assert matches[0]["vector_eh_poynting_overlap_abs"] == pytest.approx(0.95)
    assert all(r["mode_identity_physically_verified"] is False for r in matches)


def test_vector_overlap_rejects_bad_modes() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        matched_vector_overlaps([], [])
    mode = MagicMock()
    mode.calculate_overlap.return_value = complex(float("nan"), 0)
    with pytest.raises(ValueError, match="nonfinite"):
        matched_vector_overlaps([mode], [mode])


def test_bidirectional_overlap_margin_flags_ambiguous_global_assignment() -> None:
    a, b = [MagicMock(), MagicMock()], [MagicMock(), MagicMock()]
    # Global optimum chooses (0,1),(1,0), but candidate 1 has a
    # stronger competing previous mode 1; the reverse margin is negative.
    a[0].calculate_overlap.side_effect = [0.70 + 0j, 0.60 + 0j]
    a[1].calculate_overlap.side_effect = [0.95 + 0j, 0.80 + 0j]
    matches = matched_vector_overlaps(a, b)
    assert matches[0]["current_mode_index"] == 1
    assert matches[1]["current_mode_index"] == 0
    assert matches[0]["reverse_overlap_margin"] == pytest.approx(-0.20)
    assert matches[0]["assignment_ambiguous_nonpositive_margin"] is True
    assert all(row["mode_identity_physically_verified"] is False for row in matches)


def test_bidirectional_overlap_margin_unambiguous_swap() -> None:
    a, b = [MagicMock(), MagicMock()], [MagicMock(), MagicMock()]
    a[0].calculate_overlap.side_effect = [0.10 + 0j, 0.95 + 0j]
    a[1].calculate_overlap.side_effect = [0.92 + 0j, 0.05 + 0j]
    matches = matched_vector_overlaps(a, b)
    assert all(row["bidirectional_min_overlap_margin"] > 0 for row in matches)
    assert all(not row["assignment_ambiguous_nonpositive_margin"] for row in matches)


def test_native_output_directory_is_immutable_before_solver_import(tmp_path: Path) -> None:
    destination = tmp_path / "native"
    destination.mkdir()
    sentinel = destination / "raw_vector_eh_ri0_wl0.npz"
    sentinel.write_bytes(b"original-eigenfield-evidence")
    with pytest.raises(FileExistsError, match="already exists"):
        run_vector_grid(destination)
    assert sentinel.read_bytes() == b"original-eigenfield-evidence"
    assert sorted(p.name for p in destination.iterdir()) == [sentinel.name]


def test_native_output_rejects_file_destination(tmp_path: Path) -> None:
    destination = tmp_path / "native"
    destination.write_text("immutable", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_vector_grid(destination)
    assert destination.read_text(encoding="utf-8") == "immutable"


def test_native_output_rejects_existing_empty_directory(tmp_path: Path) -> None:
    # Exclusive ownership is necessary even when no artifacts are visible yet.
    destination = tmp_path / "native"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_vector_grid(destination)
    assert list(destination.iterdir()) == []
