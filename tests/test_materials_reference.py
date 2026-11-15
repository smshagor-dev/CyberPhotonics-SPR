"""Reproducible CC0 measured Au optical-source tests, no FEM claims."""
from __future__ import annotations

import hashlib
from importlib import resources

import numpy as np
import pytest

from sprpcf.simulation.materials_reference import (
    au_johnson_christy_nk,
    au_johnson_christy_permittivity,
    au_reference_manifest,
)


def test_source_exact_measured_knots_and_reference_provenance() -> None:
    values = au_johnson_christy_nk(np.array([0.7560, 0.8211, 1.9370]))
    assert np.allclose(values, [0.14 + 4.542j, 0.16 + 5.083j, 0.92 + 13.78j])
    meta = au_reference_manifest()
    assert meta["row_count"] == 49
    assert meta["database_license"] == "CC0-1.0"
    assert meta["source_doi"].endswith("PhysRevB.6.4370")
    raw = resources.files("sprpcf.simulation").joinpath("data", "au_johnson_christy_1972.csv").read_bytes()
    assert meta["dataset_sha256"] == hashlib.sha256(raw).hexdigest()


def test_interpolation_is_linear_in_n_and_k_and_eps_passive() -> None:
    mid = (0.7560 + 0.8211) / 2
    n = complex(au_johnson_christy_nk(mid))
    assert n.real == pytest.approx(0.15)
    assert n.imag == pytest.approx((4.542 + 5.083) / 2)
    eps = complex(au_johnson_christy_permittivity(mid))
    assert eps == pytest.approx(n**2)
    assert eps.real < 0
    assert eps.imag > 0
    assert complex(au_johnson_christy_permittivity(0.8)).imag > 0


def test_out_of_range_fails_closed_no_extrapolation() -> None:
    for invalid in [0.1, 2.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="no extrapolation"):
            au_johnson_christy_nk(invalid)
    with pytest.raises(ValueError, match="no extrapolation"):
        au_johnson_christy_nk(np.array([0.8, 2.0]))


def test_material_reference_never_claims_sensor_measurement() -> None:
    metadata = au_reference_manifest()
    assert metadata["source_type"] == "measured_bulk_film_reference"
    assert any("not measurements of the simulated sensor film" in s for s in metadata["limitations"])
