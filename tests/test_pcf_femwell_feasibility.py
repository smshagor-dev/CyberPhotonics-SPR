"""Static topology/material controls for the optional 2D Femwell feasibility probe."""
from __future__ import annotations

import math

import pytest

pytest.importorskip("shapely")
from sprpcf.simulation.pcf_femwell_probe import (
    ExploratoryPcfTopology,
    fitted_gold_permittivity,
    silica_index_sellmeier,
)


def test_exploratory_pcf_partition_valid() -> None:
    topology = ExploratoryPcfTopology()
    regions = topology.build()
    assert set(("analyte", "gold", "silica")).issubset(regions)
    assert len([key for key in regions if key.startswith("hole_")]) == 6
    area = sum(shape.area for shape in regions.values())
    assert math.isclose(area, (2 * topology.domain_halfwidth_um) ** 2, abs_tol=1e-5)
    for polygon in regions.values():
        assert polygon.is_valid and polygon.area > 0
    assert all(x.intersection(y).area < 1e-9 for i, x in enumerate(regions.values())
               for j, y in enumerate(regions.values()) if j > i)


def test_exploratory_topology_rejects_overlaps_and_clipping() -> None:
    with pytest.raises(ValueError, match="too close"):
        ExploratoryPcfTopology(channel_x_um=1.65).build()
    with pytest.raises(ValueError, match="clips"):
        ExploratoryPcfTopology(domain_halfwidth_um=1.0).build()
    with pytest.raises(ValueError, match="invalid"):
        ExploratoryPcfTopology(gold_thickness_um=0.6).build()


def test_dispersion_source_is_passive_at_pilot_wavelength() -> None:
    n_silica = silica_index_sellmeier(0.8)
    gold = fitted_gold_permittivity(0.8)
    assert 1.44 < n_silica < 1.46
    assert gold.imag > 0 and gold.real < -(1.36**2)
    with pytest.raises(ValueError, match="0.45"):
        silica_index_sellmeier(0.1)
