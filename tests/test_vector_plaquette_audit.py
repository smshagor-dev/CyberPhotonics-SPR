"""Software-only graph fixtures; never interpret as electromagnetic evidence."""
from sprpcf.simulation.vector_plaquette_audit import audit_mode_plaquettes


def edges(*, swap_bottom=False, weak=False):
    result = []
    for axis, coordinates in (
        ("wavelength", [(0, 0), (1, 0)]),
        ("analyte_ri", [(0, 0), (0, 1)]),
    ):
        for i, j in coordinates:
            for mode in range(2):
                edge = {
                    "varying": axis,
                    "previous_mode_index": mode,
                    "current_mode_index": (
                        1 - mode if swap_bottom and axis == "wavelength" and i == 1 else mode
                    ),
                    "bidirectional_min_overlap_margin": (
                        -0.1 if weak and i == 0 and j == 0 and mode == 0 else 0.8
                    ),
                    "vector_eh_poynting_overlap_abs": 0.95,
                }
                if axis == "wavelength":
                    edge.update(ri_index=i, previous_wl_index=j, current_wl_index=j + 1)
                else:
                    edge.update(wavelength_index=j, previous_ri_index=i, current_ri_index=i + 1)
                result.append(edge)
    return result


def audit(records):
    return audit_mode_plaquettes(records, n_ri=2, n_wavelength=2, n_modes=2)


def test_commuting_square():
    result = audit(edges())
    assert result["graph_consistent"]
    assert result["physics_qualified"] is False


def test_noncommuting_assignment_detected():
    result = audit(edges(swap_bottom=True))
    assert len(result["noncommuting_plaquettes"]) == 2
    assert not result["graph_consistent"]


def test_weak_overlap_detected():
    assert not audit(edges(weak=True))["graph_consistent"]


def test_missing_edge_rejected():
    assert not audit(edges()[:-1])["graph_consistent"]


def test_duplicate_edge_rejected():
    records = edges()
    records.append(records[0])
    assert not audit(records)["graph_consistent"]


def test_many_to_one_mode_assignment_rejected_even_when_paths_commute():
    records = edges()
    # All four directed neighbors map both source modes onto target mode 0.
    # Both routes commute, but this is not a permutation or valid tracking.
    for record in records:
        record["current_mode_index"] = 0
    result = audit(records)
    assert not result["graph_consistent"]
    assert any("nonbijective mode assignment" in error for error in result["errors"])


def test_single_neighbor_mode_collapse_rejected():
    records = edges()
    for record in records:
        if record["varying"] == "wavelength" and record["ri_index"] == 0:
            record["current_mode_index"] = 0
    result = audit(records)
    assert not result["graph_consistent"]
    assert any("nonbijective mode assignment" in error for error in result["errors"])
