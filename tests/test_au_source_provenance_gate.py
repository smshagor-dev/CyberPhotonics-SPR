"""Regression: measured Au selection must never self-certify provenance."""
import pytest
from sprpcf.simulation.open_vector_grid import au_source_provenance


@pytest.mark.parametrize("source,selected", [
    ("fitted", False),
    ("johnson_christy_1972", True),
])
def test_au_source_selection_is_not_independent_verification(source, selected):
    flags = au_source_provenance(source)
    assert flags["Au_reference_dataset_selected"] is selected
    assert flags["Au_material_measured_data_provenance_verified"] is False
    assert flags["Au_reference_transcription_independently_verified"] is False


def test_unknown_au_source_fails_closed():
    with pytest.raises(ValueError, match="unknown Au optical source"):
        au_source_provenance("untraceable")
