"""Constructed regression tests only; never physical sensor evidence."""
import pytest
from scripts.film_transfer_crosscheck import crosscheck, write_immutable


def test_seven_thicknesses_agree():
    result = crosscheck()
    assert result['matched'] == 7 and result['failed'] == 0
    assert max(row['absolute_neff_difference'] for row in result['rows']) < 1e-7
    assert not result['physics_qualified']
    assert not result['independent_fem_crosscheck']


def test_immutable_output(tmp_path):
    path = tmp_path / 'transfer.json'
    write_immutable(path, crosscheck((.1, .2)))
    with pytest.raises(FileExistsError):
        write_immutable(path, crosscheck((.1, .2)))
