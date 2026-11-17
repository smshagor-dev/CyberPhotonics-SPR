"""Software/numerical diagnostics only; no publication physics qualification."""
import pytest

from research.planar_tm_domain_sensitivity import campaign


def test_domain_campaign_is_conservative():
    result = campaign()
    assert len(result["records"]) == 4
    assert result["last_pair_relative_beta_drift"] < 1e-5
    assert result["domain_convergence_diagnostic_only"] is True
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


@pytest.mark.parametrize("domains", [
    (1.6, 1.6, 3.2),
    (3.2, 2.4, 4.0),
    (1.6, float("nan"), 4.0),
    (1.6, -2.4, 4.0),
    (1.6, 2.4),
])
def test_invalid_domain_campaign_rejected(domains):
    with pytest.raises(ValueError):
        campaign(domains_um=domains)
